"""PT: método do p14_memory_r1 — Memory-R1 (arXiv 2508.19828).

EN: p14_memory_r1 method — Memory-R1.

Dois agentes treinados por RL:
1. **Memory Manager** — escolhe ADD/UPDATE/DELETE/NOOP para cada fato novo
   (state = features do fato + banco; recompensa = ganho de acurácia do QA).
2. **Answer Agent** — destila memória: dos ≤60 candidatos recuperados,
   escolhe os relevantes para responder (recompensa = exact match).

As políticas são logits lineares sobre features de token-overlap, treinadas
em numpy com PPO simplificado (clipping) e GRPO (vantagem relativa ao grupo:
reward - média do grupo). ``HeuristicManager`` é o baseline não-treinado
(regra fixa) e ``UntrainedManager`` faz tudo ADD.

Contrast with MemRL (p03): lá o RL põe valores de utilidade NAS memórias com
o LLM congelado; aqui o RL treina a política do GERENTE de memória (e do
agente que responde).
"""

from __future__ import annotations

import numpy as np

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.facts import extract_facts as _shared_extract
from papers.common.retrieval import VectorIndex
from papers.common.utils import token_set

OPS = ["ADD", "UPDATE", "DELETE", "NOOP"]
MAX_CANDIDATES = 60


def exact_match(gold: str, pred: str) -> float:
    """PT: recompensa binária de exact match (normalizada por tokens).
    EN: binary exact-match reward over token sets."""
    return float(token_set(gold) <= token_set(pred) and bool(token_set(gold)))


class LinearPolicy:
    """PT: política linear numpy softmax sobre features; treinável por
    REINFORCE/GRPO com vantagem. EN: linear softmax policy over features,
    trainable via REINFORCE/GRPO with advantage."""

    def __init__(self, n_features: int, n_actions: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.w = rng.normal(scale=0.01, size=(n_features, n_actions))

    def probs(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, x: np.ndarray, rng: np.random.RandomState) -> int:
        return int(rng.choice(len(self.w[0]), p=self.probs(x)))

    def update(self, x: np.ndarray, a: int, adv: float, lr: float) -> None:
        """PT: gradiente REINFORCE com vantagem (GRPO usa adv relativa ao
        grupo; PPO simplificado clipa a vantagem). EN: REINFORCE gradient
        with advantage (GRPO = group-relative adv; simplified PPO clips)."""
        p = self.probs(x)
        grad = -p
        grad[a] += 1.0
        # PT: REINFORCE ascendente: w += lr * adv * x (e_a - p).
        # EN: REINFORCE ascent: w += lr * adv * x (e_a - p).
        self.w += lr * adv * np.outer(x, grad)


_SLOTS = ["city", "job", "pet", "breakfast", "hobby", "other"]


def fact_features(fact: str, sims: list[str]) -> np.ndarray:
    """PT: features do estado do manager: [overlap máx, tem-similar?, overlap
    médio, bias, |fato|, é-remoção?, tem-conflito?, tem-duplicata?] +
    one-hot do slot — necessário para a política distinguir fatos novos de
    contradições. EN: manager state features + slot one-hot (needed so the
    policy can tell fresh facts from contradictions)."""
    ft = token_set(fact)
    ov = [len(ft & token_set(s)) / (len(ft | token_set(s)) or 1)
          for s in sims] or [0.0]
    is_removal = float(fact.startswith("remove:"))
    key = fact.rsplit(" is ", 1)[0]
    has_conflict = float(any(
        s.rsplit(" is ", 1)[0] == key and s.lower() != fact.lower()
        for s in sims))
    has_dup = float(any(s.lower() == fact.lower() for s in sims))
    slot = next((s for s in _SLOTS if f" {s} " in fact
                 or fact.startswith(f"remove: {s}")), "other")
    onehot = [float(s == slot) for s in _SLOTS]
    return np.array([max(ov), float(bool(sims)), float(np.mean(ov)),
                     1.0, min(len(ft) / 10.0, 1.0), is_removal,
                     has_conflict, has_dup, *onehot])


class MemoryR1Manager:
    """PT: manager do Memory-R1 — política sobre ops de memória.
    EN: Memory-R1 manager — policy over memory ops."""

    def __init__(self, embedder: Embedder | None = None, seed: int = 0) -> None:
        self._index = VectorIndex(embedder or HashingEmbedder())
        self.memories: dict[str, str] = {}
        self._n = 0
        self.policy = LinearPolicy(14, len(OPS), seed)
        self.rng = np.random.RandomState(seed + 1)
        self.greedy = False  # PT: True = argmax (política treinada/deploy).
        # EN: True = argmax (deployed trained policy).

    def _apply(self, op: int, fact: str) -> None:
        name = OPS[op]
        if name == "ADD":
            self._n += 1
            self.memories[f"m{self._n}"] = fact
            self._index.add(f"m{self._n}", fact)
        elif name == "UPDATE":
            # PT: substitui a memória de mesma chave (slot); sem conflito,
            # comporta-se como ADD. EN: replace the same-key memory; without
            # conflict, behaves as ADD.
            key = fact.rsplit(" is ", 1)[0]
            for d, t in list(self.memories.items()):
                if t.rsplit(" is ", 1)[0] == key:
                    del self.memories[d]
            self._n += 1
            self.memories[f"m{self._n}"] = fact
            self._index.add(f"m{self._n}", fact)
        elif name == "DELETE":
            if fact.startswith("remove:"):
                tgt = fact.split()[-1]
                for d, t in list(self.memories.items()):
                    if tgt in t:
                        del self.memories[d]
            else:
                # PT: só a mesma chave (mesmo slot) — não varre memórias
                # aleatórias. EN: same key (slot) only.
                key = fact.rsplit(" is ", 1)[0]
                for d, t in list(self.memories.items()):
                    if t.rsplit(" is ", 1)[0] == key:
                        del self.memories[d]

    def ingest(self, fact: str) -> tuple[int, np.ndarray]:
        """PT: escolhe a op pela política e aplica; devolve (ação, estado)
        para treinamento. EN: pick the op via policy and apply; return
        (action, state) for training."""
        sims = [self.memories[d]
                for d, _ in self._index.search(fact, 10)
                if d in self.memories]
        x = fact_features(fact, sims)
        a = (int(np.argmax(self.policy.probs(x))) if self.greedy
             else self.policy.act(x, self.rng))
        self._apply(a, fact)
        return a, x

    def retrieve(self, query: str, k: int = MAX_CANDIDATES) -> list[str]:
        return [self.memories[d]
                for d, _ in self._index.search(query, min(k, MAX_CANDIDATES))
                if d in self.memories][:k]


class HeuristicManager(MemoryR1Manager):
    """PT: baseline com regra fixa: contraditório→UPDATE, remove→DELETE,
    duplicata→NOOP, senão ADD. EN: fixed-rule baseline."""

    def ingest(self, fact: str) -> tuple[int, np.ndarray]:
        sims = [self.memories[d]
                for d, _ in self._index.search(fact, 10)
                if d in self.memories]
        x = fact_features(fact, sims)
        key = fact.rsplit(" is ", 1)[0]
        dup = any(s.lower() == fact.lower() for s in sims)
        confl = any(s.rsplit(" is ", 1)[0] == key and
                    s.lower() != fact.lower() for s in sims)
        if fact.startswith("remove:"):
            a = OPS.index("DELETE")
        elif confl:
            a = OPS.index("UPDATE")
        elif dup:
            a = OPS.index("NOOP")
        else:
            a = OPS.index("ADD")
        self._apply(a, fact)
        return a, x


class UntrainedManager(MemoryR1Manager):
    """PT: baseline não-treinado — sempre ADD (acumula contradições).
    EN: untrained baseline — always ADD (contradictions pile up)."""

    def ingest(self, fact: str) -> tuple[int, np.ndarray]:
        sims = [self.memories[d]
                for d, _ in self._index.search(fact, 10)
                if d in self.memories]
        x = fact_features(fact, sims)
        self._apply(OPS.index("ADD"), fact)
        return OPS.index("ADD"), x


def extract_facts(text: str) -> list[str]:
    """PT: delega ao extrator compartilhado (common.facts).
    EN: delegates to the shared extractor (common.facts)."""
    return _shared_extract(text)


def distill(candidates: list[str], question: str, k: int = 5) -> list[str]:
    """PT: Answer Agent — destilação: dos ≤60 candidatos retorna os k com
    maior overlap com a pergunta (policy não-treinada usa overlap; treinada
    reordena). EN: memory distillation — top-k by overlap with the question
    over ≤60 candidates."""
    q = token_set(question)
    scored = sorted(candidates, key=lambda c: len(q & token_set(c)),
                    reverse=True)
    return scored[:k]
