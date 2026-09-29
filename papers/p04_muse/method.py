"""MUSE (Valiente & Pilly) — implementação LLM (seção 4 do paper).

PT: Pré-deployment: roda o agente Reflexion (p02) nas tarefas in-distribution
para coletar dados e treina o Self-Assessment Model M_sa = embedder + MLP
g_η (sigmoid, BCE — Eq. 7) sobre concat(emb(τ_chunk, P^e), emb(I)), com
trajetórias segmentadas em chunks NÃO sobrepostos de 4 pares ação-observação
(Algoritmo 2). Deployment: a cada passo o World Model/Actor gera 5 rollouts
hipotéticos (temperatura 0.5), M_sa pontua cada um e o agente executa a 1ª ação
do melhor rollout (self-regulation, §4.1.3); reflexões em falhas; adaptação
online só de g_η com replay buffer.

SFT do actor e DPO do reflection LLM: construímos os datasets exatamente como o
paper define (SFT = episódios de sucesso; DPO: par positivo = falha em e_i e
sucesso em e_{i+1} após a reflexão; negativo = o inverso), salvos em JSONL —
com MockLLM não há pesos para treinar, então só geramos os dados.

EN: see PT. Builds the SFT/DPO datasets exactly as defined in the paper and
saves them as JSONL — with a MockLLM there are no weights to train.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM, task_prompt
from papers.envs.household import HouseholdEnv, TaskSpec, score_actions
from papers.p02_reflexion.prompts import reflection_body
from papers.p04_muse.prompts import first_action

CHUNK = 4           # PT: pares ação-observação por chunk. EN: pairs per chunk.
N_ROLLOUTS = 5      # PT: 5 rollouts hipotéticos. EN: 5 hypothetical rollouts.
FAIL_TTC = 100      # PT: penalidade de ttc em falha. EN: failure ttc penalty.


# ---------------------------------------------------------------------------
# Self-Assessment Model M_sa = embedder + MLP g_η (sigmoid, BCE)
# ---------------------------------------------------------------------------

class SelfAssessment:
    """PT: g_η é um MLP real (1 camada escondida tanh, saída sigmoid, BCE —
    Eq. 6/7) sobre concat(emb(τ_chunk+P^e), emb(I)). Backprop em numpy;
    inicialização com seed fixa.

    EN: g_η is a real MLP (1 hidden tanh layer, sigmoid output, BCE — Eq. 6/7)
    over concat(emb(chunk+plan), emb(task)). Numpy backprop; fixed-seed init.
    """

    def __init__(self, embedder: Embedder | None = None, lr: float = 0.5,
                 hidden: int = 64, seed: int = 0) -> None:
        self.emb = embedder or HashingEmbedder()
        self.lr = lr
        self.hidden = hidden
        self._rng = np.random.default_rng(seed)
        self.w1: np.ndarray | None = None  # (in_dim, hidden)
        self.b1: np.ndarray | None = None  # (hidden,)
        self.w2: np.ndarray | None = None  # (hidden,)
        self.b2 = 0.0

    # PT: compatibilidade com código que checa "treinado?" — w era o vetor
    # logístico; agora expomos .w como alias de w2.
    @property
    def w(self) -> np.ndarray | None:
        return self.w2

    @w.setter
    def w(self, value: np.ndarray | None) -> None:
        self.w2 = value

    def _x(self, traj_text: str, task: str) -> np.ndarray:
        e = self.emb.embed([traj_text, task])
        return np.concatenate([e[0], e[1]])

    def _ensure_init(self, in_dim: int) -> None:
        if self.w1 is None:
            scale = 1.0 / np.sqrt(in_dim)
            self.w1 = self._rng.normal(0.0, scale, (in_dim, self.hidden))
            self.b1 = np.zeros(self.hidden)
            self.w2 = self._rng.normal(0.0, 0.1, self.hidden)
            self.b2 = 0.0

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """PT: h=tanh(xW1+b1); p=σ(h·w2+b2). EN: forward pass."""
        h = np.tanh(x @ self.w1 + self.b1)  # type: ignore[operator]
        z = h @ self.w2 + self.b2  # type: ignore[operator]
        return h, float(1 / (1 + np.exp(-z)))

    def predict(self, traj_text: str, task: str) -> float:
        if self.w2 is None:
            return 0.5
        return self._forward(self._x(traj_text, task))[1]

    def train_step(self, traj_text: str, task: str, y: float) -> float:
        x = self._x(traj_text, task)
        self._ensure_init(x.shape[0])
        h, p = self._forward(x)
        # PT: BCE+sigmoid → dz = p − y; backprop pela camada tanh.
        # EN: BCE+sigmoid → dz = p − y; backprop through the tanh layer.
        dz = p - y
        dw2 = dz * h
        dh = dz * self.w2  # type: ignore[operator]
        dh *= 1 - h * h  # tanh'
        self.w2 -= self.lr * dw2  # type: ignore[operator]
        self.b2 -= self.lr * dz
        self.w1 -= self.lr * np.outer(x, dh)  # type: ignore[operator]
        self.b1 -= self.lr * dh  # type: ignore[operator]
        return float(p)


def chunk_pairs(traj: list[tuple[str, str]], plan: str, task: str,
                success: bool) -> list[tuple[str, str, float]]:
    """PT: segmenta a trajetória em chunks de CHUNK pares (não sobrepostos),
    rotulados pelo desfecho do episódio — como no Algoritmo 2.

    EN: segments the trajectory into non-overlapping CHUNK-pair chunks labeled
    by the episode outcome — per Algorithm 2.
    """
    out = []
    for i in range(0, len(traj) - CHUNK + 1, CHUNK):
        txt = plan + " " + " ".join(f"{a} {o.split(' You are in')[0]}"
                                    for a, o in traj[i:i + CHUNK])
        out.append((txt, task, 1.0 if success else 0.0))
    return out


# ---------------------------------------------------------------------------
# Agente MUSE (deployment)
# ---------------------------------------------------------------------------

@dataclass
class MuseAgent:
    llm: LLM
    sa: SelfAssessment
    reflections: list[str] = field(default_factory=list)
    use_msa: bool = True      # ablação: False ≈ ReAct+reflexão sem M_sa
    replay: list[tuple[str, str, float]] = field(default_factory=list)

    def plan(self, task: str) -> str:
        return self.llm.complete(task_prompt(
            "muse.plan", f"Write a short plan.\nTask: {task}")).strip()

    def _rollout(self, spec: TaskSpec, obs: str, actions: list[str]) -> str:
        body = (
            "Imagine a plausible next-steps trajectory.\n"
            f"Task: {spec.description}\nObservation: {obs}\n"
            + ("Reflections:\n" + "\n".join(f"- {m}" for m in self.reflections)
               + "\n" if self.reflections else "")
            + "Valid actions:\n" + "\n".join(f"- {a}" for a in actions)
        )
        return self.llm.complete(task_prompt("muse.rollout", body),
                                 temperature=0.5)

    def act(self, spec: TaskSpec, obs: str, actions: list[str],
            plan: str, recent: list[str]) -> str:
        if not self.use_msa:
            return score_actions(spec.description, obs, actions,
                                 self.reflections, recent, random.Random(0))
        # PT: 5 rollouts → M_sa pontua → executa a 1ª ação do melhor.
        # EN: 5 rollouts → M_sa scores → execute best rollout's 1st action.
        best_a, best_p = None, -1.0
        for _ in range(N_ROLLOUTS):
            ro = self._rollout(spec, obs, actions)
            a = first_action(ro)
            if a not in actions:
                continue
            p = self.sa.predict(ro + " " + plan, spec.description)
            if p > best_p:
                best_p, best_a = p, a
        if best_a is None:
            best_a = score_actions(spec.description, obs, actions,
                                   self.reflections, recent, random.Random(0))
        return best_a

    def run_episode(self, spec: TaskSpec) -> tuple[bool, list[tuple[str, str]]]:
        env = HouseholdEnv(spec)
        obs = env.reset()
        plan = self.plan(spec.description)
        traj: list[tuple[str, str]] = []
        recent: list[str] = []
        while True:
            va = env.valid_actions()
            act = self.act(spec, obs, va, plan, recent)
            res = env.step(act)
            traj.append((act, res.observation))
            recent.append(act)
            obs = res.observation
            if res.done:
                break
        success = env.done
        # PT: reflexão em falhas (§4.1.3). EN: reflect on failures.
        if not success:
            body = reflection_body(spec.description, traj, "failure",
                                   self.reflections)
            self.reflections = (self.reflections + [
                self.llm.complete(task_prompt("reflexion.reflect", body))])[-3:]
        # PT: dados novos entram no replay buffer p/ adaptação online de g_η.
        # EN: new data goes to the replay buffer for online g_η adaptation.
        self.replay += chunk_pairs(traj, plan, spec.description, success)
        return success, traj


def adapt_online(sa: SelfAssessment,
                 replay: list[tuple[str, str, float]], epochs: int = 3,
                 rng: random.Random | None = None) -> None:
    """PT: adaptação online de g_η com replay buffer (Algoritmo 4).

    EN: online adaptation of g_η only, over the replay buffer (Algorithm 4).
    """
    rng = rng or random.Random(0)
    for _ in range(epochs):
        for txt, task, y in replay:
            sa.train_step(txt, task, y)


# ---------------------------------------------------------------------------
# Datasets SFT / DPO (definições do paper; treino de pesos não roda offline)
# ---------------------------------------------------------------------------

def build_sft_dataset(
    episodes: list[tuple[TaskSpec, bool, list[tuple[str, str]]]],
) -> list[dict[str, object]]:
    """PT: SFT = episódios de sucesso. EN: SFT = successful episodes only."""
    return [{"task": s.task_id, "trajectory": [a for a, _ in traj]}
            for s, ok, traj in episodes if ok]


def build_dpo_dataset(
    episodes: list[tuple[TaskSpec, bool, list[tuple[str, str]], str]],
) -> list[dict[str, object]]:
    """PT: par DPO (chosen, rejected) por tarefa: reflexão positiva = falha em
    e_i e sucesso em e_{i+1}; negativa = o inverso.

    EN: DPO pair per task: positive reflection = fail e_i then success e_{i+1};
    negative = the inverse.
    """
    by_task: dict[str, list[tuple[bool, str]]] = {}
    for spec, ok, _traj, refl in episodes:
        by_task.setdefault(spec.task_id, []).append((ok, refl))
    out = []
    for task_id, eps in by_task.items():
        pos = neg = None
        for i in range(len(eps) - 1):
            if not eps[i][0] and eps[i + 1][0]:
                pos = eps[i][1]
            if eps[i][0] and not eps[i + 1][0]:
                neg = eps[i][1]
        if pos and neg:
            out.append({"task": task_id, "chosen": pos, "rejected": neg})
    return out


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Métricas / metrics
# ---------------------------------------------------------------------------

def auroc(scores: list[float], labels: list[int]) -> float:
    """PT: AUROC simples por pares comparáveis. EN: pairwise AUROC."""
    pos = [s for s, lbl in zip(scores, labels, strict=True) if lbl]
    neg = [s for s, lbl in zip(scores, labels, strict=True) if not lbl]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
