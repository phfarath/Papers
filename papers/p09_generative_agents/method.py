"""PT: método do p09_generative_agents — Mini-Smallville (arXiv 2304.03442).

EN: p09_generative_agents method — Mini-Smallville.

Memory stream + retrieval score = w_rec·recency + w_imp·importance +
w_rel·relevance (pesos todos = 1, componentes min-max normalizados, recency =
0.995^horas). Reflexão dispara quando a importância acumulada > 150 → 3
perguntas sobre as últimas ~100 memórias → 5 insights com citações.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM, task_prompt
from papers.p09_generative_agents import prompts  # noqa: F401

RECENCY_DECAY = 0.995          # PT: por hora. EN: per hour.
REFLECT_THRESHOLD = 150        # PT: importância acumulada. EN: cumulative imp.
REFLECT_LAST_N = 100           # PT: últimas memórias p/ perguntas. EN: last ~100.
N_INSIGHTS = 5
N_QUESTIONS = 3


@dataclass
class Memory:
    text: str
    importance: int            # PT: 1–10. EN: 1–10.
    created: float             # PT: hora do tick. EN: tick hour.
    last_access: float = 0.0
    kind: str = "obs"          # PT: obs|reflection|plan. EN: obs|reflection|plan.
    mid: str = ""


class MemoryStream:
    """PT: stream de memórias com retrieval CoLAU do paper.
    EN: memory stream with the paper's retrieval scoring."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._emb = embedder or HashingEmbedder()
        self.mems: list[Memory] = []
        self._n = 0
        self.since_reflect = 0

    def add(self, llm: LLM, text: str, now: float,
            kind: str = "obs") -> Memory:
        imp = int(llm.complete(task_prompt(
            "ga.importance", f"Memory: {text}")) or "1")
        imp = max(1, min(10, imp))
        self._n += 1
        m = Memory(text, imp, now, now, kind, f"mem{self._n}")
        self.mems.append(m)
        self.since_reflect += imp
        return m

    def retrieve(self, query: str, now: float, k: int = 5) -> list[Memory]:
        """PT: score = recency + importance + relevance (todos peso 1, min-max
        normalizados). EN: weighted sum, all weights 1, min-max normalized."""
        if not self.mems:
            return []
        rec = np.array([RECENCY_DECAY ** (now - m.last_access)
                        for m in self.mems])
        imp = np.array([m.importance for m in self.mems], dtype=float)
        texts = [m.text for m in self.mems]
        rel = (self._emb.embed(texts) @ self._emb.embed([query]).T).ravel()

        def mm(x: np.ndarray) -> np.ndarray:
            lo, hi = float(x.min()), float(x.max())
            return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)

        score = 1.0 * mm(rec) + 1.0 * mm(imp) + 1.0 * mm(rel)
        top = np.argsort(-score)[:k]
        for i in top:
            self.mems[int(i)].last_access = now
        return [self.mems[int(i)] for i in top]


@dataclass
class Agent:
    name: str
    persona: str
    stream: MemoryStream = field(default_factory=MemoryStream)
    plan: str = ""
    knows_party: bool = False
    n_reflections: int = 0

    def maybe_reflect(self, llm: LLM, now: float) -> None:
        """PT: reflexão quando a importância acumulada passa de 150: 3
        perguntas sobre as últimas ~100 memórias, retrieve por pergunta, 5
        insights com citações guardados como memórias 'reflection'.
        EN: reflection when cumulative importance >150: 3 questions over the
        last ~100 memories, retrieve per question, 5 cited insights stored
        as reflections."""
        if self.stream.since_reflect <= REFLECT_THRESHOLD:
            return
        self.stream.since_reflect = 0
        recent = "\n".join(f"- {m.text}" for m in self.stream.mems
                           [-REFLECT_LAST_N:])
        qs = llm.complete(task_prompt(
            "ga.reflect_q", f"Recent memories:\n{recent}"))
        for q in [ln[2:] for ln in qs.splitlines()
                  if ln.startswith("- ")][:N_QUESTIONS]:
            rel = self.stream.retrieve(q, now, k=10)
            cited = "\n".join(f"[{m.mid}] {m.text}" for m in rel)
            ins = llm.complete(task_prompt(
                "ga.insight", f"Question: {q}\nMemories:\n{cited}"))
            for ln in ins.splitlines():
                if ln.startswith("- "):
                    self.stream.add(llm, ln[2:], now, kind="reflection")
                    self.n_reflections += 1
                    break  # PT: 1 insight por pergunta aqui. EN: 1 per question.


@dataclass
class Smallville:
    """PT: mini-Smallville: ticks de 1h; agentes planejam, se encontram,
    conversam — e a informação 'party' se difunde. EN: mini-Smallville:
    hourly ticks; agents plan, meet, talk — party info diffuses."""
    agents: list[Agent]
    use_planning: bool = True
    use_reflection: bool = True
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def tick(self, llm: LLM, now: float) -> None:
        for a in self.agents:
            if self.use_planning and not a.plan:
                a.plan = llm.complete(task_prompt(
                    "ga.plan", f"Agent: {a.name}\nPersona: {a.persona}\n"
                               f"Hour: {int(now) % 24}"))
            if self.use_reflection:
                a.maybe_reflect(llm, now)
        # PT: encontros — com planejamento os agentes saem mais (plaza/walk).
        # EN: encounters — planning makes agents go out more (plaza/walk).
        p_meet = 0.5 if self.use_planning else 0.15
        aware = [a for a in self.agents if a.knows_party]
        unaware = [a for a in self.agents if not a.knows_party]
        for a in aware:
            for b in unaware:
                if self.rng.random() < p_meet:
                    mems = a.stream.retrieve("party invitation", now, k=3)
                    out = llm.complete(task_prompt(
                        "ga.dialogue",
                        f"Speaker: {a.name}\nSpeaker memories:\n"
                        + "\n".join(m.text for m in mems)))
                    a.stream.add(llm, f"Talked to {b.name}: {out}", now)
                    b.stream.add(llm, f"Talked to {a.name}: {out}", now)
                    if "party" in out.lower():
                        b.knows_party = True
                        b.stream.add(llm, "Invited to the party!", now)


def run_simulation(llm: LLM, ticks: int, use_planning: bool,
                   use_reflection: bool, seed: int = 0) -> list[int]:
    """PT: 4 agentes; agente 0 recebe a informação 'party' no tick 0.
    Devolve #agentes sabendo por tick. EN: 4 agents; agent 0 seeded with the
    party info at tick 0; returns aware-count per tick."""
    agents = [Agent(n, p) for n, p in
              [("Ana", "baker"), ("Bruno", "gardener"),
               ("Clara", "librarian"), ("Diego", "teacher")]]
    ville = Smallville(agents, use_planning, use_reflection,
                       random.Random(seed))
    agents[0].knows_party = True
    agents[0].stream.add(llm, "Invited to Isabella's party!", 0.0)
    curve = []
    for t in range(ticks):
        ville.tick(llm, float(t))
        curve.append(sum(a.knows_party for a in agents))
    return curve
