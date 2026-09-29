"""MemRL (MemTensor, 2026) — Intent-Experience-Utility triplets + Two-Phase
Retrieval + Monte Carlo utility update.

PT: Memória = triplas (z_i intent, e_i experience, Q_i utility).
- Fase A (Eq. de recall): C(s) = TopK_{k1}({i | cos(Emb(s),Emb(z_i)) > δ});
  se vazio, só o LLM (sem memória).
- Fase B (Eq. de seleção): score = (1−λ)·ẑ(sim) + λ·ẑ(Q), top-k2 — o z-score
  equilibra relevância semântica e utilidade aprendida.
- Atualização: Monte Carlo Q ← Q + α(r − Q) (Eq. 4) apenas nas memórias
  injetadas; opção TD (Eq. 3) via `td=True` com γ e max Q(s',·) do próximo
  estado. Após cada trajetória, um resumo (LLM) vira nova tripla (z, e, Q_init).

Defaults (não publicados no repo oficial — escolha nossa, documentada):
δ=0.25, λ=0.5 (melhor ponto da ablação do paper, Fig. 5), α=0.3, Q_init=0.5,
k1=4, k2=2, γ=0.9.

EN: see PT above. Defaults are our documented choice (the official repo does
not publish them); λ=0.5 matches the paper's ablation optimum (Fig. 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.utils import zscore

DELTA, LAM, ALPHA, Q_INIT, K1, K2, GAMMA = 0.25, 0.5, 0.3, 0.5, 4, 2, 0.9


@dataclass
class Triplet:
    """PT: (z intent, e experience, Q utility). EN: (z, e, Q) triplet."""

    z: str
    e: str
    q: float = Q_INIT


@dataclass
class MemRLMemory:
    """PT: banco de memórias MemRL com recuperação em duas fases.

    EN: MemRL memory bank with two-phase retrieval.
    """

    embedder: Embedder = field(default_factory=HashingEmbedder)
    delta: float = DELTA
    lam: float = LAM
    k1: int = K1
    k2: int = K2
    items: list[Triplet] = field(default_factory=list)

    def add(self, z: str, e: str, q: float = Q_INIT) -> None:
        self.items.append(Triplet(z, e, q))

    # ---------- Fase A ----------
    def phase_a(self, query: str) -> tuple[list[int], np.ndarray]:
        """PT: C(s) = TopK_{k1}({i | sim > δ}). EN: similarity-gated recall."""
        if not self.items:
            return [], np.zeros(0)
        zmat = self.embedder.embed([t.z for t in self.items])
        sims = (zmat @ self.embedder.embed([query]).T).ravel()
        keep = [(i, s) for i, s in enumerate(sims) if s > self.delta]
        keep.sort(key=lambda x: x[1], reverse=True)
        idx = [i for i, _ in keep[: self.k1]]
        return idx, sims

    # ---------- Fase B ----------
    def phase_b(self, query: str) -> list[int]:
        """PT: score = (1−λ)·ẑ(sim) + λ·ẑ(Q) sobre C(s), top-k2.

        EN: composite z-scored selection over C(s), top-k2.
        Returns indices of injected memories (for the utility update).
        """
        idx, sims = self.phase_a(query)
        if not idx:
            return []
        z_sim = zscore([float(sims[i]) for i in idx])
        z_q = zscore([self.items[i].q for i in idx])
        scored = [(i, (1 - self.lam) * zs + self.lam * zq)
                  for i, zs, zq in zip(idx, z_sim, z_q, strict=True)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored[: self.k2]]

    def retrieve(self, query: str) -> tuple[list[str], list[int]]:
        idx = self.phase_b(query)
        return [self.items[i].e for i in idx], idx

    # ---------- atualização / update ----------
    def update(self, idx: list[int], reward: float, alpha: float = ALPHA,
               td: bool = False, next_max_q: float = 0.0,
               gamma: float = GAMMA) -> None:
        """PT: MC (Eq.4): Q←Q+α(r−Q); TD (Eq.3): Q←Q+α(r+γ·maxQ'−Q).

        EN: Monte Carlo (Eq.4) or TD (Eq.3) update on injected memories only.
        """
        for i in idx:
            q = self.items[i].q
            target = reward + (gamma * next_max_q if td else 0.0)
            self.items[i].q = q + alpha * (target - q)
