"""Mecanismo central do paper p24 — memória situacional neuromórfica.

Core mechanism of p24 — situation-based neuromorphic memory
(Gordleeva et al. 2025).

Ideia: memória organizada em pools de padrões — um "item" (cue) só tem
resposta correta dentro de uma **situação** (contexto). A rede guarda
unidades de situação s_{k,c} que só disparam quando o cue E o contexto
casam; cada unidade aponta para a resposta y_{k,c}. A glia mantém a
situação corrente: uma variável astroglial lenta faz "latch" do contexto
na codificação e o segura quando a pista contextual some — sem ela, cues
ambíguos recuperam respostas misturadas (Hebbian plano).

Idea: memory organized in pattern pools — a cue has a correct response
only inside a **situation** (context). The network stores situation
units s_{k,c} firing only when BOTH cue and context match; each points
to response y_{k,c}. Glia holds the current situation: a slow astrocytic
variable latches the context at encoding and keeps it when the
contextual cue disappears — without it, ambiguous cues retrieve mixed
responses (flat Hebbian).
"""

import numpy as np

THETA = 0.8  # limiar de ativação da unidade de situação


class SituationMemory:
    """Pooled situation memory: cue × context → response.

    Pools: cues x_k ∈ {±1}^n, contexts c ∈ {±1}^m, responses y_{k,c}.
    Situation units s_{k,c} read [x | context] and gate the readout.
    The astrocyte latches the context vector and decays with tau_ctx.
    """

    def __init__(self, seed=0, n_cue=8, x_dim=24, ctx_dim=8, y_dim=16, tau_ctx=2.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.x_dim, self.ctx_dim, self.y_dim = x_dim, ctx_dim, y_dim
        self.cues = rng.choice([-1, 1], (n_cue, x_dim))
        self.ctxs = rng.choice([-1, 1], (2, ctx_dim))
        self.Y = rng.choice([-1, 1], (n_cue, 2, y_dim))
        self.tau_ctx = tau_ctx
        # units de situação: metade cue, metade contexto
        self.S = np.zeros((n_cue * 2, x_dim + ctx_dim))
        self.out = np.zeros((n_cue * 2, y_dim))
        for k in range(n_cue):
            for c in range(2):
                j = 2 * k + c
                self.S[j, :x_dim] = self.cues[k]
                self.S[j, x_dim:] = self.ctxs[c]
                self.out[j] = self.Y[k, c]
        # baseline Hebbian plano cue→y (mistura contextos)
        self.W_flat = (
            sum(np.outer(self.Y[k, c], self.cues[k]) for k in range(n_cue) for c in range(2))
            / (2 * n_cue)
        )
        self.astro = np.zeros(ctx_dim)  # latch contextual da glia

    def encode(self, k, c, dur=0.5, dt=0.01):
        """Apresenta cue+contexto; a glia integra o contexto (latch).

        Present cue+context; glia integrates the context (latch).
        """
        for _ in range(int(dur / dt)):
            self.astro += dt * (-self.astro + self.ctxs[c]) / (self.tau_ctx * 0.2)

    def decay_context(self, gap, dt=0.01):
        """Intervalo sem pista contextual: o latch glial decai com tau_ctx."""
        for _ in range(int(gap / dt)):
            self.astro += dt * (-self.astro) / self.tau_ctx

    def _gate(self, a, theta=THETA):
        return np.maximum(0.0, a - theta)

    def recall_gated(self, x, ctx):
        """Recall com situação: fonte do contexto = ctx vivo ou latch glial."""
        inp = np.concatenate([x, ctx])
        a = self.S @ inp / (self.S.shape[1] / 2)
        yhat = self.out.T @ self._gate(a)
        return np.sign(yhat + 1e-12)

    def recall_flat(self, x):
        """Baseline sem glia/contexto: Hebbian plano mistura as respostas."""
        return np.sign(self.W_flat @ x)

    def recall_latched(self, x):
        """Recall usando o contexto retido pela glia (pista removida)."""
        return self.recall_gated(x, self.astro)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gated_vs_flat(mem):
    """cos(resposta, y_correto) por cue/contexto: gated vs flat Hebbian."""
    fg, ff = [], []
    for k in range(mem.cues.shape[0]):
        for c in range(2):
            fg.append(cosine(mem.recall_gated(mem.cues[k], mem.ctxs[c]), mem.Y[k, c]))
            ff.append(cosine(mem.recall_flat(mem.cues[k]), mem.Y[k, c]))
    return float(np.mean(fg)), float(np.mean(ff))


def context_persistence(mem, gap, dt=0.01):
    """cos médio ao recordar após `gap` sem pista contextual.

    Mean cosine when recalling after `gap` without the contextual cue —
    relies on the glial latch. Resets the latch per item.
    """
    cs = []
    for k in range(mem.cues.shape[0]):
        for c in range(2):
            mem.encode(k, c)
            mem.decay_context(gap)
            cs.append(cosine(mem.recall_latched(mem.cues[k]), mem.Y[k, c]))
            mem.astro[:] = 0.0
    return float(np.mean(cs))
