"""Mecanismo central do paper p27 — autoatenção emergente via replicator.

Core mechanism of p27 — emergent self-attention from astrocyte-gated
replicator dynamics (Vivet & Arenas 2026).

Ideia: alocação de "atenção" emerge de uma dinâmica replicadora sobre os
ganhos: `g_{t+1} = g_t · e^{β·f_i} / Z`, onde cada item multiplica seu
ganho pelo seu fitness e a glia fornece a normalização global Z (o
denominador — como no p21). Analiticamente, `g_t = softmax(β·t·f)`: a
dinâmica multiplicativa local + normalização global produz a alocação
softmax — winner-take-all no limite βt→∞.

Idea: attention-like allocation emerges from replicator dynamics over
gains: `g_{t+1} = g_t · e^{β·f_i} / Z`, where each item multiplies its
gain by its fitness and glia supplies the global normalization Z (the
denominator — as in p21). Analytically `g_t = softmax(β·t·f)`: local
multiplicative dynamics + global normalization produce the softmax
allocation — winner-take-all as βt→∞.
"""

import numpy as np


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


class ReplicatorAttention:
    """Ganhos no simplex sob dinâmica replicadora exponenciada.

    Simplex gains under exponentiated replicator dynamics.
    `glia=False` removes the global normalization (ablation).
    """

    def __init__(self, fitness, beta=1.0, glia=True, g0=None):
        self.f = np.asarray(fitness, float)
        self.beta = beta
        self.glia = glia
        self.g = np.ones_like(self.f) / len(self.f) if g0 is None else g0.copy()

    def step(self):
        ex = np.exp(self.beta * self.f)
        z = (self.g * ex).sum() if self.glia else 1.0
        self.g = self.g * ex / z
        return self.g

    def run(self, steps):
        hist = [self.g.copy()]
        for _ in range(steps):
            hist.append(self.step().copy())
        return np.array(hist)


def entropy(g):
    g = np.clip(g, 1e-12, 1.0)
    return float(-(g * np.log(g)).sum())


def track_vs_theory(fitness, beta=1.0, steps=30, seed=0):
    """Distância de g_t ao softmax teórico softmax(β·t·f). / Distance to theory."""
    att = ReplicatorAttention(fitness, beta)
    hist = att.run(steps)
    errs = [
        np.abs(hist[t] - softmax(beta * t * np.asarray(fitness))).max()
        for t in range(steps + 1)
    ]
    return np.array(errs), hist
