"""Mecanismo central do paper p29 — LSM com unidades astrocíticas lentas.

Core mechanism of p29 — Liquid State Machine with astrocyte-like units
(Yang et al. 2026).

Ideia: um reservatório de neurônios rápidos processa a série temporal;
unidades astrogliais LENTAS (τ_a >> τ_n) integram a atividade do
reservatório e entram como features extras no readout — carregam a
memória de longa escala que os neurônios rápidos perdem. O paper
reporta um ótimo na razão astrócito:neurônio (aqui, em torno de ~0.5):
poucas não adicionam memória suficiente, demais diluem/overfitam o
readout.

Idea: a reservoir of fast neurons processes the time series; SLOW
astrocyte units (τ_a >> τ_n) integrate reservoir activity and enter the
readout as extra features — carrying the long-timescale memory fast
neurons lose. The paper reports an optimum in the astrocyte:neuron
ratio (here ~0.5): too few add little memory, too many dilute/overfit
the readout.
"""

import numpy as np


def mackey_glass(n, dt=0.1, tau=17, beta=0.2, gamma=0.1, seed=0):
    """Série Mackey-Glass clássica (caos com atraso). / Classic MG chaos."""
    lag = int(tau / dt)
    x = np.full(lag + 1, 1.2)
    out = np.zeros(n)
    for t in range(n):
        out[t] = x[-1]
        dx = beta * x[0] / (1 + x[0] ** 10) - gamma * x[-1]
        x = np.append(x[1:], x[-1] + dx * dt)
    return out


class LiquidAstro:
    """Reservatório tanh + unidades lentas astrocíticas de readout.

    tanh reservoir + slow astrocyte readout units.
    """

    def __init__(self, n=50, ratio=1.0, tau_a=80.0, g_in=0.6, spec=0.6, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.m = int(n * ratio)
        self.tau_a = tau_a
        self.Win = rng.uniform(-g_in, g_in, (n, 1))
        W = rng.uniform(-1, 1, (n, n))
        W *= spec / np.abs(np.linalg.eigvals(W)).max()
        self.W = W
        self.Wna = rng.uniform(-0.5, 0.5, (self.m, n)) if self.m else None
        self.r = np.zeros(n)
        self.a = np.zeros(self.m)

    def reset(self):
        self.r[:] = 0.0
        self.a[:] = 0.0

    def step(self, inp):
        """Avança a dinâmica; retorna features [r | a] para o readout."""
        self.r = np.tanh(self.W @ self.r + self.Win[:, 0] * inp)
        if self.m:
            self.a += (-self.a + np.tanh(self.Wna @ self.r)) / self.tau_a
        return np.concatenate([self.r, self.a])


def collect(net, s, t0, t1):
    """Roda o reservatório sobre s[t0:t1]; retorna matriz de features."""
    return np.array([net.step(s[t]) for t in range(t0, t1)])


def train_readout(F, tgt, lam=1e-2):
    return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ tgt)


def evaluate(seed, ratio, tr=800, te=800, hor=60, **kw):
    """Treina readout em s[t0:tr]→s[t+hor]; NRMSE no segmento de teste."""
    s = mackey_glass(tr + te + hor + 50, seed=seed)
    s = (s - s.mean()) / s.std()
    net = LiquidAstro(ratio=ratio, seed=seed, **kw)
    F = collect(net, s, 0, tr)
    w = train_readout(F, s[hor : hor + tr])
    net.reset()
    Fte = collect(net, s, tr, tr + te)
    pred = Fte @ w
    true = s[tr + hor : tr + hor + te]
    return float(np.sqrt(np.mean((pred - true) ** 2)) / true.std())
