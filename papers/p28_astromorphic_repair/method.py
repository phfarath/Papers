"""Mecanismo central do paper p28 — auto-reparo astromórfico.

Core mechanism of p28 — astromorphic self-repair (Han et al., AAAI 2023).

Ideia: uma rede competitiva não supervisionada (WTA + Hebbian) organiza
neurônios por domínios de entrada. A glia mantém, por neurônio, uma
estimativa lenta do seu domínio (média das entradas que ele vence). Sob
injeção de falha (neurônios mortos), a glia realoca o domínio registrado
do morto para uma **unidade reserva** — o neurônio reserva entra na
competição já no domínio certo, mantendo a cobertura sem re-aprender do
zero.

Idea: an unsupervised competitive network (WTA + Hebbian) organizes
neurons by input domains. Glia keeps, per neuron, a slow estimate of its
domain (mean of inputs it wins). Under fault injection (dead neurons),
glia reallocates the dead unit's recorded domain to a **spare unit** —
the spare enters competition already at the right domain, preserving
coverage without relearning from scratch.
"""

import numpy as np


def make_clusters(seed, k=4, d=10, n=200, spread=0.4):
    rng = np.random.default_rng(seed)
    cents = rng.normal(0, 2.0, (k, d))
    lab = rng.integers(0, k, n)
    return cents[lab] + rng.normal(0, spread, (n, d)), lab, cents


class SelfRepairNet:
    """WTA Hebbian + memória glial de domínio + unidades reserva.

    WTA Hebbian net + glial domain memory + spare units.
    """

    def __init__(self, d, n_active=6, n_spare=3, repair=True, seed=0,
                 eta=0.05, dom_rate=0.02):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_tot = n_active + n_spare
        self.W = rng.normal(0, 0.1, (self.n_tot, d))
        self.m = np.zeros((self.n_tot, d))  # domínio registrado pela glia
        self.alive = np.zeros(self.n_tot, bool)
        self.alive[:n_active] = True
        self.spares = list(range(n_active, self.n_tot))
        self.repair = repair
        self.eta, self.dom_rate = eta, dom_rate
        self.n_active = n_active

    def step(self, x):
        """Um passo: neurônio vivo mais próximo vence e aprende; glia registra."""
        idx = np.where(self.alive)[0]
        j = idx[np.argmax(self.W[idx] @ x / (np.linalg.norm(self.W[idx], axis=1) + 1e-9))]
        self.W[j] += self.eta * (x - self.W[j])
        self.m[j] += self.dom_rate * (x - self.m[j])
        return j

    def inject_fault(self, n_kill):
        """Mata `n_kill` neurônios ativos; glia realoca domínios a reservas.

        Kill `n_kill` active neurons; glia reallocates their recorded
        domains to spare units. Returns the victim indices.
        """
        victims = np.where(self.alive[: self.n_active])[0][-n_kill:]
        self.alive[victims] = False
        if self.repair:
            for v in victims:
                if not self.spares:
                    break
                s = self.spares.pop(0)
                self.W[s] = self.m[v].copy() + self.rng.normal(0, 0.05, self.W.shape[1])
                self.m[s] = self.m[v].copy()
                self.alive[s] = True
        return victims

    def coverage(self, cents):
        """cos médio do neurônio vivo mais próximo de cada centroide."""
        W = self.W[self.alive]
        return float(
            np.mean(
                [
                    max(
                        float(c @ w / (np.linalg.norm(c) * np.linalg.norm(w) + 1e-9))
                        for w in W
                    )
                    for c in cents
                ]
            )
        )


def run(seed, repair, epochs=4, fault_at=0.5, n_kill=3, **kw):
    """Stream de entradas clusterizadas; falha no meio; cobertura por passo."""
    X, _, cents = make_clusters(seed)
    net = SelfRepairNet(X.shape[1], repair=repair, seed=seed + 1, **kw)
    T = epochs * len(X)
    kill = int(T * fault_at)
    hist = []
    for t in range(T):
        net.step(X[t % len(X)])
        if t == kill:
            net.inject_fault(n_kill)
        hist.append(net.coverage(cents))
    return np.array(hist), cents


def recovery_steps(hist, kill_t, thr=0.9):
    """Passos até cobertura >= thr após a falha (ou infinito)."""
    after = hist[kill_t:]
    idx = np.where(after >= thr)[0]
    return float(idx[0]) if len(idx) else float(len(after))
