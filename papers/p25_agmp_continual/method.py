"""Mecanismo central do paper p25 — metaplasticidade guiada por astrócito.

Core mechanism of p25 — astrocyte-gated metaplasticity (AGMP) for
continual learning (Dong & He 2026).

Ideia: cada sinapse tem um astrócito que integra sua contribuição ao
erro (|∂L/∂w · w| — uso sináptico acumulado). Essa integração vira um
**gate de plasticidade**: `g = 1/(1 + κ·A)` multiplica o passo de
aprendizado — sinapses importantes/consolidadas ficam rígidas; as demais
seguem plásticas. Em aprendizado contínuo isso reduz esquecimento; κ
grande demais trava a rede (dilema estabilidade–plasticidade).

Idea: each synapse has an astrocyte integrating its contribution to the
error (|∂L/∂w · w| — accumulated synaptic usage). That integral becomes
a **plasticity gate**: `g = 1/(1 + κ·A)` multiplies the learning step —
important/consolidated synapses stiffen; the rest stay plastic. In
continual learning this reduces forgetting; too-large κ freezes the
network (the stability–plasticity dilemma).
"""

import numpy as np


def make_tasks(seed, n_tasks=3, d=12, n=200):
    """Sequência de regressões lineares com alvos diferentes.
    Sequence of linear regressions with different targets."""
    rng = np.random.default_rng(seed)
    return [
        (X := rng.normal(0, 1, (n, d)), X @ rng.normal(0, 1, d) + rng.normal(0, 0.05, n))
        for _ in range(n_tasks)
    ]


class MetaplasticLinear:
    """Regressor linear com gate de plasticidade por sinapse.

    Linear regressor with a per-synapse plasticity gate driven by an
    astrocyte-like importance accumulator.
    """

    def __init__(self, d, kappa=5.0, agmp=True, seed=0):
        self.w = np.zeros(d)
        self.A = np.zeros(d)  # importância acumulada (astrócito)
        self.kappa = kappa
        self.agmp = agmp
        self.rng = np.random.default_rng(seed)

    def gate(self):
        """g = 1/(1 + κ·A): sinapses consolidadas perdem plasticidade."""
        return 1.0 / (1.0 + self.kappa * self.A) if self.agmp else np.ones_like(self.w)

    def step(self, X, y, eta=0.05, imp_rate=0.02):
        err = X @ self.w - y
        grad = X.T @ err / len(y)
        if self.agmp:
            # astrócito integra a contribuição sináptica ao erro
            self.A += imp_rate * np.abs(grad * self.w)
        self.w -= eta * self.gate() * grad
        return float(np.mean(err**2))


def train_sequence(seed, tasks, kappa, agmp=True, eta=0.05, steps=400):
    """Treina T1→…→Tk em sequência; avalia todas as tarefas após cada fase.

    Trains the task sequence and evaluates every task after each phase.
    Returns the (phase × task) MSE matrix.
    """
    net = MetaplasticLinear(tasks[0][0].shape[1], kappa, agmp, seed)
    M = np.zeros((len(tasks), len(tasks)))
    for ti, (X, y) in enumerate(tasks):
        for _ in range(steps):
            net.step(X, y, eta)
        for tk, (Xk, yk) in enumerate(tasks):
            M[ti, tk] = np.mean((Xk @ net.w - yk) ** 2)
    return M


def forgetting(M):
    """Esquecimento médio: erro final das tarefas antigas − erro ao aprender.

    Mean forgetting: final error on old tasks minus error right after
    learning them.
    """
    n = M.shape[0]
    return float(np.mean([M[n - 1, t] - M[t, t] for t in range(n - 1)]))


def plasticity(M):
    """Erro na tarefa mais nova (capacidade de aprender). / Newest task error."""
    return float(M[-1, -1])
