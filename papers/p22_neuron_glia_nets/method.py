"""Mecanismo central do paper p22 — Redes Neurônio–Glia Artificiais (ANGN).

Core mechanism of p22 — Artificial Neuron–Glia Networks (Porto-Pazos et al. 2011).

Ideia: uma MLP de uma camada oculta treina com gradiente. Na ANGN, cada
unidade oculta tem um astrócito que integra a magnitude do gradiente dos
pesos aferentes: quando o neurônio "estagna" (gradiente ~0 com erro alto,
i.e. preso num platô), o astrócito injeta ruído nos pesos sinápticos — o
"postsynaptic noise" do paper — ajudando a escapar de ótimos locais.

A MLP de uma camada oculta treina com gradiente. Na ANGN, cada unidade
oculta tem um astrócito que integra a magnitude do gradiente dos pesos
aferentes: quando o neurônio estagna (gradiente ~0 com erro alto — preso
num platô), o astrócito injeta ruído nos pesos sinápticos, o
"postsynaptic noise" do paper, ajudando a escapar de ótimos locais.

Idea: a single-hidden-layer MLP trained by gradient descent. In the NGN,
each hidden unit has an astrocyte integrating the incoming-weights
gradient magnitude: when the neuron stalls (grad ~0 while error is high —
stuck on a plateau), the astrocyte injects noise into its synaptic
weights (the paper's "postsynaptic noise"), helping escape local optima.
"""

import numpy as np

MSE_CRIT = 0.05  # critério de "resolveu" / solved criterion
GRAD_EPS = 1e-3  # gradiente abaixo disso = estagnado / stalled


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_task(name):
    """Tarefas booleanas do paper; entradas {0,1}. / Boolean tasks."""
    if name == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.0]])
        y = np.array([0, 1, 1, 0.0])
    elif name == "parity3":
        X = np.array(
            [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float
        )
        y = X.sum(1) % 2
    elif name == "sparse_parity":
        X = np.array(
            [
                [a, b, c, d, e, f]
                for a in (0, 1)
                for b in (0, 1)
                for c in (0, 1)
                for d in (0, 1)
                for e in (0, 1)
                for f in (0, 1)
            ],
            float,
        )
        y = X[:, :3].sum(1) % 2  # só as 3 primeiras entradas importam
    elif name == "sel_and":
        X = np.array(
            [[a, b, c, d] for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)],
            float,
        )
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(float)  # só 2 entradas importam
    else:
        raise ValueError(name)
    return X, y


class NeuronGliaNet:
    """MLP 1-hidden-layer sigmoide; astrócitos injetam ruído ao estagnar.

    Astrocytes monitor per-hidden-unit incoming gradient magnitude and
    inject noise into stalled synapses (postsynaptic noise).
    """

    def __init__(self, n_in, n_hidden=4, seed=0, glia=True, stall_thr=20,
                 kick_mag=1.5, grad_eps=GRAD_EPS, init_scale=2.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.w1 = rng.normal(0, init_scale, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, init_scale, n_hidden)
        self.b2 = np.zeros(1)
        self.glia = glia
        self.stall_thr = stall_thr
        self.kick_mag = kick_mag
        self.grad_eps = grad_eps
        self.stall = np.zeros(n_hidden)  # estado do astrócito por unidade
        self.kicks = 0

    def forward(self, X):
        h = sigmoid(X @ self.w1 + self.b1)
        return h, sigmoid(h @ self.w2 + self.b2)

    def step(self, X, y, lr=0.3):
        """Uma época de GD full-batch + passo do astrócito.

        One full-batch GD epoch plus the astrocyte step.
        Returns (mse, n_kicked).
        """
        h, o = self.forward(X)
        mse = float(((o - y) ** 2).mean())
        d2 = (o - y) * o * (1 - o)                 # (S,)
        gw2 = h.T @ d2                           # (nh,)
        d1 = (d2[:, None] * self.w2) * h * (1 - h)  # (S, nh)
        gw1 = X.T @ d1                           # (n_in, nh)
        kicked = 0
        if self.glia:
            # astrócito: conta épocas consecutivas de gradiente ~0 por neurônio
            stall_now = np.abs(gw1).mean(0) < self.grad_eps
            self.stall = np.where(stall_now, self.stall + 1, 0)
            kick = (self.stall > self.stall_thr) & (mse > 2 * MSE_CRIT)
            if kick.any():
                self.w1[:, kick] += self.rng.normal(
                    0, self.kick_mag, (self.w1.shape[0], kick.sum())
                )
                self.b1[kick] += self.rng.normal(0, self.kick_mag, kick.sum())
                self.stall[kick] = 0
                kicked = int(kick.sum())
                self.kicks += kicked
        self.w1 -= lr * gw1
        self.b1 -= lr * d1.mean(0)
        self.w2 -= lr * gw2
        self.b2 -= lr * d2.mean() * 10  # bias de saída mais rápido
        return mse, kicked

    def mse(self, X, y):
        _, o = self.forward(X)
        return float(((o - y) ** 2).mean())


def train(seed, X, y, glia, n_hidden=4, lr=0.3, epochs=6000, init_scale=2.0):
    """Treina até MSE<critério ou `epochs`. / Train until solved or budget out.

    Returns dict(epochs_to_solve, final_mse, kicks, solved).
    """
    net = NeuronGliaNet(X.shape[1], n_hidden, seed, glia, init_scale=init_scale)
    solved = False
    t_solve = epochs
    for t in range(epochs):
        mse, _ = net.step(X, y, lr)
        if mse < MSE_CRIT:
            solved = True
            t_solve = t
            break
    final = net.mse(X, y)
    solved = solved or final < MSE_CRIT
    return {
        "epochs_to_solve": t_solve if solved else epochs,
        "final_mse": final,
        "kicks": net.kicks,
        "solved": solved,
    }


def run_experiment(name, glia, seeds, n_hidden, epochs=6000):
    """Roda `train` para vários seeds; agrega taxa de sucesso e tempo médio."""
    X, y = make_task(name)
    rs = [train(s, X, y, glia, n_hidden=n_hidden, epochs=epochs) for s in seeds]
    return {
        "succ": float(np.mean([r["solved"] for r in rs])),
        "t_mean": float(np.mean([r["epochs_to_solve"] for r in rs if r["solved"]]))
        if any(r["solved"] for r in rs)
        else float("nan"),
        "kicks": float(np.mean([r["kicks"] for r in rs])),
        "mse_end": float(np.mean([r["final_mse"] for r in rs])),
    }
