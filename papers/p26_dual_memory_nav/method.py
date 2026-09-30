"""Mecanismo central do paper p26 — memória em duas escalas para navegação.

Core mechanism of p26 — dual-timescale memory for navigation
(Tsybina et al. 2026).

Ideia: navegação num grid exige duas memórias. A de **longo prazo**
consolida trajetórias bem-sucedidas (trace-back Hebbiano sobre valores de
ação ao atingir o objetivo). A de **curto prazo** é glial: astrócitos
suprimem células visitadas recentemente (fadiga espacial, decai dentro do
episódio) — a exploração vira dirigida por novidade, evitando ciclos e
revisitas antes da consolidação.

Idea: grid navigation needs two memories. The **long-term** one
consolidates successful trajectories (Hebbian trace-back over action
values upon reaching the goal). The **short-term** one is glial:
astrocytes suppress recently visited cells (spatial fatigue decaying
within the episode) — exploration becomes novelty-driven, avoiding loops
and revisits before consolidation.
"""

import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Navigator:
    """Agente no grid com Q de longo prazo + supressão glial de visitadas.

    Grid agent with long-term action values + glial suppression of
    visited cells.
    """

    def __init__(self, grid=10, goal=(9, 9), glia=True, beta=1.0, alpha=0.3,
                 lam=0.8, eps=0.15, decay=0.98, seed=0):
        self.G = grid
        self.goal = goal
        self.glia = glia
        self.beta, self.alpha, self.lam = beta, alpha, lam
        self.eps, self.decay = eps, decay
        self.Q = np.zeros((grid, grid, 4))  # memória de longo prazo
        self.rng = np.random.default_rng(seed)

    def _nexts(self, pos):
        x, y = pos
        return [
            (min(max(x + dx, 0), self.G - 1), min(max(y + dy, 0), self.G - 1))
            for dx, dy in DIRS
        ]

    def episode(self, max_steps=500):
        """Um episódio: anda até o objetivo ou esgota passos.

        One episode: walk to the goal or exhaust the step budget.
        Returns (steps, revisits, reached).
        """
        sup = np.zeros((self.G, self.G))  # supressão glial (escala curta)
        pos = (0, 0)
        traj = []
        revisits = 0
        for st in range(max_steps):
            nexts = self._nexts(pos)
            score = np.array(
                [
                    self.Q[nx, ny, a] - (self.beta * sup[nx, ny] if self.glia else 0.0)
                    for a, (nx, ny) in enumerate(nexts)
                ]
            )
            score += self.rng.normal(0, 1e-6, 4)  # desempate aleatório
            a = int(np.argmax(score)) if self.rng.random() > self.eps else int(
                self.rng.integers(4)
            )
            nx, ny = nexts[a]
            if self.glia:
                if sup[nx, ny] > 0.3:
                    revisits += 1
                sup *= self.decay
                sup[nx, ny] += 1.0
            traj.append((pos[0], pos[1], a))
            pos = (nx, ny)
            if pos == self.goal:
                # consolidação: trace-back reforça a trajetória (longo prazo)
                aa = self.alpha
                for px, py, pa in traj[::-1]:
                    self.Q[px, py, pa] += aa * (1 - self.Q[px, py, pa])
                    aa *= self.lam
                return st + 1, revisits, True
        return max_steps, revisits, False


def learn(seed, glia, n_ep=30, **kw):
    """Roda `n_ep` episódios; retorna curvas de passos e revisitas."""
    nav = Navigator(glia=glia, seed=seed, **kw)
    steps, revs, fails = [], [], 0
    for _ in range(n_ep):
        st, rv, ok = nav.episode()
        steps.append(st)
        revs.append(rv)
        fails += not ok
    return np.array(steps, float), float(np.sum(revs)), fails
