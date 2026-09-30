"""Mecanismo central do paper p30 — autômato híbrido neurônio–astrócito.

Core mechanism of p30 — Neural-astrocyte architecture implementing a
hybrid automaton for evidence accumulation (Vedovati et al. 2026).

Ideia: o agente tem modos discretos (Q-tables por contexto latente) e um
acumulador glial contínuo que integra a **evidência de regra quebrada** —
recompensa abaixo do esperado (`max(0, Q_max − r)`), com vazamento. Ao
cruzar o limiar, dispara a transição de modo (a parte "autômato" do
sistema híbrido): troca de contexto sem re-aprender, pois cada modo tem
sua própria memória de valores.

Idea: the agent has discrete modes (per-latent-context Q-tables) and a
continuous glial accumulator integrating **broken-rule evidence** —
reward below expectation (`max(0, Q_max − r)`), with leak. Crossing the
threshold triggers a mode transition (the "automaton" part of the hybrid
system): switch context without relearning, since each mode keeps its
own value memory.
"""

import numpy as np


class HybridAutomaton:
    """Modos Q + acumulador glial de evidência → transição de contexto.

    Per-mode Q memory + glial evidence accumulator → context transition.
    `glia=False`: single-Q baseline (no modes, no switching).
    """

    def __init__(self, n_arms=2, glia=True, eta=0.05, accum_rate=0.2,
                 leak=0.01, theta=1.2, eps=0.1, seed=0):
        self.glia = glia
        self.eta, self.eps = eta, eps
        self.accum_rate, self.leak, self.theta = accum_rate, leak, theta
        self.Q = np.zeros((2, n_arms))
        self.mode = 0
        self.A = 0.0
        self.rng = np.random.default_rng(seed)
        self.switches = 0

    def act(self):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[self.mode] if self.glia else self.Q[0]
        return int(np.argmax(q + self.rng.normal(0, 1e-6, q.shape)))

    def learn(self, arm, r):
        self.Q[self.mode if self.glia else 0, arm] += self.eta * (
            r - self.Q[self.mode if self.glia else 0, arm]
        )

    def accumulate(self, r):
        """Integra evidência de regra quebrada; retorna True se trocou de modo."""
        if not self.glia:
            return False
        surprise = max(0.0, self.Q[self.mode].max() - r)
        self.A += self.accum_rate * surprise - self.leak
        if self.A > self.theta:
            self.mode = 1 - self.mode
            self.A = 0.0
            self.switches += 1
            return True
        return False


def run(seed, glia, T=1200, p_hi=0.85, p_lo=0.15, flip_every=60, **kw):
    """Bandit de regra latente: o braço bom alterna a cada `flip_every`.

    Latent-rule bandit: the good arm flips every `flip_every` trials.
    Returns (total_reward, switch_delays, false_switches).
    """
    rng = np.random.default_rng(seed + 777)
    ctx = (np.arange(T) // flip_every) % 2
    probs = np.where(ctx[:, None] == 0, [p_hi, p_lo], [p_lo, p_hi])
    ag = HybridAutomaton(glia=glia, seed=seed, **kw)
    rew = 0.0
    delays, false_sw = [], 0
    last_flip = 0
    for t in range(T):
        arm = ag.act()
        r = float(rng.random() < probs[t, arm])
        rew += r
        ag.learn(arm, r)
        if ag.accumulate(r):
            if t - last_flip > 15:
                false_sw += 1
            else:
                delays.append(t - last_flip)
        if t > 0 and ctx[t] != ctx[t - 1]:
            last_flip = t
    return rew, delays, false_sw
