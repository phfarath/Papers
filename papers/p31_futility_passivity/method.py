"""Modelo conceitual do paper p31 — glia acumula evidência de futilidade.

Conceptual model of p31 — Glia Accumulate Evidence that Actions Are
Futile and Suppress Unsuccessful Behavior (Mu et al., Cell 2019).

NOTA: o paper é experimental (larvas de zebrafish); isto é um modelo
abstrato do achado, não uma replicação da biologia. O agente tenta agir;
cada tentativa frustrada aumenta a evidência glial E; um sucesso zera E.
Quando E cruza o limiar, o comportamento é suprimido (passividade) por um
período — economizando esforço quando a ação é realmente fútil, mas
pagando o preço de falsos positivos em regimes ruidosos-viáveis.

NOTE: the paper is experimental biology (zebrafish larvae); this is an
abstract model of the finding, not a biological replication. The agent
acts; each futile attempt raises glial evidence E; a success resets E.
When E crosses the threshold behavior is suppressed (passivity) for a
period — saving effort when actions are truly futile, at the cost of
false positives in noisy-but-viable regimes.
"""

import numpy as np


class FutilityAgent:
    """Acumulador glial de futilidade → passividade.

    Glial futility accumulator → passivity.
    `glia=False`: agente sem acumulador (tenta sempre).
    """

    def __init__(self, glia=True, alpha=0.25, leak=0.02, theta=1.2,
                 cool=10, pleak=0.2, seed=0):
        self.glia = glia
        self.alpha, self.leak, self.theta = alpha, leak, theta
        self.cool, self.pleak = cool, pleak
        self.E = 0.0
        self.passive = 0
        self.rng = np.random.default_rng(seed)

    def step(self, p):
        """Um período: se passivo, descansa e E decai; senão tenta (p de acertar).

        One period: if passive, rest and decay E; else attempt with success
        probability p. Returns (attempted, won).
        """
        if self.passive > 0:
            self.passive -= 1
            self.E = max(0.0, self.E - self.pleak)
            return False, 0.0
        o = float(self.rng.random() < p)
        if self.glia:
            self.E = 0.0 if o else max(0.0, self.E + self.alpha - self.leak)
            if self.E > self.theta:
                self.passive = self.cool
                self.E = self.theta * 0.8
        return True, o


def make_schedule(kind, T=400, p_viable=0.3):
    if kind == "futile":
        return np.zeros(T)
    if kind == "viable":
        return np.full(T, p_viable)
    if kind == "phases":
        return np.concatenate(
            [np.zeros(T * 3 // 8), np.full(T * 3 // 8, p_viable), np.zeros(T - T * 3 // 4)]
        )
    raise ValueError(kind)


def run(seed, schedule, glia=True, **kw):
    """Roda o agente sobre a tabela de p por período.

    Runs the agent over the per-period success-probability schedule.
    Returns (attempts, wins, passive_periods, passive_history).
    """
    ag = FutilityAgent(glia=glia, seed=seed, **kw)
    attempts, wins, npass = 0, 0.0, 0
    hist = []
    for p in schedule:
        att, o = ag.step(p)
        attempts += att
        wins += o
        npass += ag.passive > 0
        hist.append(int(ag.passive > 0))
    return attempts, wins, npass, np.array(hist)
