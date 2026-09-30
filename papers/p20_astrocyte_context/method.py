"""Astrocytes as a Mechanism for Contextually-Guided Network Dynamics and
Function (Gong, Pasqualetti, Papouin & Ching, PLOS Comput. Biol. 2024;
DOI 10.1371/journal.pcbi.1012186).

PT: Laços aninhados em escalas temporais separadas: pesos rápidos f
(plasticidade da tarefa, regra delta), astrócitos a_c lentos que
CONSOLIDAM f enquanto o contexto c está ativo, e restauração
contextualizada f ← f + η_r(a_c − f) — o sinal de contexto GATEIA qual
traço astroglial modula a dinâmica neural agora. É o modelo do paper de
metaplasticidade: os astrócitos guardam "como a rede aprendeu" por
contexto e reintroduzem essa configuração quando o contexto volta, sem
reaprender do zero. Detalhe crucial do paper (reconhecido pelos autores):
a rede RECEBE um sinal de contexto — não descobre a mudança sozinha; por
isso incluímos a ablação "sem sinal de contexto".

EN: nested feedback loops over separated timescales: fast weights f
(task plasticity, delta rule), slow astrocytes a_c CONSOLIDATING f while
context c is active, and context-gated restoration
f ← f + η_r(a_c − f) — the context signal gates which astrocytic trace
modulates neural dynamics now. This is the paper's metaplasticity model:
astrocytes store "how the network learned" per context and reinstate it
when the context returns, without relearning from scratch. Crucial detail
acknowledged by the authors: the network RECEIVES a context signal — it
does not discover the switch autonomously; hence the "no context signal"
ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ETA_F = 0.3        # PT: plasticidade rápida (regra delta).
ETA_A = 0.08       # PT: consolidação astroglial (lenta).
ETA_R = 0.5        # PT: força de restauração contexto→neurônio.
N_INPUTS = 4


@dataclass
class ContextGatedNet:
    """PT: rede neurônio–astrócito com duas escalas temporais.

    - f: pesos rápidos dirigidos pelo erro da tarefa (regra delta).
    - a[c]: traço astroglial lento por contexto — consolida f.
    - restauração: a cada trial, f += η_r·(a_c − f) para o contexto ATIVO
      (gateado pelo sinal de contexto, como no modelo do paper).
    - `context_signal=False` abla a informação contextual: todos os
      contextos compartilham um único traço — a restauração vira ruído de
      arrasto (caveat do paper).

    EN: neuron–astrocyte net with two timescales. Fast weights f are
    driven by task error (delta rule); the slow astrocytic trace a[c]
    consolidates f per context; each trial restores
    f += η_r·(a_c − f) for the ACTIVE context (gated by the context
    signal, as in the paper's model). `context_signal=False` ablates
    contextual information: every context shares one trace — restoration
    becomes a drag artifact.
    """

    contexts: list[str]
    n: int = N_INPUTS
    eta_f: float = ETA_F
    eta_a: float = ETA_A
    eta_r: float = ETA_R
    context_signal: bool = True
    f: np.ndarray = field(init=False)
    a: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.f = np.zeros(self.n)
        keys = self.contexts if self.context_signal else ["__shared__"]
        self.a = {k: np.zeros(self.n) for k in keys}

    def _key(self, ctx: str) -> str:
        return ctx if self.context_signal else "__shared__"

    def trial(self, x: np.ndarray, target: np.ndarray, ctx: str) -> float:
        """PT: um trial: mede |erro|, atualiza f (delta) e a_c
        (consolidação + restauração gateada). EN: one trial."""
        err = float(target @ x - self.f @ x)
        self.f += self.eta_f * err * x / self.n
        k = self._key(ctx)
        if self.eta_r > 0:
            self.f += self.eta_r * (self.a[k] - self.f)
            self.a[k] += self.eta_a * (self.f - self.a[k])
        return abs(err)


def make_targets(rng: np.random.Generator, contexts: list[str],
                 n: int = N_INPUTS) -> dict[str, np.ndarray]:
    """PT: mapas-alvo bipolares {±1}^N por contexto; o segundo é a reversão
    exata do primeiro (conflito máximo, como no protocolo do paper).

    EN: bipolar {±1}^N target maps per context; the second is the exact
    reversal of the first (maximal conflict, as in the paper's protocol).
    """
    targets: dict[str, np.ndarray] = {}
    for i, c in enumerate(contexts):
        if i == 1 and contexts:
            targets[c] = -targets[contexts[0]]  # PT: reversão.
        else:
            targets[c] = rng.choice([-1.0, 1.0], n)
    return targets


def run_sequence(net: ContextGatedNet, targets: dict[str, np.ndarray],
                 seq: list[str], trials: int,
                 rng: np.random.Generator) -> list[list[float]]:
    """PT: devolve, por fase, os erros |err| dos 4 primeiros trials
    (custo de adaptação pós-troca). EN: per-phase early-trial errors."""
    early: list[list[float]] = []
    for ctx in seq:
        errs = []
        for t in range(trials):
            x = rng.choice([-1.0, 1.0], net.n)
            e = net.trial(x, targets[ctx], ctx)
            if t < 4:
                errs.append(e)
        early.append(errs)
    return early
