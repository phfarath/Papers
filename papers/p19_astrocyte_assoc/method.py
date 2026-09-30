"""Neuron-Astrocyte Associative Memory (Kozachkov, Slotine & Krotov, PNAS
2025; arXiv:2311.08135).

PT: Rede tripartite — neurônios x ∈ {±1}^N, padrões ξ^μ e astrócitos, cada
um contactando várias sinapses. O encadeamento sinapse→astrócito→sinapse
gera interações efetivas de ordem superior às do Hopfield clássico: a
família de energias E = −Σ_μ F(m_μ), m_μ = (ξ^μ·x)/N, inclui o Hopfield
(F = m², recuperação ∝ m) e Dense Associative Memories (F = m^{K+1},
recuperação ∝ m^K) como casos; mais sinapses por astrócito ⇒ maior K ⇒
capacidade de memória supralinear em N — a tese central do paper. A
recuperação suave com ganhos softmax(β·m) é o caso limite "Transformer"
(apenas `beta` definido).

EN: tripartite network — neurons x ∈ {±1}^N, patterns ξ^μ and astrocytes
each contacting many synapses. The synapse→astrocyte→synapse cascade
yields effective interactions of higher order than classic Hopfield: the
energy family E = −Σ_μ F(m_μ), m_μ = (ξ^μ·x)/N, spans Hopfield (F = m²,
retrieval ∝ m) and Dense Associative Memories (F = m^{K+1}, retrieval
∝ m^K); more synapses per astrocyte ⇒ larger K ⇒ supralinear memory
capacity in N — the paper's central claim. Soft readout with softmax(β·m)
gains is the "Transformer" limiting case (set `beta`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_NEURONS = 64
FLIP_FRAC = 0.3      # PT: fração de bits invertidos na degradação do cue.
STEPS = 30           # PT: passos síncronos de dinâmica. EN: update steps.
RECALL_MIN = 0.8     # PT: cosseno mínimo p/ contar como recuperado.


def make_patterns(rng: np.random.Generator, n_patterns: int,
                  n: int = N_NEURONS) -> np.ndarray:
    """PT: padrões bipolares {±1}^N. EN: bipolar {±1}^N patterns."""
    return rng.choice([-1.0, 1.0], size=(n_patterns, n))


def degrade(rng: np.random.Generator, pattern: np.ndarray,
            flip_frac: float = FLIP_FRAC) -> np.ndarray:
    """PT: cue = padrão com `flip_frac` dos bits invertidos.

    EN: cue = pattern with `flip_frac` of the bits flipped.
    """
    x = pattern.copy()
    idx = rng.choice(pattern.size, size=int(pattern.size * flip_frac),
                     replace=False)
    x[idx] *= -1.0
    return x


def overlaps(patterns: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PT: m_μ = (ξ^μ·x)/N ∈ [−1,1]. EN: normalized pattern overlaps."""
    return patterns @ x / x.size


@dataclass
class NeuronAstrocyteMemory:
    """PT: memória associativa neurônio–astrócito.

    - `order` K: K=1 → Hopfield clássico (só sinapses); K≥2 → astrócitos
      contactando K sinapses medeiam interações de ordem K+1 — o campo
      local fica h_i = Σ_μ ξ_i^μ sign(m_μ)|m_μ|^K.
    - `beta` definido → leitura suave softmax(β·m) (caso limite
      Transformer): os ganhos g_μ normalizam as similaridades numa
      distribuição de atenção.

    EN: neuron–astrocyte associative memory. `order` K: K=1 → classic
    Hopfield (synapses only); K≥2 → astrocytes contacting K synapses
    mediate order-(K+1) interactions — local field
    h_i = Σ_μ ξ_i^μ sign(m_μ)|m_μ|^K. Set `beta` for the soft
    softmax(β·m) readout (the Transformer limiting case).
    """

    patterns: np.ndarray
    order: int = 3
    beta: float | None = None

    def gains(self, m: np.ndarray) -> np.ndarray:
        """PT: ganhos g_μ por padrão — o "sinal" que os astrócitos devolvem
        às sinapses: potência K-ésima dura, ou softmax(β·m) suave.

        EN: per-pattern gains g_μ — the astrocytic feedback signal onto
        synapses: hard K-th power, or soft softmax(β·m).
        """
        if self.beta is not None:  # PT: caso limite Transformer.
            w = np.exp(self.beta * (m - m.max()))
            return w / w.sum()
        return np.sign(m) * np.abs(m) ** self.order

    def field(self, x: np.ndarray) -> np.ndarray:
        """PT: campo local h_i = Σ_μ ξ_i^μ g_μ. EN: local field."""
        return self.patterns.T @ self.gains(overlaps(self.patterns, x))

    def step(self, x: np.ndarray) -> np.ndarray:
        """PT: passo síncrono x' = sign(h) (duro) ou direção de h (suave).

        EN: synchronous step x' = sign(h) (hard) or h's direction (soft).
        """
        h = self.field(x)
        if self.beta is not None:  # PT: recuperação suave → contínua.
            return h / max(np.linalg.norm(h), 1e-12)
        out = np.sign(h)
        out[out == 0] = 1.0
        return out

    def recall(self, cue: np.ndarray, steps: int = STEPS) -> np.ndarray:
        """PT: itera a dinâmica até `steps` ou convergência.

        EN: iterate dynamics until `steps` or fixed point.
        """
        x = cue.copy()
        for _ in range(steps):
            x_new = self.step(x)
            if np.allclose(x_new, x):
                x = x_new
                break
            x = x_new
        return x


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    """PT: |x·y|/(‖x‖‖y‖). EN: absolute cosine similarity."""
    return float(abs(x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y),
                                  1e-12))


def recall_score(mem: NeuronAstrocyteMemory, rng: np.random.Generator,
                 target: int, flip_frac: float = FLIP_FRAC,
                 steps: int = STEPS) -> float:
    """PT: cosseno final |x·ξ_target| com cue degradado.

    EN: final |cosine(x, ξ_target)| from a degraded cue.
    """
    cue = degrade(rng, mem.patterns[target], flip_frac)
    return cosine(mem.recall(cue, steps), mem.patterns[target])


def capacity_sweep(rng: np.random.Generator, n: int, loads: list[float],
                   order: int = 3, beta: float | None = None,
                   trials: int = 12) -> tuple[list[float], list[float]]:
    """PT: por carga P/N devolve (fração recuperada, cosseno médio).

    EN: per load P/N returns (recall fraction ≥ RECALL_MIN, mean cosine).
    """
    wins, coss = [], []
    for load in loads:
        p = max(int(round(n * load)), 2)
        w, c = 0.0, 0.0
        for _ in range(trials):
            patterns = make_patterns(rng, p, n)
            mem = NeuronAstrocyteMemory(patterns, order=order, beta=beta)
            target = int(rng.integers(p))
            s = recall_score(mem, rng, target)
            c += s
            w += s >= RECALL_MIN
        wins.append(w / trials)
        coss.append(c / trials)
    return wins, coss
