"""Building Transformers from Neurons and Astrocytes (Kozachkov,
Kastanenka & Krotov, PNAS 2023; DOI 10.1073/pnas.2219150120, bioRxiv
2022.10.12.511910).

PT: A correspondência proposta no paper: a operação central do bloco
Transformer — softmax(QKᵀ/√d)·V — emerge de uma dinâmica neurônio–
astrócito em duas escalas temporais:

- neurônios n_i (rápidos, τ_n): dn_i/dt = (−n_i + s_i)/τ_n — carregam os
  scores de atenção s_i = (q·k_i)/√d;
- astrócito A (lento, τ_a): dA/dt = (−A + Σ_j exp(n_j))/τ_a — integra a
  atividade neural total (pooling espacial dos processos astrogliais);
- saída: y_i = exp(n_i)/A — normalização divisiva. No ponto fixo,
  n_i = s_i e A = Σ_j exp(s_j), logo y = softmax(s)·V — a atenção.

O papel do astrócito é justamente o DENOMINADOR da softmax: pooling lento
da atividade conjunta. Sem ele (A≡1) a saída é a atenção linear não
normalizada. Importante: isto é uma correspondência computacional
proposta pelos autores — não evidência de que o cérebro "rode" um
Transformer.

EN: the paper's proposed correspondence — the core Transformer
computation softmax(QKᵀ/√d)·V emerges from a two-timescale
neuron–astrocyte dynamics: fast neurons n_i carry attention scores
s_i = (q·k_i)/√d; a slow astrocyte A integrates total neural activity
(spatial pooling across astrocytic processes); output
y_i = exp(n_i)/A (divisive normalization). At the fixed point y =
softmax(s)·V — attention. The astrocyte's role is precisely the softmax
DENOMINATOR. This is a proposed computational correspondence, not
evidence that the brain runs a Transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU_N = 1.0        # PT: escala neural rápida. EN: fast neural timescale.
TAU_A = 10.0       # PT: escala astroglial lenta. EN: slow astrocyte.
DT = 0.05
T_MAX = 4000
TOL = 1e-9


def softmax_ref(scores: np.ndarray) -> np.ndarray:
    """PT: referência exata softmax(s). EN: exact softmax reference."""
    w = np.exp(scores - scores.max())
    return w / w.sum()


@dataclass
class NeuronAstrocyteAttention:
    """PT: circuito neurônio–astrócito cuja dinâmica converge para a
    atenção softmax. `astrocyte=False` desliga a normalização (A≡1) —
    atenção linear não normalizada, o baseline de ablação.

    EN: neuron–astrocyte circuit whose dynamics converge to softmax
    attention. `astrocyte=False` disables normalization (A≡1) —
    unnormalized linear attention, the ablation baseline.
    """

    tau_n: float = TAU_N
    tau_a: float = TAU_A
    astrocyte: bool = True

    def run(self, scores: np.ndarray, values: np.ndarray,
            dt: float = DT, t_max: int = T_MAX,
            track: bool = False) -> tuple[np.ndarray, np.ndarray,
                                        list[float]]:
        """PT: integra a ODE até tolerância; devolve (y, pesos, erros).

        EN: integrate the ODE to tolerance; returns (y, weights, errors).
        """
        n = np.zeros_like(scores)
        a = 0.0
        ref = softmax_ref(scores)
        errs: list[float] = []
        w = np.zeros_like(scores)
        for _ in range(t_max):
            n += dt * (-n + scores) / self.tau_n
            if self.astrocyte:
                a += dt * (-a + np.exp(n).sum()) / self.tau_a
            else:
                a = 1.0
            w = np.exp(n) / a
            errs.append(float(np.abs(w - ref).max()))
            if errs[-1] < TOL:
                break
        return w @ values, w, errs


def attention_scores(rng: np.random.Generator, n_keys: int = 10,
                     d: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """PT: scores s_i = (q·k_i)/√d e valores V de uma query aleatória.

    EN: scores s_i = (q·k_i)/√d and values V for one random query.
    """
    q = rng.normal(size=d)
    k = rng.normal(size=(n_keys, d))
    v = rng.normal(size=(n_keys, 6))
    return k @ q / np.sqrt(d), v
