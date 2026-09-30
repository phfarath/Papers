"""PT: testes do p21 — a ODE neurônio–astrócito converge para a atenção
softmax; sem astrócito, diverge da referência; determinismo.

EN: p21 tests — the neuron–astrocyte ODE converges to softmax attention;
ablated astrocyte diverges from the reference; determinism.
"""
import numpy as np

from papers.p21_astrocyte_transformer.method import (
    NeuronAstrocyteAttention,
    attention_scores,
    softmax_ref,
)


def test_converges_to_softmax_attention() -> None:
    rng = np.random.default_rng(0)
    s, v = attention_scores(rng)
    y, w, hist = NeuronAstrocyteAttention().run(s, v)
    ref = softmax_ref(s) @ v
    assert np.abs(y - ref).max() < 1e-6
    assert abs(w.sum() - 1.0) < 1e-6      # PT: pesos normalizam.


def test_astrocyte_does_normalization() -> None:
    rng = np.random.default_rng(1)
    s, v = attention_scores(rng)
    y_on, _, _ = NeuronAstrocyteAttention(astrocyte=True).run(s, v)
    y_off, _, _ = NeuronAstrocyteAttention(astrocyte=False).run(s, v)
    ref = softmax_ref(s) @ v
    assert np.abs(y_on - ref).max() < 1e-6
    assert np.abs(y_off - ref).max() > 0.1


def test_slow_astrocyte_converges_later() -> None:
    rng = np.random.default_rng(2)
    s, v = attention_scores(rng)
    fast = NeuronAstrocyteAttention(tau_a=1.0).run(s, v)[2]
    slow = NeuronAstrocyteAttention(tau_a=50.0).run(s, v)[2]
    assert len(fast) < len(slow)


def test_determinism() -> None:
    rng = np.random.default_rng(3)
    s, v = attention_scores(rng)
    a = NeuronAstrocyteAttention().run(s, v)[0]
    b = NeuronAstrocyteAttention().run(s, v)[0]
    assert np.array_equal(a, b)
