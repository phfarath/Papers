"""p27 — replicador + normalização glial = softmax emergente."""

import numpy as np

from papers.p27_emergent_attention.method import (
    ReplicatorAttention,
    entropy,
    track_vs_theory,
)

F = np.array([0.2, 1.5, -0.5, 0.8, -1.0, 0.0])


def test_determinism_same_seed():
    a = ReplicatorAttention(F, 1.0)
    b = ReplicatorAttention(F, 1.0)
    assert np.allclose(a.run(10), b.run(10))


def test_matches_softmax_theory():
    err, _ = track_vs_theory(F, beta=1.0, steps=20)
    assert err[-1] < 1e-10


def test_winner_take_all_limit():
    att = ReplicatorAttention(F, 1.0)
    hist = att.run(60)
    assert hist[-1].argmax() == np.argmax(F)
    assert entropy(hist[-1]) < entropy(hist[0])


def test_simplex_preserved():
    att = ReplicatorAttention(F, 2.0)
    for g in att.run(10):
        assert abs(g.sum() - 1.0) < 1e-9


def test_ablated_normalization_diverges():
    att = ReplicatorAttention(F, 1.0, glia=False)
    att.run(20)
    assert att.g.sum() > 1e3  # sem a glia os ganhos divergem
