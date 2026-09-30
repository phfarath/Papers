"""p26 — supressão glial dirige exploração; Q consolida trajetórias."""

import numpy as np

from papers.p26_dual_memory_nav.method import Navigator, learn


def test_determinism_same_seed():
    a = learn(0, True, n_ep=5)
    b = learn(0, True, n_ep=5)
    assert np.allclose(a[0], b[0]) and a[1:] == b[1:]


def test_glia_converges_faster():
    sg, _, _ = learn(0, True, n_ep=20)
    sc, _, _ = learn(0, False, n_ep=20)
    assert sg[-5:].mean() < sc[-5:].mean()


def test_glia_avoids_failures():
    fails = 0
    for s in range(4):
        _, _, f = learn(s, True, n_ep=10)
        fails += f
    assert fails == 0


def test_long_term_consolidation():
    nav = Navigator(glia=True, seed=0)
    assert nav.Q.max() == 0.0
    nav.episode()
    assert nav.Q.max() > 0.0  # trajetória consolidada


def test_beta_zero_matches_control_shape():
    a, _, _ = learn(0, True, n_ep=3, beta=0.0)
    assert a.shape == (3,)
