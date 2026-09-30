"""p29 — unidades lentas melhoram predição longa; razão tem ótimo."""

import numpy as np

from papers.p29_lsm_astrocytes.method import LiquidAstro, evaluate, mackey_glass


def test_determinism_same_seed():
    assert evaluate(0, 1.0) == evaluate(0, 1.0)


def test_mackey_glass_shape():
    s = mackey_glass(100, seed=0)
    assert s.shape == (100,) and np.all(np.isfinite(s))


def test_astrocytes_beat_no_astro():
    e0 = evaluate(0, 0.0)
    e1 = evaluate(0, 1.0)
    assert e1 < e0


def test_ratio_has_intermediate_optimum():
    # ratio 0.5 melhor que 0 e que 4 — curva em U
    e0 = evaluate(0, 0.0)
    e_mid = evaluate(0, 0.5)
    e_hi = evaluate(0, 4.0)
    assert e_mid < e0 and e_mid < e_hi


def test_feature_size_grows_with_ratio():
    a = LiquidAstro(ratio=1.0, seed=0)
    b = LiquidAstro(ratio=2.0, seed=0)
    assert a.step(0.1).shape[0] == 100
    assert b.step(0.1).shape[0] == 150
