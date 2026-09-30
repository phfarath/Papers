"""p25 — AGMP: gate glial reduz esquecimento; κ alto trava plasticidade."""

import numpy as np

from papers.p25_agmp_continual.method import (
    MetaplasticLinear,
    forgetting,
    make_tasks,
    plasticity,
    train_sequence,
)


def test_determinism_same_seed():
    a = train_sequence(0, make_tasks(0), 5.0)
    b = train_sequence(0, make_tasks(0), 5.0)
    assert np.allclose(a, b)


def test_plain_forgets():
    M = train_sequence(0, make_tasks(0), 0.0, agmp=False)
    assert forgetting(M) > 5.0  # esquecimento catastrófico


def test_agmp_reduces_forgetting():
    Mp = train_sequence(0, make_tasks(0), 0.0, agmp=False)
    Ma = train_sequence(0, make_tasks(0), 5.0)
    assert forgetting(Ma) < forgetting(Mp)


def test_too_large_kappa_loses_plasticity():
    # κ exagerado: retém o velho mas não aprende o novo
    M_soft = train_sequence(0, make_tasks(0), 5.0)
    M_hard = train_sequence(0, make_tasks(0), 50.0)
    assert plasticity(M_hard) > plasticity(M_soft)
    assert forgetting(M_hard) < forgetting(M_soft)


def test_gate_bounds_and_shape():
    net = MetaplasticLinear(5, kappa=5.0)
    g = net.gate()
    assert g.shape == (5,) and np.all(g <= 1.0) and np.all(g > 0.0)
    X, y = make_tasks(0, n_tasks=1, d=5, n=50)[0]
    for _ in range(50):
        net.step(X, y)
    assert np.all(net.gate() < 1.0)  # astrócito acumulou importância
    assert np.all(net.A >= 0.0)
