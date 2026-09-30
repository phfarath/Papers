"""p28 — glia realoca domínios a reservas; sem ela a cobertura cai."""

import numpy as np

from papers.p28_astromorphic_repair.method import SelfRepairNet, make_clusters, run


def test_determinism_same_seed():
    a, _ = run(0, True)
    b, _ = run(0, True)
    assert np.allclose(a, b)


def test_fault_drops_coverage_without_repair():
    h, _ = run(0, False)
    mid = len(h) // 2
    assert h[mid - 5] > 0.85
    assert h[mid + 1] < h[mid - 5] - 0.1


def test_repair_preserves_coverage():
    h, _ = run(0, True)
    mid = len(h) // 2
    assert h[mid + 1] > 0.9


def test_domain_memory_tracks_winners():
    X, _, _ = make_clusters(0)
    net = SelfRepairNet(X.shape[1], seed=1)
    for t in range(300):
        net.step(X[t % len(X)])
    # glia registra domínio não-nulo nos neurônios que venceram entradas
    norms = np.linalg.norm(net.m[net.alive], axis=1)
    assert (norms > 0.1).sum() >= 3


def test_spares_join_after_fault():
    X, _, _ = make_clusters(0)
    net = SelfRepairNet(X.shape[1], repair=True, seed=1)
    for t in range(300):
        net.step(X[t % len(X)])
    n_before = net.alive.sum()
    net.inject_fault(2)
    assert net.alive.sum() == n_before  # reservas substituíram os mortos
