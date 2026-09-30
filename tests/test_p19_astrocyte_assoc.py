"""PT: testes do p19 — invariantes do NAM: K=1 ≈ Hopfield, ordens altas
recuperam sob carga alta, determinismo por seed.

EN: p19 tests — NAM invariants: K=1 ≈ Hopfield, higher orders recall
under high load, seed determinism.
"""
import numpy as np

from papers.p19_astrocyte_assoc.method import (
    NeuronAstrocyteMemory,
    capacity_sweep,
    degrade,
    make_patterns,
    overlaps,
    recall_score,
)


def test_degrade_flips_exact_fraction() -> None:
    rng = np.random.default_rng(0)
    p = make_patterns(rng, 4, 64)[0]
    cue = degrade(rng, p, 0.5)
    assert (cue != p).sum() == 32
    assert set(np.unique(cue)) == {-1.0, 1.0}


def test_low_load_hopfield_recalls() -> None:
    rng = np.random.default_rng(1)
    pats = make_patterns(rng, 6, 64)
    mem = NeuronAstrocyteMemory(pats, order=1)
    scores = [recall_score(mem, rng, 0, flip_frac=0.25) for _ in range(5)]
    assert sum(scores) / len(scores) > 0.9


def test_higher_order_beats_hopfield_at_high_load() -> None:
    rng0, rng1 = np.random.default_rng(3), np.random.default_rng(3)
    hop, _ = capacity_sweep(rng0, 64, [0.8], order=1, trials=12)
    nam, _ = capacity_sweep(rng1, 64, [0.8], order=3, trials=12)
    assert nam[0] > hop[0]


def test_gains_softmax_sums_to_one() -> None:
    rng = np.random.default_rng(2)
    pats = make_patterns(rng, 8, 64)
    mem = NeuronAstrocyteMemory(pats, beta=5.0)
    g = mem.gains(overlaps(pats, pats[0]))
    assert abs(g.sum() - 1.0) < 1e-9
    assert g.argmax() == 0  # PT: o próprio padrão vence. EN: self wins.


def test_determinism() -> None:
    a = capacity_sweep(np.random.default_rng(7), 32, [0.4], order=3,
                       trials=6)
    b = capacity_sweep(np.random.default_rng(7), 32, [0.4], order=3,
                       trials=6)
    assert a == b
