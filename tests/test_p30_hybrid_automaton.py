"""p30 — acumulador glial detecta quebra de regra e troca de modo."""

import numpy as np

from papers.p30_hybrid_automaton.method import HybridAutomaton, run


def test_determinism_same_seed():
    a = run(0, True, eta=0.05)
    b = run(0, True, eta=0.05)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_glia_beats_single_q_slow_eta():
    rw_g, _, _ = run(0, True, eta=0.05)
    rw_c, _, _ = run(0, False, eta=0.05)
    assert rw_g > rw_c


def test_switch_delay_is_fast():
    _, delays, _ = run(0, True, eta=0.05)
    assert np.mean(delays) < 15


def test_accumulator_resets_on_switch():
    ag = HybridAutomaton(glia=True, seed=0, theta=0.5)
    ag.Q[0] = [1.0, 0.0]
    for _ in range(200):
        if ag.accumulate(0.0):  # recompensa sempre abaixo do esperado
            break
    assert ag.A == 0.0 and ag.switches == 1


def test_no_switch_without_glia():
    ag = HybridAutomaton(glia=False, seed=0)
    for _ in range(100):
        assert not ag.accumulate(0.0)
    assert ag.mode == 0
