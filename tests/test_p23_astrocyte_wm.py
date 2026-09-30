"""p23 — WM com astrócitos: Ca²⁺ potencializa o item, distrator não."""

from papers.p23_astrocyte_wm.method import WMNetwork, run_batch

SEEDS = list(range(6))


def test_determinism_same_seed():
    a = WMNetwork(astro=True, seed=0).simulate()
    b = WMNetwork(astro=True, seed=0).simulate()
    assert a["item"] == b["item"] and a["distractor"] == b["distractor"]


def test_control_decays():
    # sem glia a recorrência sub-crítica não sustenta o item
    r = run_batch(False, SEEDS)
    assert r["item"] < 0.5 and r["held"] < 0.5


def test_astrocytes_hold_item_over_distractor():
    r = run_batch(True, SEEDS)
    assert r["item"] > 0.8
    assert r["distractor"] < 0.3
    assert r["held"] >= 5 / 6


def test_calcium_threshold_gates_potentiation():
    # distrator breve nunca cruza o limiar de Ca²⁺ — verificar via trace
    net = WMNetwork(astro=True, seed=0)
    res = net.simulate()
    assert res["ca_tr"].max() > net.c_thr  # item cruzou o limiar
    assert net.C[net.distr].max() < net.c_thr  # distrator não


def test_alpha_zero_equals_control():
    a = WMNetwork(astro=True, alpha=0.0, seed=0).simulate()
    c = WMNetwork(astro=False, seed=0).simulate()
    assert abs(a["item"] - c["item"]) < 1e-9
