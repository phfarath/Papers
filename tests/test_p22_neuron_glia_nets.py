"""p22 — ANGN: astrócitos injetam ruído em neurônios estagnados."""

import numpy as np

from papers.p22_neuron_glia_nets.method import (
    NeuronGliaNet,
    make_task,
    run_experiment,
    train,
)

X, Y = make_task("xor")


def test_determinism_same_seed():
    a = train(0, X, Y, glia=True, n_hidden=2, epochs=800)
    b = train(0, X, Y, glia=True, n_hidden=2, epochs=800)
    assert a == b


def test_astrocyte_fires_kicks():
    # net com astrócito deve disparar kicks em algum seed que estagna
    kicks = [
        train(s, X, Y, glia=True, n_hidden=2, epochs=2000)["kicks"] for s in range(6)
    ]
    assert sum(kicks) > 0


def test_no_kicks_without_glia():
    net = NeuronGliaNet(2, 2, seed=0, glia=False)
    for _ in range(500):
        mse, kicked = net.step(X, Y)
    assert kicked == 0 and net.kicks == 0


def test_ngn_at_least_as_good_on_xor():
    # regime difícil (nh=2): NGN não deve ser pior que NN na taxa de sucesso
    seeds = list(range(24))
    nn = run_experiment("xor", False, seeds, n_hidden=2, epochs=6000)
    ngn = run_experiment("xor", True, seeds, n_hidden=2, epochs=6000)
    assert ngn["succ"] >= nn["succ"] - 1e-9
    assert ngn["succ"] >= 0.5


def test_forward_shapes_and_bounds():
    net = NeuronGliaNet(2, 3, seed=1)
    h, o = net.forward(X)
    assert h.shape == (4, 3) and o.shape == (4,)
    assert np.all((o > 0) & (o < 1))
