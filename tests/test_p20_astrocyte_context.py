"""PT: testes do p20 — consolidação por contexto, gate do sinal de
contexto, determinismo.

EN: p20 tests — per-context consolidation, context-signal gating,
determinism.
"""
import numpy as np

from papers.p20_astrocyte_context.method import (
    ContextGatedNet,
    make_targets,
    run_sequence,
)

CTX = ["A", "B"]


def test_reversal_targets() -> None:
    rng = np.random.default_rng(0)
    t = make_targets(rng, CTX)
    assert (t["B"] == -t["A"]).all()


def test_astrocyte_consolidates_active_context() -> None:
    rng = np.random.default_rng(1)
    t = make_targets(rng, CTX)
    net = ContextGatedNet(CTX)
    x = rng.choice([-1.0, 1.0], 4)
    for _ in range(40):
        net.trial(x, t["A"], "A")
    # PT: a_A consolidou para perto de f; a_B continua zerado.
    # EN: a_A consolidated toward f; a_B stays at zero.
    assert np.linalg.norm(net.a["A"]) > 0
    assert np.linalg.norm(net.a["B"]) == 0


def test_context_signal_gates_benefit() -> None:
    seq = CTX * 3
    per = {}
    for signal in (True, False):
        errs = []
        for s in range(4):
            rng = np.random.default_rng(10 + s)
            t = make_targets(rng, CTX)
            net = ContextGatedNet(CTX, context_signal=signal)
            early = run_sequence(net, t, seq, 30, rng)
            errs += [e for ph in early[2:] for e in ph]  # re-exposições
        per[signal] = float(np.mean(errs))
    assert per[True] < per[False]


def test_determinism() -> None:
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(5)
        t = make_targets(rng, CTX)
        net = ContextGatedNet(CTX)
        outs.append(run_sequence(net, t, CTX, 10, rng))
    assert outs[0] == outs[1]
