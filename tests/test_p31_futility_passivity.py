"""p31 — acumulador de futilidade suprime o comportamento quando ação falha."""

from papers.p31_futility_passivity.method import FutilityAgent, make_schedule, run


def test_determinism_same_seed():
    a = run(0, make_schedule("phases"), True)
    b = run(0, make_schedule("phases"), True)
    assert a[:3] == b[:3]


def test_futile_triggers_passivity():
    att, wins, pas, _ = run(0, make_schedule("futile"), True)
    assert att < 200 and pas > 150 and wins == 0


def test_no_glia_never_passive():
    att, _, pas, _ = run(0, make_schedule("futile"), False)
    assert att == 400 and pas == 0


def test_viable_keeps_most_wins():
    _, wg, _, _ = run(0, make_schedule("viable"), True)
    _, wc, _, _ = run(0, make_schedule("viable"), False)
    assert wg > 0.5 * wc


def test_success_resets_evidence():
    ag = FutilityAgent(glia=True, seed=0)
    for _ in range(3):
        ag.step(0.0)  # falhas acumulam E
    assert ag.E > 0
    ag.passive = 0
    ag.step(1.0)  # sucesso reseta
    assert ag.E == 0.0
