"""PT: testes do Reflexion (p02). EN: Reflexion (p02) tests."""

from papers.common.llm import MockLLM
from papers.envs.household import TASKS
from papers.p02_reflexion.method import ReflexionAgent, reflexion_code_loop, run_tests
from papers.p02_reflexion.prompts import CODE_TASKS


def test_reflexion_improves_over_baseline_on_hard_task():
    # t07: stove quebrado + porta fechada — reflexão deve ajudar.
    task = next(t for t in TASKS if t.task_id == "t07")
    wins_r = wins_b = 0
    for seed in range(8):
        r = ReflexionAgent(MockLLM(seed=seed)).run_task(task)
        b = ReflexionAgent(MockLLM(seed=seed), use_reflection=False).run_task(task)
        wins_r += any(t.success for t in r)
        wins_b += any(t.success for t in b)
    assert wins_r >= wins_b and wins_r >= 3


def test_omega_bound():
    task = TASKS[5]
    ag = ReflexionAgent(MockLLM(seed=0), omega=2)
    ag.run_task(task)
    assert len(ag.mem) <= 2


def test_code_reflexion_fixes_bugs():
    wins = sum(
        reflexion_code_loop(MockLLM(seed=s), t, use_reflection=True)[0]
        for t in CODE_TASKS for s in range(3)
    )
    base = sum(
        reflexion_code_loop(MockLLM(seed=s), t, use_reflection=False)[0]
        for t in CODE_TASKS for s in range(3)
    )
    assert wins > base


def test_run_tests():
    ok, _ = run_tests("def double(x):\n    return 2*x", ["double(3) == 6"])
    assert ok
    ok, rep = run_tests("def double(x):\n    return 2+x", ["double(3) == 6"])
    assert not ok and "FAIL" in rep
