"""PT: testes do Reflexion (p02). EN: Reflexion (p02) tests."""

from papers.common.llm import MockLLM
from papers.envs.household import TASKS
from papers.p02_reflexion.method import (
    ReflexionAgent,
    derive_internal_tests,
    reflexion_code_loop,
    run_tests,
)
from papers.p02_reflexion.prompts import CODE_PROBLEMS


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
        reflexion_code_loop(MockLLM(seed=s), t, use_reflection=True).hidden_ok
        for t in CODE_PROBLEMS for s in range(3)
    )
    base = sum(
        reflexion_code_loop(MockLLM(seed=s), t, use_reflection=False).hidden_ok
        for t in CODE_PROBLEMS for s in range(3)
    )
    assert wins > base


def test_internal_tests_derived_only_from_docstring():
    prob = CODE_PROBLEMS[0]  # double: exemplos double(3)=6, double(0)=0
    tests = derive_internal_tests(MockLLM(0), prob)
    assert tests and all(t in prob.doc.replace(" = ", " == ") or "==" in t
                         for t in tests)
    assert "double(-2)" not in "\n".join(tests)  # hidden nunca vaza


def test_run_tests():
    ok, _ = run_tests("def double(x):\n    return 2*x", ["double(3) == 6"])
    assert ok
    ok, rep = run_tests("def double(x):\n    return 2+x", ["double(3) == 6"])
    assert not ok and "FAIL" in rep
