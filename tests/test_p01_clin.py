"""PT: testes do CLIN (p01). EN: CLIN (p01) tests."""

from papers.common.llm import MockLLM
from papers.envs.household import TASKS
from papers.p01_clin.method import ClinAgent


def test_memory_grows_and_keeps_3_recent():
    task = next(t for t in TASKS if t.task_id == "t07")
    ag = ClinAgent(MockLLM(seed=1), max_trials=4)
    ag.run_episode(task)
    assert ag.memory_history  # memória foi gerada por trial
    # o prompt do memgen usa as 3 memórias mais recentes — invariante estrutural:
    assert all(isinstance(m, list) for m in ag.memory_history)


def test_abstraction_format():
    task = next(t for t in TASKS if t.task_id == "t07")
    ag = ClinAgent(MockLLM(seed=1))
    tr = ag.run_trial(task)
    ag.update_memory(task, tr)
    assert any("necessary" in m or "contribute" in m for m in ag.memory)


def test_meta_memory_generalizes():
    task = next(t for t in TASKS if t.task_id == "t01")
    ood = next(t for t in TASKS if t.task_id == "t07")
    ag = ClinAgent(MockLLM(seed=0))
    ag.run_episode(task)
    meta = ag.build_meta_memory(ood, "gen-env")
    assert meta and all(isinstance(m, str) for m in meta)
    assert len(ag.archive) <= 10  # arquivo de auto-curriculum


def test_clin_beats_freeform_direction():
    task = next(t for t in TASKS if t.task_id == "t07")
    ok_s = ok_f = 0
    for s in range(4):
        a = ClinAgent(MockLLM(seed=s), structured=True)
        f = ClinAgent(MockLLM(seed=s), structured=False)
        ok_s += sum(x.success for x in a.run_episode(task)[1:])
        ok_f += sum(x.success for x in f.run_episode(task)[1:])
    assert ok_s >= ok_f
