"""PT: testes do p17 — gravação de trajetórias, pools runbook-R, arquivos
runbook-C, premissa falsa. EN: p17 tests — trajectory recording, runbook-R
pools, runbook-C files, false premise."""
import tempfile
from pathlib import Path

from papers.common.llm import get_llm
from papers.p17_longmemeval_v2.method import (
    AgentRunbookC,
    AgentRunbookR,
    answer,
    make_questions,
    record_trajectories,
)


def test_record_trajectories() -> None:
    trajs = record_trajectories(0, noise=0.0)
    assert len(trajs) == 5
    assert all(t.steps for t in trajs)
    assert sum(t.success for t in trajs) >= 4


def test_runbook_r_pools() -> None:
    m = AgentRunbookR()
    for t in record_trajectories(0, noise=0.0):
        m.add_traj(t)
    assert m.pool_of and set(m.pool_of.values()) <= {"state", "events",
                                                     "notes"}
    ev = m.gather("admin dashboard")
    assert any("admin" in e.lower() for e in ev)


def test_runbook_c_files_and_grep() -> None:
    with tempfile.TemporaryDirectory() as td:
        m = AgentRunbookC(Path(td))
        for t in record_trajectories(0, noise=0.0):
            m.add_traj(t)
        assert (Path(td) / "w03.txt").exists()
        ev = m.gather("admin dashboard")
        assert any("admin" in e.lower() for e in ev)


def test_premise_false() -> None:
    llm = get_llm("mock", 0)
    qs = make_questions(record_trajectories(0, noise=0.0))
    pq = [q for q in qs if q.comp == "premise"][0]
    ev = ["step 3 | goto http://mini.web/git/admin | Admin Dashboard"]
    pred = answer(pq, ev, llm)
    assert "false" in pred.lower()
