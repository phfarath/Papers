"""PT: testes do p05_cer — buffer merge/dedup, distilação, retrieval separado.
EN: p05_cer tests — buffer merge/dedup, distillation, separate retrieval."""

from papers.common.llm import MockLLM
from papers.envs.web import TASKS
from papers.p05_cer.method import run_episode, run_online
from papers.p05_cer.run import build_offline_buffer


def test_offline_buffer_has_dynamics_and_skills() -> None:
    llm = MockLLM(0)
    buf = build_offline_buffer(llm)
    assert len(buf.dynamics) >= 3
    assert len(buf.skills) >= 2
    # PT: placeholders abstratos nas skills. EN: abstract placeholders.
    assert any("{" in s for s in buf.skills)


def test_buffer_dedup() -> None:
    llm = MockLLM(0)
    buf = build_offline_buffer(llm)
    n_dyn, n_skl = len(buf.dynamics), len(buf.skills)
    # PT: mergear o mesmo texto duas vezes não duplica entradas.
    # EN: merging the same text twice adds no duplicates.
    dyn_text = "\n".join(f"- {d}" for d in buf.dynamics.values())
    skl_text = "".join(f"<skill>\n{s}\n</skill>\n"
                       for s in buf.skills.values())
    buf.merge(dyn_text, skl_text)
    assert len(buf.dynamics) == n_dyn
    assert len(buf.skills) == n_skl


def test_retrieval_separate() -> None:
    llm = MockLLM(0)
    buf = build_offline_buffer(llm)
    dyn, skl = buf.retrieve(llm, "Find the secret admin dashboard")
    assert all("http://" in d for d in dyn)
    assert all("steps" in s for s in skl)


def test_cer_beats_react_on_gotcha() -> None:
    """PT: w03 (página admin sem link) só é resolvida com dynamics memory.
    EN: w03 (unlinked admin page) is only solvable via dynamics memory."""
    llm = MockLLM(0)
    buf = build_offline_buffer(llm)
    t = next(t for t in TASKS if t.task_id == "w03")
    assert not run_episode(llm, t).success
    assert run_episode(llm, t, buf).success


def test_online_buffer_grows() -> None:
    llm = MockLLM(0)
    rates, buf = run_online(llm, TASKS, rounds=2)
    assert len(rates) == 2
    assert buf.dynamics or buf.skills
