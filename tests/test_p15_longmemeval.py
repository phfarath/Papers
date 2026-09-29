"""PT: testes do p15 — scale_history monotônico, knobs do DesignRAG, loader
oficial. EN: p15 tests — monotonic scaling, DesignRAG knobs, official
loader."""
from datetime import datetime

from papers.common.conv_data import generate
from papers.common.llm import get_llm
from papers.p15_longmemeval.method import DesignRAG, load_official, scale_history


def test_scale_history_monotonic_and_longer() -> None:
    d = generate(0, n_personas=3)
    big = scale_history(d, extra_filler=3, seed=0)
    assert len(big.all_turns()) > 3 * len(d.all_turns())
    ts = [t.ts for t in big.by_persona[d.personas()[0]]]
    assert ts == sorted(ts)
    assert big.questions == d.questions


def test_facts_value_granularity() -> None:
    m = DesignRAG(value="facts")
    m.add_doc("Pedro here. I live in Lisbon.", datetime(2024, 1, 1))
    assert any("city is Lisbon" in v for v in m.docs.values())


def test_time_aware_expansion() -> None:
    m = DesignRAG(time_aware=True)
    assert "earliest" in m._expand_query("Which city did I live in before?")
    assert m._expand_query("What is my name?") == "What is my name?"


def test_chain_of_note_handler(tmp_path=None) -> None:
    from papers.common.llm import task_prompt
    llm = get_llm("mock", 0)
    body = "QUESTION: q\nEVIDENCE:\n- Pedro's city is Lisbon\n- blah\n"
    out = llm.complete(task_prompt("qa.chain_of_note", body))
    assert "Note: Pedro's city is Lisbon" in out
    assert "irrelevant" in out


def test_official_loader(tmp_path) -> None:
    import json
    item = {"question": "What is my city?", "answer": "Lisbon",
            "question_type": "single-session-user",
            "haystack_dates": ["2024-01-05"],
            "haystack_sessions": [[{"role": "user",
                                   "content": "I live in Lisbon."}]]}
    p = tmp_path / "lme.json"
    p.write_text(json.dumps([item]), encoding="utf-8")
    data = load_official(str(p))
    assert len(data.sessions) == 1 and data.questions[0].qtype == "single-hop"
