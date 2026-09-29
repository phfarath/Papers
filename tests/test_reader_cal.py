"""PT: calibração do leitor (qa.answer) — evidência-ouro ≥0.9; evidência
vazia/irrelevante → "I don't know" em ≥0.9 das não-abstenções.

EN: reader calibration — gold evidence ≥0.9; empty/irrelevant evidence →
"I don't know" on ≥0.9 of non-abstention questions.
"""

import re

from papers.common.conv_data import generate
from papers.common.llm import MockLLM
from papers.common.reader import answer_with_evidence, judge_answer

_NAMES_Q = re.compile(r"[A-Z][a-z]+")


def _persona_of(q: str, names: set[str]) -> str | None:
    return next((w for w in _NAMES_Q.findall(q) if w in names), None)


def test_gold_evidence_accuracy() -> None:
    llm = MockLLM(0)
    data = generate(0)
    names = set(data.personas())
    ok = 0
    for q in data.questions:
        n = _persona_of(q.question, names)
        ev = [f"{t.text} ({t.ts.date()})"
              for t in data.by_persona.get(n, [])]
        pred = answer_with_evidence(llm, q.question, ev)
        ok += judge_answer(llm, q.question, q.gold, pred)
    assert ok / len(data.questions) >= 0.9


def test_empty_evidence_abstains() -> None:
    llm = MockLLM(0)
    data = generate(0)
    non_abs = [q for q in data.questions if q.qtype != "abstention"]
    dk = sum(answer_with_evidence(llm, q.question, []) == "I don't know"
             for q in non_abs)
    assert dk / len(non_abs) >= 0.9


def test_irrelevant_evidence_abstains() -> None:
    llm = MockLLM(0)
    data = generate(0)
    filler = ["How is the weather today?", "Nice.", "Tell me a fun fact."]
    non_abs = [q for q in data.questions if q.qtype != "abstention"]
    dk = sum(answer_with_evidence(llm, q.question, filler)
             == "I don't know" for q in non_abs)
    assert dk / len(non_abs) >= 0.9


def test_zep_validity_and_temporal() -> None:
    """PT: evidência estilo Zep ("valid: a → b") honra now/before.
    EN: Zep-style evidence honors now/before cues."""
    llm = MockLLM(0)
    ev = ["Pedro's city is Lisbon (valid: 2024-01-01 → 2024-03-01)",
          "Pedro's city is Porto (valid: 2024-03-01 → present)"]
    assert "Porto" in answer_with_evidence(
        llm, "Where does Pedro live now?", ev)
    assert "Lisbon" in answer_with_evidence(
        llm, "Which city did Pedro live in before moving?", ev)
