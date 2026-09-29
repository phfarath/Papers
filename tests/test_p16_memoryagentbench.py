"""PT: testes do p16 — 4 competências, ingestão por chunks, coverage LRU.
EN: p16 tests — 4 competencies, chunked ingestion, LRU coverage."""
from papers.common.llm import get_llm
from papers.p16_memoryagentbench.method import (
    BenchItem,
    coverage_score,
    generate,
    score,
)


def test_bench_has_all_competencies() -> None:
    b = generate(0, n_personas=3)
    comps = {i.comp for i in b.items}
    assert comps == {"AR", "TTL", "LRU", "SF"}
    assert all(i.stream and i.question and i.gold for i in b.items)


def test_sf_stream_overwrites() -> None:
    b = generate(0, n_personas=2)
    sf = [i for i in b.items if i.comp == "SF"]
    assert len(sf) >= 3 * 2 - 2  # PT: 3 SF por persona. EN: 3 per persona.
    # PT: o stream SF contém um update posterior. EN: SF stream contains a
    # later update.
    assert any("Update:" in t.text or "moved" in t.text
               for t in sf[0].stream)


def test_coverage_score() -> None:
    assert coverage_score("Lisbon, Porto", "Lisbon and Porto") == 1.0
    assert coverage_score("Lisbon, Porto", "Lisbon") == 0.5


def test_score_uses_judge_for_ar() -> None:
    llm = get_llm("mock", 0)
    it = BenchItem("AR", [], "q", "pizza")
    assert score(llm, it, "pizza") == 1.0
    assert score(llm, it, "I don't know") == 0.0
