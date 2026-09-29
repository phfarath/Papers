"""PT: testes do p12_zep — campos bitemporais, invalidação sem deleção,
fusão RRF híbrida, MMR, contexto com intervalos de validade.
EN: p12 tests — bitemporal fields, invalidation without deletion, hybrid RRF
fusion, MMR, validity-range context."""

from datetime import datetime

from papers.common.conv_data import generate
from papers.common.llm import MockLLM
from papers.common.memory_api import MemoryItem
from papers.p12_zep.adapter import ZepAdapter
from papers.p12_zep.method import ZepGraph


def test_bitemporal_fields() -> None:
    g = ZepGraph()
    ts = datetime(2025, 1, 1)
    e = g.add("I live in Lisbon.", ts)[0]
    assert e.t_valid == ts and e.t_created == ts
    assert e.t_invalid is None and e.t_expired is None


def test_contradiction_invalidates_not_deletes() -> None:
    g = ZepGraph()
    g.add("I live in Lisbon.", datetime(2025, 1, 1))
    g.add("I moved to Porto.", datetime(2025, 3, 1))
    old = next(e for e in g.edges.values() if "Lisbon" in e.fact)
    new = next(e for e in g.edges.values() if "Porto" in e.fact)
    assert old.t_invalid == datetime(2025, 3, 1)  # invalidada, não apagada
    assert new.t_invalid is None
    assert len(g.edges) == 2


def test_retraction_invalidates_pet() -> None:
    g = ZepGraph()
    g.add("I have a parrot named Kiwi.", datetime(2025, 4, 1))
    g.add("I gave it away.", datetime(2025, 4, 20))
    pet = next(e for e in g.edges.values() if e.slot == "pet")
    assert pet.t_invalid is not None


def test_hybrid_rrf_search() -> None:
    g = ZepGraph()
    for t in generate(0).all_turns():
        g.add(t.text, t.ts)
    hits = g.search("where does the user live", k=3)
    assert hits and hits[0].slot == "city"
    assert hits[0].t_invalid is None  # PT: válido primeiro. EN: valid first.


def test_context_has_validity_ranges() -> None:
    g = ZepGraph()
    g.add("I live in Lisbon.", datetime(2025, 1, 1))
    g.add("I moved to Porto.", datetime(2025, 3, 1))
    ctx = g.context("where do I live")
    assert "valid:" in ctx and "→" in ctx and "present" in ctx


def test_mmr_option_and_adapter() -> None:
    llm = MockLLM(0)
    ad = ZepAdapter(llm, use_mmr=True)
    for t in generate(0).all_turns()[:10]:
        ad.add(MemoryItem(t.text, t.ts))
    assert isinstance(ad.retrieve("job", k=3), list)
    ad.reset()
    assert not ad.graph.edges
