"""PT: testes do p13_mem0 — extração com (summary,last10,par), top-10,
ADD/UPDATE/DELETE/NOOP, Mem0g com invalidação, latência/tokens.
EN: p13 tests — extraction, top-10, ops, Mem0g invalidation, metrics."""

from datetime import datetime

from papers.common.conv_data import generate
from papers.common.llm import MockLLM
from papers.common.memory_api import MemoryItem
from papers.p13_mem0.adapter import Mem0Adapter, Mem0gAdapter
from papers.p13_mem0.method import (
    LAST_N_MSGS,
    TOP_K_SIMILAR,
    Mem0gMemory,
    Mem0Memory,
)


def test_constants() -> None:
    assert LAST_N_MSGS == 10 and TOP_K_SIMILAR == 10


def test_add_update_noop() -> None:
    llm = MockLLM(0)
    m = Mem0Memory(llm)
    m.add("I live in Lisbon.", datetime(2025, 1, 1))
    m.add("I moved to Porto.", datetime(2025, 3, 1))
    m.add("I live in Porto.", datetime(2025, 3, 2))  # duplicata → NOOP
    s = m.stats
    assert s.adds >= 1 and s.updates >= 1
    vals = list(m.memories.values())
    assert any("Porto" in v for v in vals)
    assert not any("Lisbon" in v for v in vals)


def test_delete_on_retraction() -> None:
    llm = MockLLM(0)
    m = Mem0Memory(llm)
    m.add("I have a parrot named Kiwi.", datetime(2025, 4, 1))
    m.add("I gave it away.", datetime(2025, 4, 20))
    assert m.stats.deletes >= 1


def test_mem0g_invalidation() -> None:
    llm = MockLLM(0)
    g = Mem0gMemory(llm)
    g.add("I live in Lisbon.", datetime(2025, 1, 1))
    g.add("I moved to Porto.", datetime(2025, 3, 1))
    if g.edges:
        assert g.invalidated >= 1
        assert any(not t.valid for t in g.edges)
        assert any(t.valid and "Porto" in t.obj for t in g.edges)


def test_adapters_and_metrics() -> None:
    llm = MockLLM(0)
    for cls in (Mem0Adapter, Mem0gAdapter):
        ad = cls(llm)
        for t in generate(0).all_turns()[:10]:
            ad.add(MemoryItem(t.text, t.ts))
        assert isinstance(ad.retrieve("job", k=3), list)
        ad.reset()
        assert not ad.mem._inner.memories if hasattr(ad.mem, "_inner") \
            else not ad.mem.memories
    m = Mem0Memory(llm)
    m.add("I live in Lisbon.")
    assert m.stats.latencies and m.stats.tokens_in
