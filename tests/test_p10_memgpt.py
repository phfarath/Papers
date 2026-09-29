"""PT: testes do p10_memgpt — 7 funções, warning 70%, flush 100%+resumo,
heartbeat chaining, adapter MemorySystem.
EN: p10 tests — 7 functions, 70% warning, 100% flush+summary, heartbeat
chaining, MemorySystem adapter."""

from papers.common.conv_data import generate
from papers.common.llm import MockLLM
from papers.common.memory_api import MemoryItem
from papers.p10_memgpt.adapter import MemGPTAdapter
from papers.p10_memgpt.method import FUNCTIONS, MemGPTAgent


def test_seven_functions() -> None:
    assert len(FUNCTIONS) == 7
    for fn in ("send_message", "core_memory_append", "core_memory_replace",
               "archival_memory_insert", "archival_memory_search",
               "recall_memory_search", "heartbeat"):
        assert fn in FUNCTIONS


def test_flush_and_recursive_summary() -> None:
    llm = MockLLM(0)
    a = MemGPTAgent(llm, ctx_tokens=30)
    data = generate(0)
    for t in data.all_turns()[:12]:
        if t.speaker == "user":
            a.step(t.text, t.ts)
    assert a.flushes >= 1
    assert a.warnings >= 1
    assert a.summary  # PT: resumo recursivo não-vazio. EN: non-empty.
    assert len(a.fifo) < 12


def test_archival_insert_and_search() -> None:
    llm = MockLLM(0)
    a = MemGPTAgent(llm)
    a.call("archival_memory_insert", "I live in Lisbon.")
    out = a.call("archival_memory_search", "where do I live")
    assert "Lisbon" in out


def test_adapter_conforms() -> None:
    llm = MockLLM(0)
    ad = MemGPTAdapter(llm)
    data = generate(0)
    for t in data.all_turns()[:8]:
        ad.add(MemoryItem(t.text, t.ts))
    ev = ad.retrieve("favorite breakfast", k=3)
    assert isinstance(ev, list)
    ad.reset()
    assert not ad.agent.fifo
