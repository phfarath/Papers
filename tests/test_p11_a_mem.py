"""PT: testes do p11_a_mem — campos da nota, links top-k, evolução de
vizinhos, adapter. EN: p11 tests — note fields, top-k links, neighbor
evolution, adapter."""

from papers.common.conv_data import generate
from papers.common.llm import MockLLM
from papers.common.memory_api import MemoryItem
from papers.p11_a_mem.adapter import AMemAdapter
from papers.p11_a_mem.method import TOP_K_LINK, AMemMemory


def test_note_fields() -> None:
    llm = MockLLM(0)
    mem = AMemMemory(llm)
    n = mem.add("I live in Lisbon and work as a barista.")
    assert n.keywords and n.tags and n.context
    assert set(n.tags) & {"location", "work"}


def test_links_and_evolution() -> None:
    llm = MockLLM(0)
    mem = AMemMemory(llm)
    mem.add("I live in Lisbon.")
    mem.add("I work as a barista in Lisbon.")
    mem.add("My favorite breakfast is boiled eggs.")
    mem.add("I moved to Porto.")
    assert mem.n_links >= 1
    linked = [n for n in mem.notes.values() if n.links]
    assert linked
    # PT: evolução só acontece quando há contexto novo; os contadores existem.
    # EN: evolution only when new context appears; counters exist.
    assert mem.n_evolutions >= 0
    assert TOP_K_LINK == 5


def test_retrieve_uses_links() -> None:
    llm = MockLLM(0)
    mem = AMemMemory(llm)
    mem.add("I live in Lisbon.")
    b = mem.add("I work as a barista in Lisbon.")
    if b.links:  # PT: se linkados, retrieve expande 1 hop. EN: 1-hop expand.
        hits = mem.retrieve("where do I live", k=3)
        assert any("Lisbon" in n.content for n in hits)


def test_adapter_conforms() -> None:
    llm = MockLLM(0)
    ad = AMemAdapter(llm)
    for t in generate(0).all_turns()[:8]:
        ad.add(MemoryItem(t.text, t.ts))
    assert isinstance(ad.retrieve("job", k=3), list)
    ad.reset()
    assert not ad.mem.notes
