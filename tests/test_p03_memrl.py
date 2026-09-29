"""PT: testes do MemRL (p03). EN: MemRL (p03) tests."""

from papers.common.embeddings import HashingEmbedder
from papers.p03_memrl.method import MemRLMemory


def _mem() -> MemRLMemory:
    m = MemRLMemory(embedder=HashingEmbedder())
    m.add("heat water kettle", "use the microwave for the kettle", q=0.9)
    m.add("heat water kettle similar", "use the broken stove", q=0.1)
    m.add("unrelated gardening", "plant seeds in pot", q=0.9)
    return m


def test_phase_a_threshold_delta():
    m = _mem()
    m.delta = 0.99
    idx, _ = m.phase_a("boil the kettle")
    assert idx == []  # nada passa do limiar → só LLM (sem memória)


def test_phase_b_uses_q_not_just_similarity():
    m = _mem()
    idx = m.phase_b("heat the kettle to boil water")
    assert idx and "microwave" in m.items[idx[0]].e


def test_monte_carlo_update():
    m = _mem()
    idx, _ = m.phase_a("heat the kettle")
    m.update(idx[:1], reward=1.0, alpha=0.5)
    assert m.items[idx[0]].q > 0.9  # Q ← Q + α(r − Q)


def test_td_update_option():
    m = _mem()
    q0 = m.items[1].q  # 0.1
    m.update([1], reward=0.0, alpha=0.5, td=True, next_max_q=1.0)
    # Q ← Q + α(0 + γ·1 − Q): com γ=0.9, Q cresce
    assert m.items[1].q > q0


def test_only_injected_updated():
    m = _mem()
    idx, _ = m.phase_a("heat the kettle")
    qs_before = [t.q for t in m.items]
    m.update(idx, reward=1.0)
    for i, t in enumerate(m.items):
        if i not in idx:
            assert t.q == qs_before[i]
