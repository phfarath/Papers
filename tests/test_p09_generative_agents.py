"""PT: testes do p09_generative_agents — fórmulas do retrieval e gatilho da
reflexão. EN: p09 tests — retrieval scoring formulas and reflection trigger."""

from papers.common.llm import MockLLM
from papers.p09_generative_agents.method import (
    RECENCY_DECAY,
    REFLECT_LAST_N,
    REFLECT_THRESHOLD,
    Agent,
    MemoryStream,
    run_simulation,
)


def test_recency_decay_and_weights() -> None:
    """PT: score usa 0.995^horas e pesos 1/1/1 com min-max. EN: score uses
    0.995^hours and weights 1/1/1 min-max normalized."""
    assert RECENCY_DECAY == 0.995
    llm = MockLLM(0)
    s = MemoryStream()
    s.add(llm, "went to the plaza", now=0.0)
    s.add(llm, "invited to the party", now=0.0)
    top = s.retrieve("party", now=1.0, k=1)
    assert "party" in top[0].text
    assert top[0].importance > 1  # PT: party é importante. EN: party is high.


def test_importance_scale() -> None:
    llm = MockLLM(0)
    s = MemoryStream()
    m1 = s.add(llm, "brushed teeth", now=0.0)
    m2 = s.add(llm, "invited to a birthday party", now=0.0)
    assert 1 <= m1.importance <= 10
    assert m2.importance > m1.importance


def test_reflection_trigger() -> None:
    llm = MockLLM(0)
    a = Agent("Ana", "baker")
    # PT: acumula importância >150 → reflexão dispara. EN: exceed threshold.
    for _ in range(20):
        a.stream.add(llm, "invited to the party with friends", now=0.0)
    assert a.stream.since_reflect > REFLECT_THRESHOLD
    a.maybe_reflect(llm, now=1.0)
    assert a.n_reflections > 0
    assert any(m.kind == "reflection" for m in a.stream.mems)
    assert REFLECT_LAST_N == 100


def test_diffusion_happens() -> None:
    llm = MockLLM(0)
    curve = run_simulation(llm, ticks=10, use_planning=True,
                           use_reflection=True, seed=0)
    assert curve[-1] > curve[0]
