"""PT: testes do p14_memory_r1 — ops do manager, distilação ≤60, PPO/GRPO
treináveis, adapter. EN: p14 tests — manager ops, ≤60 distillation, trainable
PPO/GRPO, adapter."""

import numpy as np

from papers.common.conv_data import generate
from papers.common.llm import MockLLM
from papers.common.memory_api import MemoryItem
from papers.p14_memory_r1.adapter import MemoryR1Adapter
from papers.p14_memory_r1.method import (
    MAX_CANDIDATES,
    OPS,
    HeuristicManager,
    LinearPolicy,
    UntrainedManager,
    distill,
    exact_match,
)


def test_ops_exist() -> None:
    assert OPS == ["ADD", "UPDATE", "DELETE", "NOOP"]
    assert MAX_CANDIDATES == 60


def test_policy_learns() -> None:
    """PT: política linear melhora numa tarefa bandit simples.
    EN: linear policy improves on a tiny bandit task."""
    pol = LinearPolicy(6, 4, seed=0)
    rng = np.random.RandomState(0)
    x = np.array([1.0, 0, 0, 1, 0.5, 0])
    good = 2
    for _ in range(200):
        a = pol.act(x, rng)
        pol.update(x, a, 1.0 if a == good else -0.1, lr=0.5)
    assert int(np.argmax(pol.probs(x))) == good


def test_heuristic_vs_untrained() -> None:
    """PT: heurístico consolida cidade; não-treinado acumula ambas.
    EN: heuristic consolidates city; untrained keeps both."""
    heur = HeuristicManager()
    untr = UntrainedManager()
    for f in ("user's city is Lisbon", "user's city is Porto"):
        heur.ingest(f)
        untr.ingest(f)
    vals = list(heur.memories.values())
    assert any("Porto" in v for v in vals)
    assert not any("Lisbon" in v for v in vals)
    assert any("Lisbon" in v for v in untr.memories.values())


def test_distillation_le60() -> None:
    cands = [f"fact {i} about thing" for i in range(80)]
    cands[37] = "user's city is Porto"
    out = distill(cands, "where does the user live", k=5)
    assert len(out) == 5
    assert out[0] == "user's city is Porto"


def test_exact_match() -> None:
    assert exact_match("boiled eggs", "I like boiled eggs") == 1.0
    assert exact_match("Porto", "boiled eggs") == 0.0


def test_adapter_conforms() -> None:
    llm = MockLLM(0)
    ad = MemoryR1Adapter(llm)
    for t in generate(0).all_turns()[:10]:
        ad.add(MemoryItem(t.text, t.ts))
    assert isinstance(ad.retrieve("job", k=3), list)
    ad.reset()
    assert not ad.manager.memories
