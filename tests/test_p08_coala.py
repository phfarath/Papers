"""PT: testes do p08_coala — taxonomia p01–p18 + agente com ciclo CoALA.
EN: p08_coala tests — p01–p18 taxonomy + CoALA-cycle agent."""

import re
from pathlib import Path

from papers.common.llm import MockLLM
from papers.envs.household import TASKS
from papers.p08_coala.method import CoALAAgent
from papers.p08_coala.taxonomy import MODULES, describe


def test_taxonomy_covers_all_root_index_folders() -> None:
    """PT: toda pasta pNN listada no índice raiz aparece na taxonomia.
    EN: every pNN folder listed in the root index appears in the taxonomy."""
    readme = (Path(__file__).resolve().parent.parent
              / "README.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"`(p\d+_\w+)`", readme))
    covered = {m.folder for m in describe()}
    assert listed <= covered, listed - covered
    assert len(MODULES) == 18


def test_coala_agent_learns_semantic_rule() -> None:
    llm = MockLLM(0)
    agent = CoALAAgent(llm, seed=0)
    # PT: roda trials numa tarefa solúvel e espera regra semântica aprendida.
    # EN: run trials on a solvable task and expect a learned semantic rule.
    for _ in range(3):
        agent.run_episode(TASKS[0])
    assert agent.ltm.episodic  # PT: episodic sempre registrada. EN: always.
    assert any("should" in s for s in agent.ltm.semantic)


def test_internal_actions_run() -> None:
    llm = MockLLM(0)
    agent = CoALAAgent(llm, seed=0)
    r = agent.run_episode(TASKS[0])
    assert agent.wm.plan.startswith("plan:")
    assert r.steps > 0
