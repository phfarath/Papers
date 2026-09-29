"""PT: testes do p06_voyager — currículo, skill library, critic, iterative loop.
EN: p06_voyager tests — curriculum, skill library, critic, iterative loop."""

from papers.common.llm import MockLLM
from papers.envs.craft import CraftWorld
from papers.p06_voyager.method import (
    Skill,
    SkillLibrary,
    VoyagerAgent,
    run_episode,
)


def test_curriculum_proposes_next_milestone() -> None:
    llm = MockLLM(0)
    env = CraftWorld("diamond")
    a = VoyagerAgent(llm)
    assert a.propose_task(env) == "mine log"
    env._mine("log", 1)
    assert a.propose_task(env) == "craft planks"


def test_skill_library_retrieval() -> None:
    lib = SkillLibrary()
    lib.add(Skill("planks", "obtain planks: craft from log",
                  "bot.craft('planks', 1)"))
    lib.add(Skill("diamond", "obtain diamond: mine with iron_pickaxe",
                  "bot.mine('diamond', 1)"))
    llm = MockLLM(0)
    hits = lib.retrieve(llm, "craft planks")
    assert hits and hits[0].name == "planks"


def test_skill_stored_only_on_success() -> None:
    llm = MockLLM(0)
    env = CraftWorld("diamond")
    a = VoyagerAgent(llm)
    att = a.solve_task(env, "mine log")
    assert att.success and len(a.library) == 1
    # PT: item inexistente → nunca verificado → não vira skill.
    # EN: nonexistent item → never verified → never stored.
    att = a.solve_task(env, "mine unobtainium")
    assert not att.success and len(a.library) == 1


def test_iterative_rounds_cap() -> None:
    llm = MockLLM(0)
    env = CraftWorld("diamond")
    a = VoyagerAgent(llm)
    att = a.solve_task(env, "mine unobtainium")
    assert not att.success and att.rounds == 4  # MAX_ROUNDS


def test_full_run_reaches_deep_milestones() -> None:
    llm = MockLLM(0)
    env = CraftWorld("diamond")
    hist, lib = run_episode(llm, env, 12)
    items = set(hist[-1].items)
    assert {"wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
            "diamond"} <= items
    assert len(lib) >= 4


def test_zero_shot_warm_beats_cold() -> None:
    llm = MockLLM(0)
    _, lib = run_episode(llm, CraftWorld("diamond"), 12)
    env_w = CraftWorld("torch")
    hist_w, _ = run_episode(llm, env_w, 6, library=lib)
    env_c = CraftWorld("torch")
    hist_c, _ = run_episode(llm, env_c, 6)
    rw = sum(it.attempts[-1].rounds for it in hist_w)
    rc = sum(it.attempts[-1].rounds for it in hist_c)
    assert rw <= rc
