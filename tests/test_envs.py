"""PT: testes dos envs — determinismo e solvabilidade via oráculo scriptado.

EN: env tests — determinism and solvability via a scripted oracle.
The oracle is ONLY for tests: mock handlers never see it.
"""

from papers.envs.craft import CraftWorld, run_skill_code
from papers.envs.household import TASKS, HouseholdEnv
from papers.envs.web import TASKS as WEB_TASKS
from papers.envs.web import MiniWeb


# PT: roteiros-oráculo para cada tipo de tarefa (por variante).
# EN: oracle scripts per task type/variant.
def oracle(task) -> list[str]:
    v = task.variant
    t = task.task_type
    if t == "boil_water" and v == "v0":
        return ["go to kitchen", "take kettle", "heat kettle with stove"]
    if t == "boil_water" and v == "v1":
        return ["open door to kitchen", "go to kitchen", "open cabinet",
                "take kettle", "heat kettle with stove", "heat kettle with microwave"]
    if t == "cool_apple" and v == "v0":
        return ["go to living room", "take apple", "go to kitchen",
                "cool apple in fridge"]
    if t == "cool_apple" and v == "v2":
        return ["go to kitchen", "open fridge", "take apple",
                "cool apple in fridge"]
    if t == "plant_seeds" and v == "v0":
        return ["go to bathroom", "open drawer", "take seeds",
                "go to garden", "plant seeds in pot"]
    if t == "plant_seeds" and v == "v1":
        return ["go to bathroom", "open drawer", "take seeds",
                "go to garden", "plant seeds in pot"]
    if t == "clean_mug" and v == "v0":
        return ["go to living room", "take mug", "go to kitchen",
                "wash mug in sink"]
    if t == "clean_mug" and v == "v1":
        return ["go to living room", "take mug", "open door to kitchen",
                "go to kitchen", "wash mug in sink"]
    if t == "heat_soup" and v == "v0":
        return ["go to kitchen", "take soup", "heat soup with microwave"]
    if t == "heat_soup" and v == "v2":
        return ["go to kitchen", "take soup", "heat soup with stove"]
    if t == "two_in_box" and v == "v0":
        return ["go to living room", "take ball", "go to bathroom", "take key",
                "go to living room", "put ball in toolbox", "put key in toolbox"]
    if t == "two_in_box" and v == "v2":
        return ["go to bathroom", "take ball", "go to living room", "take key",
                "put ball in toolbox", "put key in toolbox"]
    raise AssertionError(f"no oracle for {task.task_id}")


def test_household_oracle_solves_all():
    for task in TASKS:
        env = HouseholdEnv(task)
        env.reset()
        res = None
        for act in oracle(task):
            res = env.step(act)
        assert env.done, f"{task.task_id} unsolved by oracle: {res}"


def test_household_deterministic_and_score():
    for task in TASKS:
        e1, e2 = HouseholdEnv(task), HouseholdEnv(task)
        assert e1.reset() == e2.reset()
        assert e1.valid_actions() == e2.valid_actions()
    env = HouseholdEnv(TASKS[0])
    env.reset()
    env.step("go to kitchen")
    env.step("take kettle")
    assert 0 < env.score < 1  # subgoal parcial / partial subgoal


def test_household_traps():
    env = HouseholdEnv(TASKS[6])  # t07 v1: stove broken, kitchen closed
    env.reset()
    r = env.step("go to kitchen")
    assert "closed" in r.observation
    env.step("open door to kitchen")
    env.step("go to kitchen")
    env.step("open cabinet")
    env.step("take kettle")
    r = env.step("heat kettle with stove")
    assert "broken" in r.observation


def test_craft_oracle_and_errors():
    env = CraftWorld("diamond")
    env.reset()
    # PT: 17 logs -> 17 planks; 5 sticks (10 planks) + table (4) + wpick (3).
    # EN: 17 logs -> 17 planks; 5 sticks (10 planks) + table (4) + wpick (3).
    script = (
        ["mine log"] * 17 + ["craft planks"] * 17
        + ["craft stick"] * 5
        + ["craft crafting_table", "craft wooden_pickaxe"]
        + ["mine cobblestone"] * 11
        + ["craft stone_pickaxe", "craft furnace"]
        + ["mine iron_ore"] * 2 + ["smelt iron_ingot"] * 2
        + ["craft iron_pickaxe", "mine diamond"]
    )
    for a in script:
        env.step(a)
    assert env.done


def test_craft_error_feedback():
    env = CraftWorld("diamond")
    env.reset()
    r = env.step("mine diamond")
    assert "without" in r.observation


def test_run_skill_code_sandbox():
    env = CraftWorld("diamond")
    env.reset()
    out = run_skill_code(env, "bot.mine('log', 2)\nprint(bot.inventory())")
    assert "'log': 2" in out
    bad = run_skill_code(env, "open('/etc/passwd')")
    assert "error" in bad


WEB_SCRIPTS = {
    "w01": ["click lnk_search", "type in_q laptop", "click lnk_item_laptop",
            "stop $999"],
    "w02": ["click lnk_search", "type in_q keyboard", "click lnk_item_keyboard",
            "click btn_add_cart", "click lnk_cart", "stop 1"],
    "w03": ["goto http://mini.web/git/admin", "stop Admin Dashboard"],
    "w04": ["goto http://mini.web/forum", "click lnk_login", "type in_user demo",
            "type in_pass demo", "goto http://mini.web/forum/topic/t1",
            "type in_reply hello", "stop 3"],
    "w05": ["goto http://mini.web/git", "click lnk_issues", "stop 2"],
}


def test_web_oracle_solves_all():
    for task in WEB_TASKS:
        env = MiniWeb(task)
        env.reset()
        for act in WEB_SCRIPTS[task.task_id]:
            env.step(act)
        assert env.score() == 1.0, task.task_id


def test_web_gotchas():
    env = MiniWeb(WEB_TASKS[3])
    env.reset()
    env.step("goto http://mini.web/forum/topic/t1")
    r = env.step("type in_reply hello")
    assert "not found" in r.observation  # sem login não há caixa de reply
    env2 = MiniWeb(WEB_TASKS[2])
    env2.reset()
    obs = env2.step("goto http://mini.web/git").observation
    assert "admin" not in obs.lower()  # URL de admin não é linkada
