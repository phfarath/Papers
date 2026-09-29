"""CraftWorld — Minecraft-lite determinístico (Voyager, MUSE-Autoskill).

PT: Árvore tecnológica: log → planks → stick → crafting_table → wooden_pickaxe
→ cobblestone → stone_pickaxe → iron_ore → furnace → iron_ingot → iron_pickaxe
→ diamond (+ laterais: coal, torch, sword). Erros viram mensagens de feedback
("cannot mine iron_ore without stone_pickaxe"), como no Voyager.

API de controle para código gerado (skills): `bot.mine`, `bot.craft`,
`bot.smelt`, `bot.inventory`, `bot.explore`. `run_skill_code` executa código em
namespace restrito (sem builtins perigosos) com orçamento de operações.

EN: tech tree (see above). Errors become feedback messages, Voyager-style.
`run_skill_code` executes generated code in a restricted namespace with an
operation budget (a cheap, deterministic timeout).
"""

from __future__ import annotations

from dataclasses import dataclass

from papers.envs import StepResult

# PT: pré-requisito de ferramenta para minerar. EN: tool prerequisite to mine.
MINE_REQ: dict[str, str | None] = {
    "log": None, "coal": "wooden_pickaxe", "cobblestone": "wooden_pickaxe",
    "iron_ore": "stone_pickaxe", "diamond": "iron_pickaxe",
}
# PT: receitas (insumos consumidos; "*" exige crafting_table no inventário).
# EN: recipes (consumed inputs; "*" requires a crafting_table in inventory).
RECIPES: dict[str, dict[str, int]] = {
    "planks": {"log": 1}, "stick": {"planks": 2},
    "crafting_table": {"planks": 4},
    "wooden_pickaxe": {"planks": 3, "stick": 2, "*": 0},
    "stone_pickaxe": {"cobblestone": 3, "stick": 2, "*": 0},
    "furnace": {"cobblestone": 8},
    "iron_pickaxe": {"iron_ingot": 2, "stick": 1, "*": 0},
    "torch": {"coal": 1, "stick": 1},
    "sword": {"cobblestone": 2, "stick": 1, "*": 0},
}
SMELTS: dict[str, tuple[str, int]] = {"iron_ingot": ("iron_ore", 1)}  # needs furnace
ITEMS = set(MINE_REQ) | set(RECIPES) | set(SMELTS)


class CraftError(Exception):
    """PT: erro de regra do mundo → vira feedback textual. EN: world-rule error."""


@dataclass
class BotAPI:
    """PT: API vista pelo código de skill. Conta chamadas p/ orçamento de tempo.

    EN: API exposed to skill code. Counts calls for a time budget.
    """

    env: CraftWorld
    budget: int = 200

    def _tick(self) -> None:
        self.budget -= 1
        if self.budget <= 0:
            raise CraftError("skill timed out (operation budget exhausted)")

    def mine(self, item: str, n: int = 1) -> str:
        self._tick()
        return self.env._mine(item, n)

    def craft(self, item: str, n: int = 1) -> str:
        self._tick()
        return self.env._craft(item, n)

    def smelt(self, item: str, n: int = 1) -> str:
        self._tick()
        return self.env._smelt(item, n)

    def inventory(self) -> dict[str, int]:
        self._tick()
        return dict(self.env.inv)

    def explore(self) -> str:
        self._tick()
        return "You explore: forest has log; cave has coal/cobblestone/iron_ore/diamond."


class CraftWorld:
    """PT: env deterministico. EN: deterministic env."""

    max_steps = 60

    def __init__(self, target: str = "diamond", target_n: int = 1) -> None:
        self.target = target
        self.target_n = target_n
        self.task_description = f"Obtain {target_n} x {target}."
        self.reset()

    def reset(self) -> str:
        self.inv: dict[str, int] = {}
        self.steps = 0
        self.done = False
        self.score = 0.0
        return "CraftWorld ready. " + BotAPI(self).explore()

    # ---------- mecânicas / mechanics ----------
    def _mine(self, item: str, n: int) -> str:
        if item not in MINE_REQ:
            raise CraftError(f"cannot mine {item}: it is not minable")
        req = MINE_REQ[item]
        if req and self.inv.get(req, 0) < 1:
            raise CraftError(f"cannot mine {item} without {req}")
        self.inv[item] = self.inv.get(item, 0) + n
        return f"mined {n} x {item}"

    def _craft(self, item: str, n: int) -> str:
        if item not in RECIPES:
            raise CraftError(f"cannot craft {item}: unknown recipe")
        need = {k: v * n for k, v in RECIPES[item].items() if k != "*"}
        if "*" in RECIPES[item] and self.inv.get("crafting_table", 0) < 1:
            raise CraftError(f"cannot craft {item} without crafting_table")
        missing = [k for k, v in need.items() if self.inv.get(k, 0) < v]
        if missing:
            raise CraftError(f"cannot craft {item}: missing {', '.join(missing)}")
        for k, v in need.items():
            self.inv[k] -= v
        self.inv[item] = self.inv.get(item, 0) + n
        return f"crafted {n} x {item}"

    def _smelt(self, item: str, n: int) -> str:
        if item not in SMELTS:
            raise CraftError(f"cannot smelt {item}")
        if self.inv.get("furnace", 0) < 1:
            raise CraftError(f"cannot smelt {item} without furnace")
        src, per = SMELTS[item]
        if self.inv.get(src, 0) < per * n:
            raise CraftError(f"cannot smelt {item}: missing {src}")
        self.inv[src] -= per * n
        self.inv[item] = self.inv.get(item, 0) + n
        return f"smelted {n} x {item}"

    # ---------- interface de env ----------
    def valid_actions(self) -> list[str]:
        acts = [f"mine {it}" for it in sorted(MINE_REQ)]
        acts += [f"craft {it}" for it in sorted(RECIPES)]
        acts += [f"smelt {it}" for it in sorted(SMELTS)]
        acts.append("explore")
        return acts

    def _progress(self) -> float:
        if self.inv.get(self.target, 0) >= self.target_n:
            return 1.0
        # PT: progresso = marcos da árvore desbloqueados (heurística simples).
        # EN: progress = unlocked tech-tree milestones (simple heuristic).
        chain = ["log", "planks", "stick", "crafting_table", "wooden_pickaxe",
                 "cobblestone", "stone_pickaxe", "iron_ore", "furnace",
                 "iron_ingot", "iron_pickaxe", "diamond"]
        hit = sum(1 for it in chain if self.inv.get(it, 0) > 0)
        return 0.9 * hit / len(chain)

    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult("Episode is over.", 0.0, True)
        self.steps += 1
        bot = BotAPI(self)
        parts = action.strip().split()
        try:
            if parts[0] == "mine":
                msg = bot.mine(parts[1])
            elif parts[0] == "craft":
                msg = bot.craft(parts[1])
            elif parts[0] == "smelt":
                msg = bot.smelt(parts[1])
            elif parts[0] == "explore":
                msg = bot.explore()
            else:
                msg = f"unknown action {action!r}"
        except CraftError as e:
            msg = str(e)
        except (IndexError, TypeError):
            msg = f"malformed action {action!r}"
        self.score = self._progress()
        self.done = self.inv.get(self.target, 0) >= self.target_n
        obs = msg + f" Inventory: {self.inv}"
        return StepResult(obs, 1.0 if self.done else 0.0,
                          self.done or self.steps >= self.max_steps)


def run_skill_code(env: CraftWorld, code: str) -> str:
    """PT: executa código de skill num namespace restrito (sem builtins perigosos).

    EN: executes skill code in a restricted namespace (no dangerous builtins).
    Returns captured output or the error message (feedback loop à la Voyager).
    """
    import contextlib
    import io

    bot = BotAPI(env)
    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "sum": sum,
        "enumerate": enumerate, "print": print, "int": int, "str": str,
        "bool": bool, "dict": dict, "list": list, "set": set, "abs": abs,
        "Exception": Exception, "for": None,
    }
    ns: dict = {"__builtins__": safe_builtins, "bot": bot}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)  # noqa: S102 - sandboxed namespace / namespace restrito
    except Exception as e:  # feedback vira texto, como no Voyager
        return f"error: {e}"
    return buf.getvalue().strip() or "ok"
