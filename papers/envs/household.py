"""HouseholdEnv — ScienceWorld/ALFWorld-lite deterministico.

PT: Cômodos ligados por um hallway, portas que podem estar fechadas, containers
que escondem itens, dispositivos que podem estar quebrados e ações distratoras.
Cada tarefa tem subgoals; `score` é a fração cumprida e sucesso dá reward=1.0.

O handler mock compartilhado "household.act" simula o que um LLM razoável faria:
pontua cada ação válida por overlap com a tarefa/observação, dá bônus forte a
ações/objetos citados como necessários/úteis na memória, penaliza ações citadas
como inúteis/quebradas e penaliza repetição. REGRA DE HONESTIDADE: o handler só
lê o prompt — nunca o estado interno do env.

EN: deterministic ScienceWorld/ALFWorld-lite. Rooms connected by a hallway,
doors that may be closed, containers hiding items, devices that may be broken
and distractor actions. Each task has subgoals; `score` is the fraction done
and success gives reward=1.0.

The shared "household.act" mock handler simulates what a reasonable LLM would
do: scores each valid action by overlap with task/observation, strongly bonuses
actions/objects cited as necessary/useful in memory, penalizes actions cited as
useless/broken, and penalizes repetition. HONESTY RULE: the handler only reads
the prompt — never env internals.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from papers.common.llm import mock_handler
from papers.common.utils import token_set
from papers.envs import StepResult

ROOMS = ["hallway", "kitchen", "garden", "bathroom", "living room"]
CONTAINER_ROOM = {
    "fridge": "kitchen",
    "cabinet": "kitchen",
    "drawer": "bathroom",
    "toolbox": "living room",
    "pot": "garden",
}
DEVICE_ROOM = {"stove": "kitchen", "microwave": "kitchen", "sink": "kitchen"}


@dataclass
class Variant:
    """PT: layout de uma variante do mundo. EN: one world-layout variant."""

    items: dict[str, str]  # item -> room, container name ou "pot"
    closed_containers: set[str] = field(default_factory=set)
    closed_doors: set[str] = field(default_factory=set)  # rooms blocked
    broken_devices: set[str] = field(default_factory=set)


# PT: 3+ variantes de ambiente (layouts diferentes, truques diferentes).
# EN: 3+ environment variants (different layouts, different traps).
VARIANTS: dict[str, Variant] = {
    "v0": Variant(
        items={
            "kettle": "kitchen", "apple": "living room", "mug": "living room",
            "seeds": "drawer", "soup": "kitchen", "ball": "living room",
            "key": "bathroom", "lighter": "living room",
            "newspaper": "living room", "pillow": "bathroom", "pen": "kitchen",
            "book": "living room", "remote": "kitchen", "candle": "bathroom",
            "toothbrush": "bathroom", "cushion": "living room",
        },
        closed_containers={"drawer"},
    ),
    "v1": Variant(  # fogão quebrado + porta fechada + item em container fechado
        items={
            "kettle": "cabinet", "apple": "living room", "mug": "living room",
            "seeds": "drawer", "soup": "fridge", "ball": "garden",
            "key": "kitchen", "lighter": "bathroom",
            "newspaper": "living room", "pillow": "bedroom", "pen": "kitchen",
            "book": "bathroom", "remote": "living room", "candle": "garden",
            "toothbrush": "bathroom", "cushion": "living room",
        },
        closed_containers={"drawer", "cabinet", "fridge"},
        closed_doors={"kitchen"},
        broken_devices={"stove"},
    ),
    "v2": Variant(  # micro-ondas quebrado + geladeira fechada + jardim fechado
        items={
            "kettle": "kitchen", "apple": "fridge", "mug": "cabinet",
            "seeds": "kitchen", "soup": "kitchen", "ball": "bathroom",
            "key": "living room", "lighter": "kitchen",
            "newspaper": "bathroom", "pillow": "living room", "pen": "garden",
            "book": "kitchen", "remote": "bathroom", "candle": "living room",
            "toothbrush": "kitchen", "cushion": "living room",
        },
        closed_containers={"fridge", "cabinet"},
        closed_doors={"garden"},
        broken_devices={"microwave"},
    ),
}
# PT: "bedroom" na v1 não existe — saneamento: mova itens para hallway.
# EN: "bedroom" in v1 does not exist — sanitize: move such items to hallway.
for _v in VARIANTS.values():
    for _it, _loc in list(_v.items.items()):
        if _loc not in ROOMS and _loc not in CONTAINER_ROOM:
            _v.items[_it] = "hallway"


@dataclass
class TaskSpec:
    """PT: uma tarefa (tipo + variante). EN: one task (type + variant)."""

    task_id: str
    task_type: str
    variant: str
    description: str
    target_items: tuple[str, ...]
    split: str  # "id" (in-distribution) ou "ood"


TASKS: list[TaskSpec] = [
    TaskSpec("t01", "boil_water", "v0", "Boil water (make the kettle hot).",
             ("kettle",), "id"),
    TaskSpec("t02", "cool_apple", "v0", "Cool the apple (make it cold).",
             ("apple",), "id"),
    TaskSpec("t03", "plant_seeds", "v0", "Plant the seeds in the pot.",
             ("seeds",), "id"),
    TaskSpec("t04", "clean_mug", "v0", "Clean the mug.",
             ("mug",), "id"),
    TaskSpec("t05", "heat_soup", "v0", "Heat the soup.",
             ("soup",), "id"),
    TaskSpec("t06", "two_in_box", "v0", "Put the ball and the key in the toolbox.",
             ("ball", "key"), "id"),
    # PT: out-of-distribution: mesmos tipos, ambientes com truques diferentes.
    # EN: out-of-distribution: same types, environments with different traps.
    TaskSpec("t07", "boil_water", "v1", "Boil water (make the kettle hot).",
             ("kettle",), "ood"),
    TaskSpec("t08", "cool_apple", "v2", "Cool the apple (make it cold).",
             ("apple",), "ood"),
    TaskSpec("t09", "plant_seeds", "v1", "Plant the seeds in the pot.",
             ("seeds",), "ood"),
    TaskSpec("t10", "heat_soup", "v2", "Heat the soup.",
             ("soup",), "ood"),
    TaskSpec("t11", "clean_mug", "v1", "Clean the mug.",
             ("mug",), "ood"),
    TaskSpec("t12", "two_in_box", "v2", "Put the ball and the key in the toolbox.",
             ("ball", "key"), "ood"),
]

_GOAL = {
    "boil_water": ("kettle", "hot"), "cool_apple": ("apple", "cold"),
    "heat_soup": ("soup", "hot"),
}


class HouseholdEnv:
    """PT: env deterministico estilo ScienceWorld. EN: deterministic env."""

    max_steps = 30

    def __init__(self, task: TaskSpec) -> None:
        self.spec = task
        self.task_description = task.description
        self._variant = VARIANTS[task.variant]
        self._reset_state()

    def _reset_state(self) -> None:
        v = self._variant
        self.room = "hallway"
        self.loc = dict(v.items)  # item -> room | container | "inventory"
        self.temp = {it: "normal" for it in v.items}
        self.cleaned: set[str] = set()
        self.planted: set[str] = set()
        self.container_open = {c: c not in v.closed_containers for c in CONTAINER_ROOM}
        self.door_open = {r: r not in v.closed_doors for r in ROOMS if r != "hallway"}
        self.steps = 0
        self.score = 0.0
        self.done = False

    def reset(self) -> str:
        self._reset_state()
        return self._observe()

    # ---------- dinâmica / dynamics ----------
    def _item_room(self, item: str) -> str | None:
        loc = self.loc.get(item)
        if loc == "inventory":
            return self.room
        if loc in CONTAINER_ROOM:
            return CONTAINER_ROOM[loc]
        return loc

    def _visible(self, item: str) -> bool:
        loc = self.loc.get(item)
        if loc == "inventory":
            return True
        if loc in CONTAINER_ROOM:
            return self.container_open[loc] and self.room == CONTAINER_ROOM[loc]
        return loc == self.room

    def _observe(self) -> str:
        seen: list[str] = []
        for c, room in CONTAINER_ROOM.items():
            if room == self.room:
                seen.append(f"{c} ({'open' if self.container_open[c] else 'closed'})")
        for d, room in DEVICE_ROOM.items():
            if room == self.room:
                seen.append(d)
        for it, loc in self.loc.items():
            if self._visible(it) and loc != "inventory":
                seen.append(it)
        if self.room == "garden":
            seen.append("pot (open)")
        inv = [it for it, loc in self.loc.items() if loc == "inventory"]
        obs = f"You are in the {self.room}. You see: {', '.join(sorted(set(seen))) or 'nothing'}."
        if inv:
            obs += f" You are carrying: {', '.join(sorted(inv))}."
        return obs

    def valid_actions(self) -> list[str]:
        acts = [f"go to {r}" for r in ROOMS if r != self.room]
        acts += [f"open door to {r}" for r, o in self.door_open.items() if not o]
        for c, room in CONTAINER_ROOM.items():
            if room == self.room and not self.container_open[c]:
                acts.append(f"open {c}")
        for it in sorted(self.loc):
            if self._visible(it) and self.loc[it] != "inventory":
                acts.append(f"take {it}")
        inv = [it for it, loc in self.loc.items() if loc == "inventory"]
        for it in inv:
            for c, room in CONTAINER_ROOM.items():
                if room == self.room and self.container_open[c] and c != "pot":
                    acts.append(f"put {it} in {c}")
            for d, room in DEVICE_ROOM.items():
                if room == self.room and d != "sink":
                    acts.append(f"heat {it} with {d}")
            if "lighter" in inv and it != "lighter":
                acts.append(f"heat {it} with lighter")
            if self.room == "kitchen" and self.container_open["fridge"]:
                acts.append(f"cool {it} in fridge")
            if self.room == "kitchen":
                acts.append(f"wash {it} in sink")
            if self.room == "garden" and self.container_open["pot"]:
                acts.append(f"plant {it} in pot")
        return sorted(acts)

    # ---------- subgoals ----------
    def _subgoals(self) -> list[bool]:
        t = self.spec.task_type
        if t in ("boil_water", "cool_apple", "heat_soup"):
            item, want = _GOAL[t]
            loc_room = self._item_room(item)
            return [
                self.loc[item] == "inventory" or self.temp[item] == want
                or loc_room == self.room or self.temp[item] == want,
                self.temp[item] == want,
            ] if t != "cool_apple" else [
                self.loc[item] == "inventory" or self.temp[item] == want,
                self.temp[item] == want,
            ]
        if t == "plant_seeds":
            return [
                self.loc["seeds"] == "inventory" or "seeds" in self.planted,
                "seeds" in self.planted,
            ]
        if t == "clean_mug":
            return [
                self.loc["mug"] == "inventory" or "mug" in self.cleaned,
                "mug" in self.cleaned,
            ]
        if t == "two_in_box":
            return [
                sum(1 for it in self.spec.target_items if self.loc[it] == "toolbox") >= 1,
                all(self.loc[it] == "toolbox" for it in self.spec.target_items),
            ]
        return [False]

    def _update_score(self) -> None:
        subs = self._subgoals()
        self.score = sum(subs) / len(subs)
        if all(subs):
            self.done = True

    # ---------- passo / step ----------
    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult("Episode is over.", 0.0, True)
        self.steps += 1
        a = action.strip()
        obs = self._apply(a)
        self._update_score()
        reward = 1.0 if self.done else 0.0
        done = self.done or self.steps >= self.max_steps
        return StepResult(obs + " " + self._observe(), reward, done, {"action": a})

    def _apply(self, a: str) -> str:
        if a.startswith("go to "):
            dest = a[6:]
            if dest not in ROOMS:
                return f"There is no room called {dest}."
            if dest != "hallway" and not self.door_open.get(dest, True):
                return f"The door to the {dest} is closed."
            self.room = dest
            return f"You walk to the {dest}."
        if a.startswith("open door to "):
            dest = a[13:]
            if dest in self.door_open and not self.door_open[dest]:
                self.door_open[dest] = True
                return f"You open the door to the {dest}."
            return "Nothing happens."
        if a.startswith("open "):
            c = a[5:]
            if c in self.container_open and not self.container_open[c] \
                    and CONTAINER_ROOM[c] == self.room:
                self.container_open[c] = True
                inside = [i for i, loc in self.loc.items() if loc == c]
                extra = f" Inside you see: {', '.join(inside)}." if inside else ""
                return f"You open the {c}.{extra}"
            return "Nothing happens."
        if a.startswith("take "):
            it = a[5:]
            if self._visible(it) and self.loc.get(it) != "inventory":
                self.loc[it] = "inventory"
                return f"You take the {it}."
            return f"You cannot take the {it}."
        if a.startswith("put ") and " in " in a:
            it, c = a[4:].split(" in ", 1)
            if self.loc.get(it) == "inventory" and self.container_open.get(c) \
                    and CONTAINER_ROOM.get(c) == self.room:
                self.loc[it] = c
                return f"You put the {it} in the {c}."
            return f"You cannot put the {it} there."
        if a.startswith("heat ") and " with " in a:
            it, d = a[5:].split(" with ", 1)
            if self.loc.get(it) != "inventory":
                return f"You need to hold the {it} first."
            ok_device = (
                (d in DEVICE_ROOM and DEVICE_ROOM[d] == self.room)
                or (d == "lighter" and self.loc.get("lighter") == "inventory")
            )
            if not ok_device:
                return f"There is no usable {d} here."
            if d in self._variant.broken_devices:
                return f"The {d} is broken. Nothing happens."
            self.temp[it] = "hot"
            return f"You heat the {it} with the {d}."
        if a.startswith("cool ") and a.endswith(" in fridge"):
            it = a[5:-10]
            if self.loc.get(it) == "inventory" and self.room == "kitchen" \
                    and self.container_open["fridge"]:
                self.temp[it] = "cold"
                return f"You cool the {it} in the fridge."
            return f"You cannot cool the {it}."
        if a.startswith("wash ") and a.endswith(" in sink"):
            it = a[5:-8]
            if self.loc.get(it) == "inventory" and self.room == "kitchen":
                self.cleaned.add(it)
                return f"You wash the {it} in the sink."
            return f"You cannot wash the {it}."
        if a.startswith("plant ") and a.endswith(" in pot"):
            it = a[6:-7]
            if self.loc.get(it) == "inventory" and self.room == "garden":
                self.planted.add(it)
                self.loc[it] = "pot"
                return f"You plant the {it} in the pot."
            return f"You cannot plant the {it}."
        return "Nothing happens."


# ---------------------------------------------------------------------------
# Política mock compartilhada (task "household.act")
# Shared mock policy — honest: prompt text only.
# ---------------------------------------------------------------------------

_GOOD_MARKS = (
    "necessary", "should", "helpful", "useful", "use ", "try", "recommend",
    "instead", "next time", "worked", "helped", "need to", "must",
)
_BAD_MARKS = (
    "does not contribute", "not contribute", "useless", "broken", "avoid",
    "failed", "do not", "don't", "never", "pointless", "waste",
)


_LANDMARK_ROOM = {
    "fridge": "kitchen", "sink": "kitchen", "stove": "kitchen",
    "microwave": "kitchen", "cabinet": "kitchen", "pot": "garden",
    "drawer": "bathroom", "toolbox": "living",
}


def _action_key_tokens(action: str) -> set[str]:
    return token_set(action)


def score_actions(
    task: str, observation: str, actions: list[str], memory_lines: list[str],
    recent: list[str], rng: random.Random, noise: float = 0.9,
) -> str:
    """PT: pontua cada ação e devolve a melhor (argmax + ruído controlado).

    PT (detalhe): a heurística imita um LLM — overlap com a tarefa e com os
    objetos visíveis, bônus/penalidade vindos do TEXTO da memória e penalidade
    de repetição. Nada aqui lê estado interno do env.
    EN: scores each action and returns argmax + controlled noise. The heuristic
    imitates an LLM — overlap with task and visible objects, bonus/penalty from
    memory TEXT and a repetition penalty. Nothing reads env internals.
    """
    task_toks = token_set(task)
    obs_toks = token_set(observation)
    # PT: o agente "sabe" o que carrega via texto da observação (You are carrying).
    # EN: the agent "knows" what it carries via the observation text.
    carrying = "carrying:" in observation
    # PT: verbos plausíveis da tarefa — um LLM filtraria ações cujo verbo não
    # serve ao objetivo ("put the mug in the toolbox" não limpa a caneca).
    # EN: plausible task verbs — an LLM filters actions whose verb cannot serve
    # the goal ("put the mug in the toolbox" doesn't clean the mug).
    verbs = {"go", "open", "take", "explore"}
    hints = {
        "heat": {"heat", "boil", "hot", "warm"}, "cool": {"cool", "cold", "chill"},
        "wash": {"clean", "wash", "dirty"}, "plant": {"plant", "grow", "seed"},
        "put": {"put", "store", "place", "inside"},
    }
    for verb, words in hints.items():
        if task_toks & words:
            verbs.add(verb)
    best_a, best_s = actions[0], -float("inf")
    for act in actions:
        at = _action_key_tokens(act)
        s = 1.0 * (len(at & task_toks) / max(len(at), 1))
        s += 0.15 * len(at & obs_toks)
        if act.split(" ", 1)[0] not in verbs:
            s -= 0.9  # PT: verbo incompatível com a tarefa. EN: verb mismatch.
        if act.startswith("go to "):
            s += 0.08  # PT: nudge de exploração. EN: exploration nudge.
            # PT: "senso comum" do mock: marcos → cômodo (fridge→kitchen etc.).
            # Uma boa memória cita o marco; ir ao cômodo certo é promovido.
            # EN: mock "common sense": landmark → room map. Good memory citing
            # a landmark promotes going to the right room.
            for line in memory_lines:
                if not any(m in line.lower() for m in _GOOD_MARKS):
                    continue
                lt = token_set(line)
                for mark, room in _LANDMARK_ROOM.items():
                    if mark in lt and act.endswith(room):
                        s += 1.5
        if act.startswith("take ") and carrying and not (at & task_toks):
            s -= 0.35  # PT: mãos ocupadas com o que importa. EN: hands busy.
        # PT: carregando o item da tarefa → ignorar ações sobre outros objetos.
        # EN: carrying the task item → ignore actions on unrelated objects.
        obj_verbs = ("take ", "put ", "heat ", "cool ", "wash ", "plant ")
        if carrying and act.startswith(obj_verbs) and not (at & task_toks):
            s -= 0.5
        for line in memory_lines:
            # PT: avalia cada cláusula — uma frase pode misturar conselho bom e
            # ruim ("stove is broken ...; use the microwave instead").
            # EN: score each clause — one sentence may mix good and bad advice.
            for clause in re.split(r"[;.]", line):
                lt = token_set(clause)
                hit = len(at & lt)
                if hit == 0:
                    continue
                low = clause.lower()
                if any(m in low for m in _BAD_MARKS):
                    s -= 2.0 * hit
                elif any(m in low for m in _GOOD_MARKS):
                    # PT: bônus exige que o OBJETO da ação (token após o verbo)
                    # esteja na cláusula — "take the apple" não promove
                    # "take pillow" nem "heat newspaper with microwave".
                    # EN: bonus requires the action's OBJECT (token after the
                    # verb) to appear in the clause — keeps "take the apple"
                    # from promoting "take pillow"/"heat newspaper".
                    parts = act.split()
                    obj = parts[1] if len(parts) > 1 else ""
                    if hit >= 2 and obj in lt:
                        s += 2.0 * hit
        s -= 0.8 * sum(1 for r in recent if r == act)
        # PT: não desfazer progresso recente (tirar o que acabou de guardar).
        # EN: don't undo recent progress (un-take what you just stored).
        if act.startswith("take ") and any(r.startswith("put ") and act[5:] in r
                                           for r in recent[-6:]):
            s -= 1.2
        s += rng.gauss(0, noise)
        if s > best_s:
            best_s, best_a = s, act
    return best_a


@mock_handler("household.act")
def _household_act(prompt: str, rng: random.Random) -> str:
    """PT: handler mock da política do Household. Só lê o prompt.

    EN: mock Household policy handler. Reads the prompt only.
    """
    section = None
    task = obs = ""
    actions: list[str] = []
    memory: list[str] = []
    recent: list[str] = []
    for line in prompt.splitlines():
        low = line.strip().lower()
        if low.startswith("task:"):
            task, section = line[5:].strip(), None
        elif low.startswith("observation:"):
            obs, section = line[12:].strip(), None
        elif low.startswith("valid actions"):
            section = "act"
        elif low.startswith(("memory", "reflection", "reflections", "lessons")):
            section = "mem"
        elif low.startswith("recent actions"):
            section = "rec"
        elif line.startswith("- "):
            if section == "act":
                actions.append(line[2:].strip())
            elif section == "mem":
                memory.append(line[2:].strip())
            elif section == "rec":
                recent.append(line[2:].strip())
    if not actions:
        return "look"
    return score_actions(task, obs, actions, memory, recent, rng)


def run_episode(
    llm, task: TaskSpec, memory_lines: list[str] | None = None,
    reflections: list[str] | None = None, seed_rng: random.Random | None = None,
) -> dict:
    """PT: roda um episódio com a política "household.act".

    EN: runs one episode with the "household.act" policy.
    """
    from papers.common.llm import task_prompt

    env = HouseholdEnv(task)
    obs = env.reset()
    mem = list(memory_lines or []) + list(reflections or [])
    trajectory: list[tuple[str, str]] = []
    recent: list[str] = []
    while True:
        va = env.valid_actions()
        body = (
            "You are an agent solving a household task. Pick the single best next "
            "action from the valid actions list. Reply with ONLY the action.\n"
            f"Task: {env.task_description}\n"
            f"Observation: {obs}\n"
            + ("Memory:\n" + "\n".join(f"- {m}" for m in mem) + "\n" if mem else "")
            + ("Recent actions:\n" + "\n".join(f"- {r}" for r in recent[-8:]) + "\n"
               if recent else "")
            + "Valid actions:\n" + "\n".join(f"- {a}" for a in va)
        )
        act = llm.complete(task_prompt("household.act", body))
        if act not in va:  # PT: tolera resposta fora da lista. EN: tolerate off-list.
            act = score_actions(task.description, obs, va, mem, recent,
                                seed_rng or random.Random(0))
        res = env.step(act)
        trajectory.append((act, res.observation))
        recent.append(act)
        obs = res.observation
        if res.done:
            break
    return {
        "task_id": task.task_id, "success": env.done, "score": env.score,
        "steps": env.steps, "trajectory": trajectory,
    }
