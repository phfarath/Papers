"""CLIN — prompts e handlers mock / prompts and mock handlers.

PT: handlers do MockLLM para o CLIN (Majumdar et al., 2023):
- "clin.goal": controller — gera o próximo sub-goal em texto.
- "clin.act": executor — converte o goal numa ação válida.
- "clin.memgen": memory generator — abstrações causais "X [should/may] be
  necessary to Y" / "X does not contribute to Y" a partir da última trial
  (g_t, a_t, o_t), da recompensa final e das 3 memórias mais recentes.
- "clin.metagen": meta-memória (prompt diferente para Gen-Env vs Gen-Task).
- "clin.freeform": ablação — conselhos livres sem a sintaxe causal.

Nota didática (do paper): "causal" aqui ≠ causalidade comprovada — são hipóteses
linguísticas que ajudam o agente.

EN: MockLLM handlers for CLIN (see PT). Didactic note: "causal" here is not
proven causality — they are linguistic hypotheses that help the agent.
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.envs.household import score_actions

# PT: verbo-alvo da tarefa a partir da descrição. EN: task verb from description.
_VERB = {"hot": "heat", "boil": "heat", "cold": "cool", "cool": "cool",
         "clean": "wash", "plant": "plant", "put": "put"}
_WHERE = {"heat": "with the microwave or stove", "cool": "in the fridge",
          "wash": "in the sink", "plant": "in the pot", "put": "in the toolbox"}


def _item_of(desc: str) -> str:
    m = re.search(r"the (\w+)", desc.lower())
    return m.group(1) if m else "item"


def _verb_of(desc: str) -> str:
    d = desc.lower()
    for w, v in _VERB.items():
        if w in d:
            return v
    return "use"


# ---------------------------------------------------------------------------
@mock_handler("clin.goal")
def _goal(prompt: str, rng: random.Random) -> str:
    """PT: controller — próximo goal a partir da tarefa e do que o agente carrega.

    EN: controller — next goal from the task and what the agent carries.
    """
    desc = obs = ""
    for line in prompt.splitlines():
        if line.lower().startswith("task:"):
            desc = line[5:].strip()
        if line.lower().startswith("observation:"):
            obs = line[12:].strip().lower()
    item = _item_of(desc)
    verb = _verb_of(desc)
    if "carrying:" in obs and item in obs:
        return f"{verb} the {item} {_WHERE.get(verb, '')}".strip()
    if desc.lower().startswith("put"):
        return desc  # PT: goal já é o próprio comando. EN: goal is the task.
    return f"find and take the {item}"


@mock_handler("clin.act")
def _act(prompt: str, rng: random.Random) -> str:
    """PT: executor — goal → ação válida, condicionado pela memória.

    EN: executor — goal → valid action, conditioned on memory.
    """
    section = None
    task = goal = obs = ""
    actions: list[str] = []
    memory: list[str] = []
    recent: list[str] = []
    for line in prompt.splitlines():
        low = line.strip().lower()
        if low.startswith("task:"):
            task, section = line[5:].strip(), None
        elif low.startswith("goal:"):
            goal, section = line[5:].strip(), None
        elif low.startswith("observation:"):
            obs, section = line[12:].strip(), None
        elif low.startswith("valid actions"):
            section = "act"
        elif low.startswith("memory"):
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
    # PT: o executor usa o goal + descrição como contexto de overlap.
    # EN: the executor scores actions by goal + task overlap.
    return score_actions(f"{goal} | {task}", obs, actions, memory, recent, rng)


# ---------------------------------------------------------------------------
def _abstractions_from(prompt: str) -> list[str]:
    """PT: gera abstrações causais a partir do texto da trajetória no prompt.

    EN: generates causal abstractions from the trajectory text in the prompt.
    """
    low = prompt.lower()
    out: list[str] = []
    for line in prompt.splitlines():
        if line.lower().startswith("task:"):
            desc = line[5:].strip()
    item = _item_of(desc) if desc else "item"
    verb = _verb_of(desc) if desc else "use"

    for m in re.finditer(r"door to the ([a-z ]+) is closed", low):
        out.append(f"opening the door to the {m.group(1)} may be necessary "
                   "to enter the room")
    for m in re.finditer(r"the (\w+) is closed", low):
        if m.group(1) not in ("door", "kitchen", "garden", "bathroom",
                              "hallway", "living"):
            out.append(f"opening the {m.group(1)} may be necessary to find "
                       "items inside it")
    for m in re.finditer(r"the (\w+) is broken", low):
        dev = m.group(1)
        alt = "microwave" if dev == "stove" else "stove"
        out.append(f"the {dev} does not contribute to heating")
        # PT: só recomenda o dispositivo alternativo quando a tarefa pede calor
        # — antes dizia "microwave necessary to heat the mug" p/ clean_mug e
        # desviava o executor. EN: only recommend the alt device for heat tasks.
        if verb == "heat":
            out.append(f"using the {alt} should be necessary to heat the {item}")
    if "final reward: 1" in low or "reward: 1.0" in low:
        out.append(f"taking the {item} should be necessary to {verb} it")
        out.append(f"{verb}ing the {item} {_WHERE.get(verb, '')} should be "
                   "necessary to finish the task")
    if not out:
        out.append(f"finding and taking the {item} may be necessary to "
                   "start the task")
    return list(dict.fromkeys(out))[:6]


@mock_handler("clin.memgen")
def _memgen(prompt: str, rng: random.Random) -> str:
    """PT: memory generator — formato de abstrações causais do §3.2.

    EN: memory generator — causal-abstraction format from §3.2.
    """
    return "\n".join("- " + a for a in _abstractions_from(prompt))


@mock_handler("clin.metagen")
def _metagen(prompt: str, rng: random.Random) -> str:
    """PT: meta-memória (§3.3): generaliza as melhores memórias para uma nova
    tarefa/ambiente — mantém conselhos sobre containers/portas/dispositivos
    (válidos em qualquer ambiente) e generaliza conselhos de itens.

    EN: meta-memory (§3.3): generalizes the best memories — keeps
    container/door/device advice (valid anywhere) and generalizes item advice.
    """
    out: list[str] = []
    for line in prompt.splitlines():
        low = line.strip().lower()
        if not low.startswith("- "):
            continue
        if any(w in low for w in ("door", "closed", "open", "broken",
                                  "microwave", "stove", "contribute")):
            out.append(line[2:].strip())
    out.append("moving to different rooms may be necessary to find the task items")
    if "gen-task" in low:
        out.append("checking containers first may be necessary to find items")
    return "\n".join("- " + a for a in dict.fromkeys(out))


@mock_handler("clin.freeform")
def _freeform(prompt: str, rng: random.Random) -> str:
    """PT: ablação — conselho livre SEM a sintaxe causal "X necessary to Y".

    EN: ablation — free-form advice WITHOUT the causal syntax.
    """
    low = prompt.lower()
    out: list[str] = []
    for m in re.finditer(r"the (\w+) is broken", low):
        out.append(f"remember the {m.group(1)} was broken last time")
    for m in re.finditer(r"door to the ([a-z ]+) is closed", low):
        out.append(f"last time the {m.group(1)} door was closed")
    if not out:
        out.append("last attempt ran out of steps while exploring")
    return "\n".join("- " + a for a in dict.fromkeys(out))
