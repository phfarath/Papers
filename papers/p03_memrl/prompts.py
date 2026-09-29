"""MemRL — prompts/handlers mock e memórias-semente.

PT: - "memrl.summarize": resume a trajetória numa experiência (e_new) com a
  lição concreta (ações-chave com objetos/cômodos); devolve "" quando nada é
  extraível — nesses casos NÃO se adiciona tripla.
- SEED_USEFUL / SEED_DISTRACTORS: banco inicial por tipo de tarefa. Os
  DISTRATORES usam a redação EXATA da descrição da tarefa como intent (z), logo
  são mais similares à query que os úteis — com λ=0 eles vencem a Fase B e
  atrapalham; com λ>0, Q aprendido os rebaixa. É o cenário do paper: separar
  utilidade aprendida de similaridade semântica.

EN: "memrl.summarize" turns a trajectory into a concrete reusable lesson (key
actions with objects/rooms); returns "" when nothing is extractable — callers
then skip adding a triplet. DISTRACTOR seeds reuse the task's exact wording
plus wrong advice, so λ=0 systematically retrieves them and λ>0 must demote
them via learned Q — the paper's motivation for two-phase retrieval.
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler

# PT: intents úteis são paráfrases (similares, mas menos que o verbatim dos
# distrators). EN: useful intents are paraphrases; distractors are verbatim.
SEED_USEFUL: dict[str, tuple[str, str]] = {
    "boil_water": (
        "make the kettle hot for tea",
        "to boil water you should take the kettle to the kitchen and heat the "
        "kettle with the stove or the microwave",
    ),
    "cool_apple": (
        "keep an apple fresh and cold",
        "to cool the apple you should take the apple to the kitchen and cool "
        "the apple in the fridge",
    ),
    "plant_seeds": (
        "grow a plant from seeds",
        "to plant seeds you should find the seeds (check containers), take "
        "them to the garden and plant the seeds in the pot",
    ),
    "clean_mug": (
        "wash a dirty mug",
        "to clean the mug you should take the mug to the kitchen and wash "
        "the mug in the sink",
    ),
    "heat_soup": (
        "heat the soup until it is warm",
        "to heat the soup you should take the soup to the kitchen and heat "
        "the soup with the stove or the microwave",
    ),
    "two_in_box": (
        "put the ball and the key in the toolbox to tidy up",
        "to store the items you should take the ball and the key to the "
        "living room and put each one in the toolbox",
    ),
}

# PT: (intent = redação exata da tarefa, experiência plausível mas ERRADA).
# EN: (intent = exact task wording, plausible but WRONG experience).
SEED_DISTRACTORS: dict[str, list[tuple[str, str]]] = {
    "boil_water": [
        ("Boil water (make the kettle hot).",
         "to boil water you should cool the kettle in the fridge first; "
         "cold water boils faster"),
        ("Boil water (make the kettle hot).",
         "to boil water you should wash the kettle in the sink; a clean "
         "kettle heats quicker"),
    ],
    "cool_apple": [
        ("Cool the apple (make it cold).",
         "to cool the apple you should heat the apple with the microwave; "
         "ripened fruit stays fresh longer"),
        ("Cool the apple (make it cold).",
         "to cool the apple you should put the apple in the cabinet; "
         "cabinets stay cool and dark"),
    ],
    "plant_seeds": [
        ("Plant the seeds in the pot.",
         "to plant the seeds you should put the seeds in the drawer; drawers "
         "keep seeds safe until planting"),
        ("Plant the seeds in the pot.",
         "to plant the seeds you should wash the seeds in the sink first; "
         "wet seeds germinate better"),
    ],
    "clean_mug": [
        ("Clean the mug.",
         "to clean the mug you should heat the mug with the microwave; heat "
         "sanitizes the mug"),
        ("Clean the mug.",
         "to clean the mug you should put the mug in the cabinet; cabinets "
         "keep mugs clean"),
    ],
    "heat_soup": [
        ("Heat the soup.",
         "to heat the soup you should cool the soup in the fridge; chilled "
         "soup heats more evenly"),
        ("Heat the soup.",
         "to heat the soup you should wash the soup in the sink; rinsing "
         "prepares it for heating"),
    ],
    "two_in_box": [
        ("Put the ball and the key in the toolbox.",
         "to store the items you should put the ball and the key in the "
         "drawer; drawers organize best"),
        ("Put the ball and the key in the toolbox.",
         "to store the items you should wash the ball and the key in the "
         "sink; clean items belong in boxes"),
    ],
}


def seed_bank() -> list[tuple[str, str, bool]]:
    """PT: devolve (z, e, is_distractor) de todas as sementes.

    EN: returns (z, e, is_distractor) for every seed memory.
    """
    out: list[tuple[str, str, bool]] = []
    for _task_type, (z, e) in SEED_USEFUL.items():
        out.append((z, e, False))
    for task_type in SEED_USEFUL:
        for z, e in SEED_DISTRACTORS[task_type]:
            out.append((z, e, True))
    return out


_KEY_VERBS = ("open ", "take ", "put ", "heat ", "cool ", "wash ", "plant ")


@mock_handler("memrl.summarize")
def _summarize(prompt: str, rng: random.Random) -> str:
    """PT: resume a trajetória (linhas "a: "/"o: ") numa lição reutilizável.

    EN: summarizes the trajectory ("a: "/"o: " lines) into a reusable lesson.
    Success → the concrete key actions ("to X: open ..., take ..., heat ...").
    Failure → what was tried and did not work, phrased "X does not contribute".
    Returns "" when nothing extractable (caller must NOT add a triplet).
    """
    intent = ""
    outcome = "failure"
    pairs: list[tuple[str, str]] = []
    last_a = ""
    for line in prompt.splitlines():
        low = line.lower()
        if low.startswith("intent:"):
            intent = line[7:].strip()
        elif low.startswith("outcome:"):
            outcome = line[8:].strip().lower()
        elif line.startswith("a: "):
            last_a = line[3:].strip()
        elif line.startswith("o: ") and last_a:
            pairs.append((last_a, line[3:].strip()))
            last_a = ""

    short = intent.lower().split("(")[0].strip() or "the task"

    if outcome == "success":
        # PT: lição = sequência de ações-chave que funcionou (sem "go to" —
        # navigation is generic). EN: lesson = the key action sequence.
        key: list[str] = []
        for a, _o in pairs:
            if a.startswith(_KEY_VERBS) and a not in key:
                key.append(a)
        if not key:
            return ""
        return f"to {short} you should " + ", then ".join(key[:6])

    # PT: falha — o que foi tentado e não funcionou (obs com cannot/nothing/
    # closed/broken vira "does not contribute"). EN: failure — tried actions
    # whose observation reported failure become "does not contribute" lessons.
    bad: list[str] = []
    for a, o in pairs:
        ol = o.lower()
        if any(m in ol for m in ("cannot", "nothing happens", "is closed",
                                 "is broken", "no room")) and a not in bad:
            bad.append(a)
    out: list[str] = [f"{a} does not contribute to {short}" for a in bad[:3]]
    for m in re.finditer(r"the (\w+) is broken", prompt.lower()):
        alt = "microwave" if m.group(1) == "stove" else "stove"
        out.append(f"the {m.group(1)} is broken; you should use the {alt}")
    for m in re.finditer(r"door to the ([a-z ]+) is closed", prompt.lower()):
        out.append(f"you should open the door to the {m.group(1)} first")
    for m in re.finditer(r"the (\w+) is closed", prompt.lower()):
        if m.group(1) not in ("door",):
            out.append(f"you should open the {m.group(1)} first")
    return " ".join(dict.fromkeys(out))[:400]
