"""MemRL — prompts/handlers mock e memórias-semente.

PT: - "memrl.summarize": resume a trajetória numa experiência (e_new) + intent.
- SEED_MEMORIES: banco inicial com experiências úteis e DISTRATORAS
  (semanticamente parecidas mas inúteis) — é o cenário do paper para mostrar que
  Q aprendido separa utilidade de similaridade.

EN: "memrl.summarize" turns a trajectory into an experience summary. SEED
memories include DISTRACTORS (semantically similar but useless) — the paper's
scenario for showing learned Q separates utility from similarity.
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler

# PT: intent (z) + experience (e). Úteis citam a ação certa; distratoras citam
# a ação errada com vocabulário parecido.
# EN: useful seeds cite the right action; distractors the wrong one.
SEED_MEMORIES: list[tuple[str, str]] = [
    ("heat the kettle for boiling water",
     "you should heat the kettle with the microwave; the stove is unreliable"),
    ("boil water quickly",
     "using the microwave should be necessary to heat water-based items"),
    ("cool food items fast",
     "you should cool the item in the fridge; the fridge is in the kitchen"),
    ("clean dirty dishes",
     "you should wash the item in the sink; the sink is in the kitchen"),
    ("plant seeds",
     "you should open the drawer to find seeds; plant the seeds in the pot"),
    ("store two items",
     "you should put each item in the toolbox; the toolbox is in the living room"),
    # distratores — parecidos mas errados / distractors — similar but wrong
    ("heat water container",
     "you should heat the kettle with the stove; it is fast"),
    ("boil water appliance",
     "heating with the stove should be necessary to boil water"),
    ("cool produce item",
     "you should cool the apple in the cabinet; cabinets keep things fresh"),
    ("wash kitchenware item",
     "you should wash the mug with the microwave; soap optional"),
    ("grow the plant",
     "you should plant the seeds in the fridge; cold helps germination"),
    ("organize items",
     "you should put the ball in the drawer; drawers organize best"),
]


@mock_handler("memrl.summarize")
def _summarize(prompt: str, rng: random.Random) -> str:
    """PT: resume trajetória → experiência; intent = linha 'Intent:' do prompt.

    EN: trajectory → experience; intent = the prompt's 'Intent:' line.
    """
    low = prompt.lower()
    out = []
    for m in re.finditer(r"the (\w+) is broken", low):
        alt = "microwave" if m.group(1) == "stove" else "stove"
        out.append(f"the {m.group(1)} is broken; you should use the {alt}")
    for m in re.finditer(r"door to the ([a-z ]+) is closed", low):
        out.append(f"opening the door to the {m.group(1)} is necessary")
    for m in re.finditer(r"the (\w+) is closed", low):
        if m.group(1) not in ("door",):
            out.append(f"you should open the {m.group(1)} first")
    succ = "outcome: success" in low
    if succ:
        out.append("the retrieved strategy worked; repeat it")
    return " ".join(dict.fromkeys(out)) or "episode finished without clear lesson"
