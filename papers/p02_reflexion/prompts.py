"""Reflexion — prompts e handlers mock / prompts and mock handlers.

PT: handlers do MockLLM usados pelo Reflexion:
- "household.act" (reuso de papers.envs.household) é o Actor (ReAct).
- "reflexion.reflect": Self-Reflection — lê a trajetória no prompt e escreve uma
  reflexão verbal apontando o que falhou e o que tentar depois.
- "code.write" / "code.reflect": setting de programação — o mock gera código com
  bugs plausíveis e corrige quando a reflexão aponta o caso que falhou.

REGRA DE HONESTIDADE: handlers só leem o prompt (+rng). O "conhecimento de
programação" do mock vive DENTRO do handler (um mini-corpus), assim como um LLM
real tem conhecimento próprio — ele não lê gabarito nem estado do env.

EN: MockLLM handlers for Reflexion (see PT above). HONESTY RULE: handlers only
read the prompt (+rng). The mock's "programming knowledge" lives INSIDE the
handler (a mini corpus), like a real LLM's own knowledge — it never reads gold
answers or env state.
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler

# ---------------------------------------------------------------------------
# Self-reflection (household)
# ---------------------------------------------------------------------------

_REFLECT_TMPL = """\
You failed a household task. Analyze the trajectory and write ONE short \
reflection (2-4 sentences): what went wrong and exactly what to try next time. \
Be concrete: name actions, objects, rooms, and broken/closed things observed.
Task: {task}
Trajectory (action -> observation excerpt):
{traj}
Final outcome: {outcome}
Previous reflections (do not repeat):
{prev}
Reflection:"""


def reflection_body(task: str, traj: list[tuple[str, str]], outcome: str,
                    prev: list[str]) -> str:
    """PT: monta o corpo do prompt de reflexão. EN: builds the reflection prompt."""
    lines = "\n".join(f"{a} -> {o.split(' You are in')[0]}" for a, o in traj)
    return _REFLECT_TMPL.format(
        task=task, traj=lines, outcome=outcome,
        prev="\n".join(f"- {p}" for p in prev) or "(none)"
    )


@mock_handler("reflexion.reflect")
def _reflect(prompt: str, rng: random.Random) -> str:
    """PT: reflexão verbal honesta — extrai pistas da trajetória no prompt.

    EN: honest verbal reflection — mines hints from the trajectory in the prompt.
    """
    low = prompt.lower()
    out: list[str] = []
    # PT: item-alvo extraído da descrição da tarefa no prompt.
    # EN: target item extracted from the task description in the prompt.
    mt = re.search(r"task: (.+)", low)
    desc = mt.group(1) if mt else ""
    im = re.search(r"the (\w+)", desc)
    item = im.group(1) if im else "item"
    # PT: portas/containers fechados observados → abrir antes é necessário.
    # EN: observed closed doors/containers → opening first is necessary.
    for m in re.finditer(r"door to the ([a-z ]+) is closed", low):
        out.append(f"Opening the door to the {m.group(1)} is necessary before entering;")
    for m in re.finditer(r"the (\w+) is closed", low):
        if m.group(1) not in ("door", "kitchen", "garden", "bathroom",
                              "hallway", "living", "pot"):
            out.append(f"You should open the {m.group(1)} to find items inside it;")
    for m in re.finditer(r"the (\w+) is broken", low):
        dev = m.group(1)
        alt = "microwave" if dev == "stove" else "stove"
        out.append(
            f"The {dev} is broken and does not contribute to heating; "
            f"you should heat the {item} with the {alt} instead.")
    if "you need to hold" in low:
        out.append("You should take the item before trying to use it;")
    if "cannot" in low or "nothing happens" in low:
        out.append("Repeating failed actions does not contribute to the task.")
    if not out:
        # PT: conselho genérico útil: nomear o item e o verbo da tarefa.
        # EN: useful generic advice: name the task item and verb.
        verb = "use"
        for w, v in (("hot", "heat"), ("boil", "heat"), ("cold", "cool"),
                     ("clean", "wash"), ("plant", "plant"), ("put", "put")):
            if w in desc:
                verb = v
        where = {"heat": "with the microwave or stove", "cool": "in the fridge",
                 "wash": "in the sink", "plant": "in the pot",
                 "put": "in the toolbox"}.get(verb, "")
        out.append(
            f"You should take the {item} and {verb} it {where} "
            "to finish the task; avoid useless actions.")
    if "final outcome: success" in low or "outcome: success" in low:
        out = ["The plan worked; repeat the successful sequence of actions "
               "next time."]
    # PT: dedup mantendo ordem. EN: order-preserving dedup.
    return " ".join(list(dict.fromkeys(out))[:3])


# ---------------------------------------------------------------------------
# Setting de programação / programming setting
# ---------------------------------------------------------------------------

# PT: mini-corpus "de conhecimento" do mock — como o conhecimento interno de um
# LLM. Cada entrada: (código correto, bug_plausível).
# EN: the mock's mini knowledge corpus — like an LLM's own knowledge.
_CORPUS: dict[str, tuple[str, str]] = {
    "double": (
        "def double(x):\n    return 2 * x",
        "def double(x):\n    return 2 + x",
    ),
    "is_even": (
        "def is_even(n):\n    return n % 2 == 0",
        "def is_even(n):\n    return n % 2 == 1",
    ),
    "fib": (
        "def fib(n):\n    a, b = 0, 1\n"
        "    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "def fib(n):\n    a, b = 1, 1\n"
        "    for _ in range(n):\n        a, b = b, a + b\n    return a",
    ),
    "reverse_words": (
        "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
        "def reverse_words(s):\n    return s[::-1]",
    ),
    "clamp": (
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        "def clamp(x, lo, hi):\n    return max(lo, x)",
    ),
}

CODE_TASKS: list[dict] = [
    {"name": "double", "sig": "double(x)",
     "doc": "Return twice the input number.",
     "tests": ["double(3) == 6", "double(-2) == -4", "double(0) == 0"]},
    {"name": "is_even", "sig": "is_even(n)",
     "doc": "Return True iff n is even.",
     "tests": ["is_even(4) == True", "is_even(3) == False", "is_even(0) == True"]},
    {"name": "fib", "sig": "fib(n)",
     "doc": "Return the n-th Fibonacci number with fib(0)=0, fib(1)=1.",
     "tests": ["fib(0) == 0", "fib(1) == 1", "fib(6) == 8"]},
    {"name": "reverse_words", "sig": "reverse_words(s)",
     "doc": "Reverse the order of words in the sentence s.",
     "tests": ["reverse_words('a b c') == 'c b a'",
               "reverse_words('hello') == 'hello'"]},
    {"name": "clamp", "sig": "clamp(x, lo, hi)",
     "doc": "Clamp x into the inclusive range [lo, hi].",
     "tests": ["clamp(5, 0, 10) == 5", "clamp(-1, 0, 10) == 0",
               "clamp(99, 0, 10) == 10"]},
]


@mock_handler("code.write")
def _code_write(prompt: str, rng: random.Random) -> str:
    """PT: gera código a partir da docstring no prompt; introduz um bug plausível
    na 1ª tentativa e corrige quando uma reflexão aponta a falha ("should").

    EN: writes code from the prompt's docstring; injects a plausible bug on the
    first attempt and fixes it once a reflection points to the failure.
    """
    name = ""
    for line in prompt.splitlines():
        if line.lower().startswith("function:"):
            name = line.split(":", 1)[1].strip()
    correct, buggy = _CORPUS.get(name, ("pass", "pass"))
    has_reflection = "should" in prompt.lower() or "instead" in prompt.lower()
    if has_reflection:
        return correct
    return buggy if rng.random() < 0.7 else correct


@mock_handler("code.reflect")
def _code_reflect(prompt: str, rng: random.Random) -> str:
    """PT: reflexão para código — cita o primeiro teste que falhou.

    EN: code reflection — cites the first failing test from the prompt.
    """
    m = re.search(r"FAIL: (.+)", prompt)
    if m:
        return (f"The submission failed on `{m.group(1).strip()}`. "
                "You should fix the implementation to handle this case instead "
                "of repeating the same logic.")
    return "All visible tests passed but the evaluator failed; recheck edge cases."
