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
from dataclasses import dataclass

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
# LLM. Cada entrada: (código correto, bug plausível). Alguns bugs passam nos
# testes internos derivados dos exemplos e só falham nos testes ESCONDIDOS
# (edge cases) — o paper discute exatamente esse falso positivo.
# EN: the mock's mini knowledge corpus — like an LLM's own knowledge. Each
# entry: (correct code, plausible bug). Some bugs pass the internal tests
# derived from docstring examples and only fail on HIDDEN edge-case tests —
# the paper discusses exactly this false positive.
_CORPUS: dict[str, tuple[str, str]] = {
    "double": (
        "def double(x):\n    return 2 * x",
        "def double(x):\n    return 2 + x",  # erra os exemplos / fails examples
    ),
    "is_even": (
        "def is_even(n):\n    return n % 2 == 0",
        "def is_even(n):\n    return n % 2 == 1",  # fails examples
    ),
    "fib": (
        "def fib(n):\n    a, b = 0, 1\n"
        "    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "def fib(n):\n    a, b = 1, 1\n"
        "    for _ in range(n):\n        a, b = b, a + b\n    return a",
    ),
    "reverse_words": (
        "def reverse_words(s):\n    return ' '.join(s.split()[::-1])",
        "def reverse_words(s):\n    return s[::-1]",  # inverte caracteres
    ),
    "square_list": (
        "def square_list(xs):\n    return [x * x for x in xs]",
        "def square_list(xs):\n    return [x * 2 for x in xs]",  # fails examples
    ),
    # Bugs "tricky": passam nos exemplos da docstring, falham em edge cases.
    "clamp": (
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        # PT: fixa o limite inferior em 0 — passa nos exemplos (lo=0).
        # EN: hardcodes the lower bound at 0 — passes the examples (lo=0).
        "def clamp(x, lo, hi):\n    return min(hi, x) if x > 0 else 0",
    ),
    "sum_digits": (
        "def sum_digits(n):\n    return sum(int(d) for d in str(abs(n)))",
        # PT: só soma 2 dígitos — passa nos exemplos, falha em 3+ dígitos.
        # EN: only sums 2 digits — passes the examples, fails on 3+ digits.
        "def sum_digits(n):\n    return n // 10 + n % 10",
    ),
    "is_palindrome": (
        "def is_palindrome(s):\n    return s == s[::-1]",
        # PT: exige len>2 — passa nos exemplos, falha em 'aa'/''.
        # EN: requires len>2 — passes the examples, fails on 'aa'/''.
        "def is_palindrome(s):\n    return len(s) > 2 and s == s[::-1]",
    ),
}


@dataclass
class CodeProblem:
    """PT: problema de código com exemplos na docstring + testes escondidos.

    EN: coding problem with docstring examples + hidden edge-case tests.
    """

    name: str
    sig: str
    doc: str
    hidden: list[str]


CODE_PROBLEMS: list[CodeProblem] = [
    CodeProblem("double", "double(x)",
                "Return twice the input number. "
                "Examples: double(3) = 6, double(0) = 0.",
                ["double(-2) == -4", "double(0.5) == 1.0", "double(10) == 20"]),
    CodeProblem("is_even", "is_even(n)",
                "Return True iff n is even. "
                "Examples: is_even(4) = True, is_even(3) = False.",
                ["is_even(0) == True", "is_even(-2) == True",
                 "is_even(7) == False"]),
    CodeProblem("fib", "fib(n)",
                "Return the n-th Fibonacci number, fib(0)=0, fib(1)=1. "
                "Examples: fib(0) = 0, fib(1) = 1, fib(6) = 8.",
                ["fib(2) == 1", "fib(10) == 55", "fib(3) == 2"]),
    CodeProblem("reverse_words", "reverse_words(s)",
                "Reverse the order of words in s. "
                "Examples: reverse_words('a b c') = 'c b a', "
                "reverse_words('hi') = 'hi'.",
                ["reverse_words('a  b') == 'b a'",
                 "reverse_words('x y z w') == 'w z y x'"]),
    CodeProblem("square_list", "square_list(xs)",
                "Return the list of squared elements. "
                "Examples: square_list([1, 2]) = [1, 4], "
                "square_list([]) = [].",
                ["square_list([-3]) == [9]", "square_list([0, 5]) == [0, 25]"]),
    CodeProblem("clamp", "clamp(x, lo, hi)",
                "Clamp x into the inclusive range [lo, hi]. "
                "Examples: clamp(5, 0, 10) = 5, clamp(99, 0, 10) = 10, "
                "clamp(-1, 0, 10) = 0.",
                ["clamp(-5, -10, 0) == -5", "clamp(15, 10, 20) == 15",
                 "clamp(25, 10, 20) == 20"]),
    CodeProblem("sum_digits", "sum_digits(n)",
                "Return the sum of the decimal digits of n. "
                "Examples: sum_digits(12) = 3, sum_digits(5) = 5.",
                ["sum_digits(999) == 27", "sum_digits(100) == 1",
                 "sum_digits(12345) == 15"]),
    CodeProblem("is_palindrome", "is_palindrome(s)",
                "Return True iff s reads the same forwards and backwards. "
                "Examples: is_palindrome('aba') = True, "
                "is_palindrome('ab') = False.",
                ["is_palindrome('aa') == True", "is_palindrome('') == True",
                 "is_palindrome('abc') == False"]),
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


@mock_handler("code.tests")
def _code_tests(prompt: str, rng: random.Random) -> str:
    """PT: evaluator INTERNO — deriva asserts SOMENTE dos exemplos da docstring
    presente no prompt ("name(args) = value" → "name(args) == value"). Não conhece
    os testes escondidos — como os testes auto-gerados do paper.

    EN: INTERNAL evaluator — derives asserts ONLY from the docstring examples
    in the prompt ("name(args) = value" → "name(args) == value"). It never sees
    the hidden tests — like the paper's self-generated tests.
    """
    doc = ""
    for line in prompt.splitlines():
        if line.lower().startswith("doc:"):
            doc = line[4:].strip()
    tests: list[str] = []
    for m in re.finditer(r"(\w+)\(([^()]*)\)\s*=\s*"
                         r"(True|False|\[[^\]]*\]|'[^']*'|\"[^\"]*\"|-?[\d.]+)",
                         doc):
        tests.append(f"{m.group(1)}({m.group(2)}) == {m.group(3)}")
    return "\n".join(tests)


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
