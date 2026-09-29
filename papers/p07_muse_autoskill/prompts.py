"""PT: prompts + mock handlers do p07_muse_autoskill (arXiv 2605.27366).

EN: prompts + mock handlers for p07_muse_autoskill (arXiv 2605.27366).

NOTA / NOTE: este paper é DIFERENTE do MUSE de p04 (Valiente & Pilly) —
este trata de *autoskill*: criar diretórios de skill com SKILL.md + scripts +
tests + memória, ciclo de vida completo e contexto como DAG comprimido.

Handler tasks (honesty rule — só o texto do prompt / prompt text only):
- ``muse.select``: escolhe a skill do catálogo (disclosure progressivo —
  o catálogo mostra nome+descrição; SKILL.md completo só após seleção).
- ``muse.write_skill``: cria SKILL.md + script + teste a partir da tarefa.
- ``muse.refine``: corrige o script a partir do error trace.
- ``muse.memory``: registra aprendizado no .memory.md da skill.
- ``muse.compress``: resume um nó do DAG (Level-1).
- ``muse.merge_chain``: funde uma cadeia de nós (Level-2).
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.common.utils import token_set


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


@mock_handler("muse.select")
def _select(prompt: str, rng: random.Random) -> str:
    """PT: escolhe a skill cujo nome/descrição mais se sobrepõe à tarefa
    (catálogo só expõe nome+descrição — progressive disclosure, §4.2).
    EN: pick the skill whose name+description best overlaps the task."""
    task = token_set(_section(prompt, "Task"))
    best, best_s = "none", 0
    for line in _section(prompt, "Catalog").splitlines():
        m = re.match(r"- (\w+) — (.*)", line.strip())
        if not m:
            continue
        score = len(task & token_set(m.group(1) + " " + m.group(2)))
        if score > best_s:
            best, best_s = m.group(1), score
    return best


# PT: "conhecimento de mundo" do mock — geradores por tipo de operação.
# EN: the mock's "world knowledge" — generators per operation kind.
_GENERATORS: list[tuple[set[str], str, str, str]] = [
    ({"reverse"},
     "import sys\nprint(sys.argv[1][::-1])",
     "def test(s): return s[::-1]", "abc"),
    ({"vowels", "vowel"},
     "import sys\nprint(sum(c in 'aeiou' for c in sys.argv[1]))",
     "def test(s): return sum(c in 'aeiou' for c in s)", "banana"),
    ({"sum"},
     "import sys\nprint(sum(int(x) for x in sys.argv[1].split(',')))",
     "def test(s): return sum(int(x) for x in s.split(','))", "1,2"),
    ({"uppercase", "upper"},
     "import sys\nprint(sys.argv[1].upper())",
     "def test(s): return s.upper()", "abc"),
    ({"sort"},
     "import sys\nprint(','.join(sorted(sys.argv[1].split(','), key=int)))",
     "def test(s): return ','.join(sorted(s.split(','), key=int))", "2,1"),
    ({"max", "largest", "biggest"},
     "import sys\nprint(max(int(x) for x in sys.argv[1].split(',')))",
     "def test(s): return max(int(x) for x in s.split(','))", "1,9"),
    ({"repeat"},
     "import sys\nw,n=sys.argv[1].split('*')\nprint(w*int(n))",
     "def test(s): w,n=s.split('*'); return w*int(n)", "ab*2"),
    ({"words", "word"},
     "import sys\nprint(len(sys.argv[1].split()))",
     "def test(s): return len(s.split())", "a b c"),
]


def _op_of(text: str) -> tuple[str, str, str] | None:
    toks = token_set(text)
    for keys, script, test, sample in _GENERATORS:
        if toks & keys:
            return script, test, sample
    return None


@mock_handler("muse.write_skill")
def _write_skill(prompt: str, rng: random.Random) -> str:
    """PT: cria a skill a partir da descrição da tarefa — emite SKILL.md
    (frontmatter) + script + teste em blocos marcados. EN: create the skill
    from the task description — emits SKILL.md (frontmatter) + script +
    test in marked blocks."""
    desc = _section(prompt, "Task")
    gen = _op_of(desc)
    name = "_".join(sorted(token_set(desc) & {"reverse", "vowels", "sum",
                                            "uppercase", "sort", "max",
                                            "repeat", "words"})) or "helper"
    script, test, arg = (gen if gen else (
        "import sys\nprint(sys.argv[1])", "def test(s): return s", "abc"))
    # PT: o 'LLM' às vezes escreve código com bug (determinístico via rng) —
    # é isso que o test-gate e o refine existem para capturar.
    # EN: the 'LLM' sometimes writes buggy code (rng-seeded) — this is what
    # the test gate and refine exist to catch.
    if rng.random() < 0.3:
        script = "import sys\nprint(sys.argv[1])  # BUG: echoes input"
    return (f"name: {name}\n"
            f"description: perform {desc.strip()}\n"
            "inputs: one text argument\noutputs: one line\n"
            f"<script>\n{script}\n</script>\n"
            f"<test>\n{test}\n</test>\n"
            f"<arg>{arg}</arg>")


@mock_handler("muse.refine")
def _refine(prompt: str, rng: random.Random) -> str:
    """PT: refina o script a partir do error trace (§4.3): se a skill atual
    falha e a tarefa sugere outro gerador, troca a implementação.
    EN: refine the script from the error trace — if the current skill fails
    and the task suggests another generator, swap the implementation."""
    desc = _section(prompt, "Task")
    gen = _op_of(desc)
    script = gen[0] if gen else "import sys\nprint(sys.argv[1])"
    return f"<script>\n{script}\n</script>"


@mock_handler("muse.memory")
def _memory(prompt: str, rng: random.Random) -> str:
    """PT: aprendizado no .memory.md — nota curta sobre sucesso/falha.
    EN: learning note for .memory.md — short success/failure line."""
    outcome = _section(prompt, "Outcome")
    task = _section(prompt, "Task")
    return f"- [{'ok' if 'success' in outcome else 'fail'}] {task[:60]}"


@mock_handler("muse.compress")
def _compress(prompt: str, rng: random.Random) -> str:
    """PT: Level-1 — resume um nó grande do DAG em ~1 linha.
    EN: Level-1 — summarize a large DAG node into ~1 line."""
    node = _section(prompt, "Node")
    words = node.split()
    keep = " ".join(words[:18])
    return f"[summary] {keep}…"


@mock_handler("muse.merge_chain")
def _merge_chain(prompt: str, rng: random.Random) -> str:
    """PT: Level-2 — funde uma cadeia de nós adjacentes num nó-resumo.
    EN: Level-2 — merge a chain of adjacent nodes into one summary node."""
    chain = _section(prompt, "Chain")
    labels = re.findall(r"\b(step|task|skill|action|obs)\b", chain.lower())
    return f"[merged {len(chain.splitlines()) or 1} nodes: " \
           f"{', '.join(sorted(set(labels))) or 'context'}]"
