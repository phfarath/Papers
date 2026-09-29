"""PT: prompts + mock handlers do p13_mem0 (arXiv 2504.19413).

EN: prompts + mock handlers for p13_mem0.

Tasks (honesty rule — só texto do prompt / prompt text only):
- ``mem0.extract``: extrai fatos do par (user,assistant) usando o resumo +
  últimas 10 msgs fornecidas no prompt (§3.1).
- ``mem0.ops``: dado os fatos extraídos + top-10 memórias similares, decide
  ADD/UPDATE/DELETE/NOOP por fato (§3.2).
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler


def _sec(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z_ ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


# PT: (slot, padrão) → fato "user's <slot> is <valor>".
# EN: (slot, pattern) → fact "user's <slot> is <value>".
_PATS = [
    ("city", r"I live in ([A-Z][a-zA-Z]+)"),
    ("city", r"I moved to ([A-Z][a-zA-Z]+)"),
    ("job", r"I (?:work as|quit my job as|started (?:a new job )?as) "
            r"(\w[\w ]*?)(?: at| in| and|\.)",),
    ("pet", r"(?:my|a) (?:cat|dog|parrot)(?: is | named )?(\w+)"),
    ("breakfast", r"my favorite breakfast is ([\w ]+)"),
    ("hobby", r"(?:taken up|hobby is|enjoy) ([\w ]+?)(?:\.| and|$)"),
]


@mock_handler("mem0.extract")
def _extract(prompt: str, rng: random.Random) -> str:
    """PT: extrai linhas "fato" do par novo; retratações viram fatos de
    remoção. EN: extract fact lines from the new pair; retractions become
    removal facts."""
    pair = _sec(prompt, "New pair")
    out: list[str] = []
    for slot, pat in _PATS:
        for m in re.finditer(pat, pair):
            out.append(f"user's {slot} is {m.group(1).strip()}")
    if re.search(r"gave .* away|no longer|don't have", pair, flags=re.I):
        out.append("remove: pet")
    if not out:
        short = pair.strip().splitlines()[0][:80] if pair.strip() else ""
        if short:
            out.append(short)
    return "\n".join(dict.fromkeys(out)) or "none"


@mock_handler("mem0.ops")
def _ops(prompt: str, rng: random.Random) -> str:
    """PT: por fato, ADD se não similar, UPDATE se contradiz, DELETE se
    retratação, NOOP se duplicata. EN: per fact ADD/UPDATE/DELETE/NOOP."""
    facts = [ln.strip() for ln in _sec(prompt, "Facts").splitlines()
             if ln.strip()]
    sims = [ln.strip() for ln in _sec(prompt, "Similar").splitlines()
            if ln.strip()]
    lines = []
    for f in facts:
        key = f.rsplit(" is ", 1)[0].strip()
        dup = next((s for s in sims if s.lower() == f.lower()), None)
        confl = next((s for s in sims
                      if s.rsplit(" is ", 1)[0].strip() == key
                      and s.lower() != f.lower()), None)
        if f.startswith("remove:"):
            tgt = f.split()[1]
            lines.append(f"DELETE {tgt}")
        elif dup is not None:
            lines.append(f"NOOP {f}")
        elif confl is not None:
            lines.append(f"UPDATE {confl} => {f}")
        else:
            lines.append(f"ADD {f}")
    return "\n".join(lines) or "none"
