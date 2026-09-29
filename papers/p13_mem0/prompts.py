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

from papers.common.facts import extract_facts
from papers.common.llm import mock_handler


def _sec(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z_ ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


# PT: extração delegada ao helper compartilhado (common.facts) — o "LLM".
# EN: extraction delegated to the shared helper (common.facts) — the "LLM".


@mock_handler("mem0.extract")
def _extract(prompt: str, rng: random.Random) -> str:
    """PT: extrai linhas "fato" do par novo; retratações viram fatos de
    remoção. EN: extract fact lines from the new pair; retractions become
    removal facts."""
    pair = _sec(prompt, "New pair")
    out: list[str] = extract_facts(pair)
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
            tgt = " ".join(f.split()[1:])
            lines.append(f"DELETE {tgt}")
        elif dup is not None:
            lines.append(f"NOOP {f}")
        elif confl is not None:
            lines.append(f"UPDATE {confl} => {f}")
        else:
            lines.append(f"ADD {f}")
    return "\n".join(lines) or "none"
