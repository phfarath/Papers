"""PT: prompts + mock handlers do p15_longmemeval (LongMemEval,
arXiv 2410.10813). Honesty rule: só texto do prompt / prompt text only.

EN: prompts + mock handlers for p15_longmemeval (LongMemEval,
arXiv 2410.10813). Honesty rule: prompt text only.

Tasks:
- ``qa.chain_of_note``: reading strategy do paper (§4, CP4 — "Chain-of-Note"
  reading): dada a pergunta e a evidência, escreve UMA nota por evidência
  (fato extraível + data, ou "irrelevant"), em vez de responder direto.
"""

from __future__ import annotations

import random
import re

from papers.common.facts import extract_facts
from papers.common.llm import mock_handler


def _evidence_block(prompt: str) -> list[str]:
    m = re.search(r"EVIDENCE:\n(.*)", prompt, flags=re.S)
    if not m:
        return []
    return [ln[2:].strip() for ln in m.group(1).splitlines()
            if ln.startswith("- ")]


@mock_handler("qa.chain_of_note")
def _chain_of_note(prompt: str, rng: random.Random) -> str:
    """PT: uma nota por evidência: "Note: {fato (data)}" ou
    "Note: irrelevant". EN: one note per evidence."""
    notes = []
    for ev in _evidence_block(prompt):
        facts = [f for f in extract_facts(ev) if not f.startswith("remove:")]
        if facts:
            notes.extend(f"Note: {f}" for f in facts)
        else:
            d = re.search(r"\d{4}-\d{2}-\d{2}", ev)
            facts_ok = bool(re.search(r"\w+'s \w+ is", ev))
            if facts_ok:
                notes.append(f"Note: {re.sub(r'^- ', '', ev).strip()}")
            elif d and len(ev.split()) > 4:
                notes.append(f"Note: {d.group(0)} {ev.strip()[:80]}")
            else:
                notes.append("Note: irrelevant")
    return "\n".join(notes)
