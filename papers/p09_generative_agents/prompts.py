"""PT: prompts + mock handlers do p09_generative_agents (arXiv 2304.03442).

EN: prompts + mock handlers for p09_generative_agents.

Tasks (honesty rule — só texto do prompt / prompt text only):
- ``ga.importance``: nota de importância 1–10 de uma memória (§retrieval).
- ``ga.reflect_q``: 3 perguntas de alto nível sobre as últimas ~100 memórias.
- ``ga.insight``: 5 insights a partir das memórias relevantes, COM citações
  (ids das memórias) como no paper.
- ``ga.plan``: plano diário → agenda por hora → passo de 5–15 min.
- ``ga.dialogue``: decide se uma informação é compartilhada na conversa.
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.common.utils import token_set

_HIGH = {"party", "invited", "invitation", "birthday", "wedding", "fired",
         "moved", "quit", "sick", "dead", "won", "love"}
_MID = {"met", "talked", "bought", "made", "plan", "coffee", "helped"}


@mock_handler("ga.importance")
def _importance(prompt: str, rng: random.Random) -> str:
    """PT: 1–10 conforme "quão poignante" é a memória (paper: mundane≈1,
    life-changing≈10). EN: 1–10 by poignancy."""
    txt = token_set(prompt)
    score = 1
    if txt & _MID:
        score = 4
    if txt & _HIGH:
        score = 9 if txt & {"party", "invited", "invitation"} else 8
    return str(score)


@mock_handler("ga.reflect_q")
def _reflect_q(prompt: str, rng: random.Random) -> str:
    """PT: gera até 3 perguntas de alto nível derivadas das últimas memórias.
    EN: generate up to 3 high-level questions from the recent memories."""
    mems = [ln[2:].strip() for ln in prompt.splitlines()
            if ln.startswith("- ")]
    toks: set[str] = set()
    for m in mems[-100:]:
        toks |= token_set(m)
    qs = []
    if toks & _HIGH:
        qs.append("What social event is coming up?")
    if toks & _MID:
        qs.append("Who has the agent been talking to?")
    if toks:
        qs.append("What is the agent's main concern right now?")
    return "\n".join(f"- {q}" for q in qs[:3]) or "- none"


@mock_handler("ga.insight")
def _insight(prompt: str, rng: random.Random) -> str:
    """PT: 5 insights com citações: 'insight (because of mem1, mem3)'.
    EN: 5 insights with citations like 'insight (because of mem1, mem3)'."""
    lines = [ln for ln in prompt.splitlines() if ln.startswith("[")]
    cits = [re.match(r"\[(mem\d+)\]", ln).group(1) for ln in lines
            if re.match(r"\[(mem\d+)\]", ln)]
    out = []
    joined = " ".join(lines).lower()
    base = [
        ("A party is being organized.", "party"),
        ("Several agents know each other.", "met"),
        ("The agents coordinate plans.", "plan"),
        ("Invitations are spreading.", "invit"),
        ("The town is socially active.", "talked"),
    ]
    for i, (ins, key) in enumerate(base):
        if key in joined and cits:
            out.append(f"- {ins} (because of {cits[i % len(cits)]})")
    while len(out) < 5 and cits:
        out.append(f"- The community shares news quickly. "
                   f"(because of {cits[len(out) % len(cits)]})")
    return "\n".join(out[:5])


@mock_handler("ga.plan")
def _plan(prompt: str, rng: random.Random) -> str:
    """PT: plano diário → agenda por hora → ação de 5–15 min. EN: day plan →
    hourly agenda → 5–15 min step."""
    name = re.search(r"Agent: (\w+)", prompt)
    persona = _sec(prompt, "Persona")
    hour = re.search(r"Hour: (\d+)", prompt)
    h = int(hour.group(1)) if hour else 8
    acts = ["sleep", "have breakfast", "work", "grab coffee", "work",
            "have lunch", "work", "take a walk", "chat at the plaza",
            "have dinner", "read", "sleep"]
    act = acts[min(len(acts) - 1, max(0, (h - 6) // 2))]
    return (f"{name.group(1) if name else 'agent'}: day plan = routine of a "
            f"{persona[:30] or 'villager'}; at {h}:00 → {act} "
            f"(10 min step)")


def _sec(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


@mock_handler("ga.dialogue")
def _dialogue(prompt: str, rng: random.Random) -> str:
    """PT: um turno de diálogo — se quem fala recuperou a 'party invitation'
    nas memórias relevantes, compartilha. EN: one dialogue turn — share the
    party invite if it's in the speaker's relevant memories."""
    mems = _sec(prompt, "Speaker memories")
    if "party" in mems.lower() or "invit" in mems.lower():
        return "shares: 'Hey — are you going to the party?'"
    return "small talk: 'Nice weather today.'"
