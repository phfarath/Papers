"""MUSE (LLM impl) — prompts e handlers mock / prompts and mock handlers.

PT: handlers do MockLLM:
- "muse.plan": plano inicial P^e a partir da descrição da tarefa (I).
- "muse.rollout": rollout hipotético (temperatura 0.5 no paper) — gera uma
  trajetória imaginária de 6 passos a partir do estado atual. O handler é
  honesto: só vê task/obs/ações válidas/memória no prompt, e imagina desfechos
  plausíveis — pode imaginar fracasso também ("does not contribute").

EN: MockLLM handlers: "muse.plan" (initial plan P^e), "muse.rollout"
(hypothetical trajectory, temperature 0.5 in the paper) — imagines a 6-step
trajectory from the current state, honestly from prompt text only.
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.envs.household import score_actions


@mock_handler("muse.plan")
def _plan(prompt: str, rng: random.Random) -> str:
    """PT: plano P^e = 'achar item → pegar → verbo'. EN: plan P^e template."""
    desc = ""
    for line in prompt.splitlines():
        if line.lower().startswith("task:"):
            desc = line[5:].strip()
    item = re.search(r"the (\w+)", desc.lower())
    item = item.group(1) if item else "item"
    verb = "use"
    for w, v in (("hot", "heat"), ("boil", "heat"), ("cold", "cool"),
                 ("clean", "wash"), ("plant", "plant"), ("put", "put")):
        if w in desc.lower():
            verb = v
    return (f"1. find the {item} by moving between rooms; "
            f"2. take the {item}; 3. {verb} the {item} to finish.")


@mock_handler("muse.rollout")
def _rollout(prompt: str, rng: random.Random) -> str:
    """PT: gera uma trajetória imaginária de 6 passos (a: / o: por linha).

    EN: generates a 6-step imagined trajectory (a: / o: lines). Noise via rng
    emulates temperature 0.5 diversity; imagined failures are possible.
    """
    section = None
    task = obs = ""
    actions: list[str] = []
    memory: list[str] = []
    for line in prompt.splitlines():
        low = line.strip().lower()
        if low.startswith("task:"):
            task = line[5:].strip()
        elif low.startswith("observation:"):
            obs = line[12:].strip()
        elif low.startswith("valid actions"):
            section = "act"
        elif low.startswith("reflections") or low.startswith("memory"):
            section = "mem"
        elif line.startswith("- "):
            (actions if section == "act" else memory).append(line[2:].strip())
    if not actions:
        return "a: look\no: nothing"
    lines = []
    # PT: imagina 6 passos; ~15% das vezes "imagina" um desfecho ruim
    # (a honestidade exige diversidade real de rollouts).
    # EN: imagine 6 steps; ~15% of rollouts imagine a poor ending — real
    # rollout diversity is required for M_sa to discriminate.
    bad = rng.random() < 0.15
    for _ in range(6):
        if bad:
            act = rng.choice(actions)
        else:
            act = score_actions(task, obs, actions, memory, [], rng, noise=0.6)
        outcome = "ok" if rng.random() < 0.8 else "nothing happens"
        lines.append(f"a: {act}\no: you {act}. {outcome}")
    lines.append("outcome: " + ("failure" if bad else "likely success"))
    return "\n".join(lines)


def first_action(rollout_text: str) -> str | None:
    """PT: extrai a 1ª ação do rollout. EN: extract the rollout's first action."""
    m = re.search(r"^a:\s*(.+)$", rollout_text, re.MULTILINE)
    return m.group(1).strip() if m else None
