"""PT: prompts + mock handlers do p08_coala (CoALA, arXiv 2309.02427).

EN: prompts + mock handlers for p08_coala (CoALA, arXiv 2309.02427).

Handler tasks (honesty rule — só o texto do prompt / prompt text only):
- ``coala.retrieve``: internal action — lê a LTM e devolve as linhas
  relevantes para a tarefa corrente (retrieval, §3.2).
- ``coala.learn``: internal action — escreve na LTM uma regra extraída da
  trajetória (learning/write, §3.2).
- ``coala.reason``: internal action — atualiza a working memory com o plano
  corrente (reasoning, §3.2).
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.common.utils import token_set


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


@mock_handler("coala.retrieve")
def _retrieve(prompt: str, rng: random.Random) -> str:
    """PT: retrieval — devolve as linhas da LTM com maior overlap de tokens
    com a tarefa (como um retrieval real faria). EN: return LTM lines with
    highest token overlap with the task."""
    tt = token_set(_section(prompt, "Task"))
    scored: list[tuple[int, str]] = []
    for line in _section(prompt, "Long-term memory").splitlines():
        line = line.strip("- ").strip()
        if not line:
            continue
        s = len(tt & token_set(line))
        if s:
            scored.append((s, line))
    scored.sort(key=lambda x: -x[0])
    return "\n".join(f"- {ln}" for _, ln in scored[:3]) or "- none"


@mock_handler("coala.learn")
def _learn(prompt: str, rng: random.Random) -> str:
    """PT: learning — da trajetória bem-sucedida extrai a regra-chave
    (objeto da tarefa + appliance usado), como uma escrita na memória
    semântica. EN: extract the key rule (task object + appliance used) from a
    successful trajectory, like a semantic-memory write."""
    task = _section(prompt, "Task")
    traj = _section(prompt, "Trajectory")
    obj = ""
    m = re.search(r"the (\w+)", task.lower())
    if m:
        obj = m.group(1)
    used = re.findall(
        r"(heat|cool|wash|plant) (\w+) (?:with|in) (\w+)", traj.lower())
    if obj and used:
        v, _o, dev = used[-1]
        return f"you should {v} the {obj} with the {dev}"
    if obj:
        return f"you should take the {obj}"
    return ""


@mock_handler("coala.reason")
def _reason(prompt: str, rng: random.Random) -> str:
    """PT: reasoning — resume o progresso atual da working memory num plano
    de uma linha. EN: summarize current WM progress into a one-line plan."""
    obs = _section(prompt, "Observation")
    tt = _section(prompt, "Task")
    if "carrying:" in obs:
        return "plan: item secured — perform the task verb on it."
    return f"plan: explore and locate the object needed for: {tt[:60]}"
