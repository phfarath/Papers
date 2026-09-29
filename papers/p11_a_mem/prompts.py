"""PT: prompts + mock handlers do p11_a_mem (A-MEM, arXiv 2502.12110).

EN: prompts + mock handlers for p11_a_mem (A-MEM).

Tasks (honesty rule — só texto do prompt / prompt text only):
- ``amem.note``: constrói os campos da nota (keywords, tags, contexto) a
  partir do conteúdo — a "note construction" do paper (§3.2).
- ``amem.link``: decide quais vizinhos top-k devem ser linkados (§3.3).
- ``amem.evolve``: evolução do vizinho — atualiza tags/contexto do vizinho
  quando a nota nova traz contexto relevante (§3.3, neighbor evolution).
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.common.utils import token_set

_DOMAINS = {
    "food": {"breakfast", "favorite", "coffee", "tea", "bread", "eggs"},
    "location": {"live", "moved", "city", "lisbon", "porto", "berlin",
                 "madrid"},
    "work": {"work", "job", "quit", "barista", "nurse", "pilot", "teacher"},
    "pets": {"cat", "dog", "parrot", "pet", "rex", "nilo", "kiwi"},
    "hobby": {"surfing", "chess", "baking", "birdwatching", "hobby"},
}


def _sec(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z_ ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


@mock_handler("amem.note")
def _note(prompt: str, rng: random.Random) -> str:
    """PT: campos da nota: keywords (tokens-chave), tags (domínios), contexto
    (frase-índice). EN: note fields: keywords, domain tags, context."""
    text = _sec(prompt, "Content")
    toks = token_set(text)
    tags = [d for d, words in _DOMAINS.items() if toks & words] or ["general"]
    kws = sorted(toks - {"the", "a", "i", "my", "is", "to", "and", "of"})[:5]
    ctx = f"user fact about {', '.join(tags)}"
    return (f"keywords: {', '.join(kws)}\n"
            f"tags: {', '.join(tags)}\n"
            f"context: {ctx}")


@mock_handler("amem.link")
def _link(prompt: str, rng: random.Random) -> str:
    """PT: entre os vizinhos top-k listados, emite os ids a linkar (overlap de
    tags/contexto). EN: among the listed top-k neighbors, emit ids to link."""
    new_tags = token_set(_sec(prompt, "New note"))
    out = []
    for ln in _sec(prompt, "Neighbors").splitlines():
        m = re.match(r"- (n\d+): (.*)", ln.strip())
        if m and new_tags & token_set(m.group(2)):
            out.append(m.group(1))
    return ", ".join(out) if out else "none"


@mock_handler("amem.evolve")
def _evolve(prompt: str, rng: random.Random) -> str:
    """PT: evolução do vizinho: se a nota nova adiciona contexto, o vizinho
    ganha uma tag nova (marcada '+tag') ou 'no change'. EN: neighbor
    evolution — neighbor gains a new tag ('+tag') or 'no change'."""
    new = token_set(_sec(prompt, "New note"))
    old = token_set(_sec(prompt, "Neighbor"))
    fresh = sorted((new & set().union(*_DOMAINS.values())) - old)
    return f"+{fresh[0]}" if fresh else "no change"
