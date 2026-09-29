"""PT: prompts + mock handlers do p10_memgpt (MemGPT, arXiv 2310.08560).

EN: prompts + mock handlers for p10_memgpt (MemGPT, arXiv 2310.08560).

Tasks (honesty rule — só texto do prompt / prompt text only):
- ``memgpt.step``: decide a próxima chamada de função (das 7 do paper) dada a
  mensagem do usuário e o estado da memória; responde uma linha
  ``call <fn>(<arg>)`` ou ``heartbeat`` para encadear.
- ``memgpt.summarize``: resume recursivamente a metade mais antiga do FIFO.
"""

from __future__ import annotations

import random
import re

from papers.common.facts import extract_facts
from papers.common.llm import mock_handler


def _sec(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z_ ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


@mock_handler("memgpt.step")
def _step(prompt: str, rng: random.Random) -> str:
    """PT: escolhe a função — heurística honesta: fatos novos sobre o usuário
    → archival_memory_insert; pergunta sobre o usuário → archival/recall
    search; senão send_message. EN: pick the function — new user facts →
    archival insert; user question → archival/recall search; else
    send_message."""
    msg = _sec(prompt, "User message")
    low = msg.lower()
    # PT: fatos extraíveis (mesmo com prefixo de nome "Rui here. …") →
    # archival. EN: extractable facts (even name-prefixed) → archival.
    facts = extract_facts(msg)
    is_q = "?" in msg or low.startswith(("what", "where", "when", "do i",
                                        "who", "which", "how"))
    if is_q and not facts:
        return "call archival_memory_search(query=msg)"
    if facts:
        return "call archival_memory_insert(text=msg)\n" \
               "call send_message(text='Got it, noted.')"
    if low.startswith("forget"):
        return "call core_memory_replace(old=pet, new=none)"
    return "call send_message(text='Understood.')"


@mock_handler("memgpt.summarize")
def _summarize(prompt: str, rng: random.Random) -> str:
    """PT: resumo recursivo — comprime a metade mais antiga do FIFO (com o
    resumo anterior) num parágrafo. EN: recursive summary — compress the
    oldest half of the FIFO (with the previous summary) into a paragraph."""
    old = _sec(prompt, "Messages to summarize")
    prev = _sec(prompt, "Previous summary")
    facts = re.findall(r"(?:live in|work as|moved to|quit.*?(?:as|to) "
                       r"|favorite \w+ is|name is|have|picked up|forget) "
                       r"[\w ]+", old.lower())
    head = (prev + " " if prev else "")
    return f"{head}Earlier: user mentioned " + "; ".join(sorted(set(facts))[:6]
                                                          or ["chat"])
