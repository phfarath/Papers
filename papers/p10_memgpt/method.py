"""PT: método do p10_memgpt — MemGPT (arXiv 2310.08560).

EN: p10_memgpt method — MemGPT.

Memória em camadas estilo OS (§3): core memory (blocos persona/human com
limite de chars, sempre no contexto), recall memory (FIFO da conversa),
archival memory (store vetorial paginável). O agente opera por CHAMADAS DE
FUNÇÃO — as 7 do paper: send_message, core_memory_append,
core_memory_replace, archival_memory_insert, archival_memory_search,
recall_memory_search, heartbeat. Contexto com WARNING a 70% e FLUSH a 100%:
descarta ~50% das mensagens FIFO mais antigas resumindo-as recursivamente.

Layered OS-style memory (§3): core memory (persona/human blocks with char
limits, always in context), recall memory (conversation FIFO), archival
memory (paged vector store). The agent acts via FUNCTION CALLS — the paper's
7: send_message, core_memory_append, core_memory_replace,
archival_memory_insert, archival_memory_search, recall_memory_search,
heartbeat. Context WARNING at 70% and FLUSH at 100%: evict ~50% of the
oldest FIFO messages by recursively summarizing them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM, task_prompt
from papers.common.retrieval import VectorIndex
from papers.p10_memgpt import prompts  # noqa: F401

WARN_FRAC = 0.70   # PT: aviso de pressão de contexto. EN: context warning.
FLUSH_FRAC = 1.00  # PT: flush. EN: flush.
EVICT_FRAC = 0.50  # PT: ~50% do FIFO resumido. EN: ~50% FIFO summarized.
MAX_HEARTBEATS = 3

FUNCTIONS = [
    "send_message", "core_memory_append", "core_memory_replace",
    "archival_memory_insert", "archival_memory_search",
    "recall_memory_search", "heartbeat",
]


@dataclass
class CoreBlock:
    name: str
    limit: int
    text: str = ""


@dataclass
class Msg:
    role: str
    text: str
    ts: datetime | None = None


class MemGPTAgent:
    """PT: agente MemGPT com loop de chamadas de função + heartbeat chaining.
    EN: MemGPT agent with a function-call loop + heartbeat chaining."""

    def __init__(self, llm: LLM, embedder: Embedder | None = None,
                 ctx_tokens: int = 120) -> None:
        self.llm = llm
        self.ctx_tokens = ctx_tokens
        self.core = {
            "persona": CoreBlock("persona", 200, "helpful assistant"),
            "human": CoreBlock("human", 200, ""),
        }
        self.summary = ""
        self.fifo: list[Msg] = []
        self.archival = VectorIndex(embedder or HashingEmbedder())
        self.arc_texts: dict[str, str] = {}
        self._arc_n = 0
        self.flushes = 0
        self.warnings = 0

    # ---------- pressão de contexto (§3.4) ----------
    def _ctx_used(self) -> float:
        toks = sum(len(m.text.split()) for m in self.fifo)
        toks += sum(len(b.text.split()) for b in self.core.values())
        toks += len(self.summary.split())
        return toks / self.ctx_tokens

    def _maybe_flush(self) -> None:
        if self._ctx_used() >= WARN_FRAC:
            self.warnings += 1
        if self._ctx_used() < FLUSH_FRAC:
            return
        self.flushes += 1
        n_evict = max(1, int(len(self.fifo) * EVICT_FRAC))
        old = self.fifo[:n_evict]
        self.fifo = self.fifo[n_evict:]
        out = self.llm.complete(task_prompt(
            "memgpt.summarize",
            "Previous summary: " + (self.summary or "none") + "\n"
            "Messages to summarize:\n" + "\n".join(m.text for m in old)))
        self.summary = out.strip()

    # ---------- as 7 funções (§3.2/Tab.1) ----------
    def call(self, fn: str, arg: str = "") -> str:
        """PT: dispatcher das 7 function calls. EN: 7-function dispatcher."""
        if fn == "send_message":
            return f"sent: {arg}"
        if fn == "core_memory_append":
            self.core["human"].text += " " + arg
            return "core appended"
        if fn == "core_memory_replace":
            self.core["human"].text = arg
            return "core replaced"
        if fn == "archival_memory_insert":
            self._arc_n += 1
            did = f"arc{self._arc_n}"
            self.archival.add(did, arg)
            self.arc_texts[did] = arg
            return "archived"
        if fn == "archival_memory_search":
            hits = self.archival.search(arg, 5)
            return "; ".join(self.arc_texts[d] for d, _ in hits) or "none"
        if fn == "recall_memory_search":
            toks = set(arg.lower().split())
            hits = [m.text for m in self.fifo
                    if toks & set(m.text.lower().split())]
            return "; ".join(hits[-5:]) or "none"
        if fn == "heartbeat":
            return "heartbeat"
        raise ValueError(f"unknown function {fn}")

    def step(self, user_text: str, ts: datetime | None = None) -> list[str]:
        """PT: um turno: ingest → decide chamadas → heartbeat chaining.
        EN: one turn: ingest → decide calls → heartbeat chaining."""
        self.fifo.append(Msg("user", user_text, ts))
        self._maybe_flush()
        calls: list[str] = []
        state = (f"Core human: {self.core['human'].text or 'empty'}\n"
                 f"Summary: {self.summary or 'none'}\n"
                 f"FIFO messages: {len(self.fifo)}")
        for _ in range(MAX_HEARTBEATS):
            out = self.llm.complete(task_prompt(
                "memgpt.step",
                f"User message: {user_text}\nMemory state:\n{state}"))
            lines = [ln for ln in out.splitlines() if ln.startswith("call ")]
            if not lines:
                break
            for ln in lines:
                m = re.match(r"call (\w+)\((.*?)\)", ln)
                if not m:
                    continue
                fn, arg = m.group(1), m.group(2)
                if fn not in FUNCTIONS:
                    continue
                # PT: resolve placeholders do mock: text=msg/query=msg → a
                # mensagem real. EN: resolve mock placeholders to the real
                # user message.
                arg = re.sub(r"^(text|query|old|new)=", "", arg).strip("'")
                arg = {"msg": user_text, "pet": "", "none": ""}.get(arg, arg)
                calls.append(self.call(fn, arg))
            # PT: heartbeat chaining — o agente pode pedir outro turno.
            # EN: heartbeat chaining — agent may request another turn.
            if "heartbeat" not in out:
                break
        self.fifo.append(Msg("agent", "; ".join(calls) or "ok", ts))
        return calls
