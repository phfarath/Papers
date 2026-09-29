"""PT: adapter MemorySystem do p10_memgpt — ingest via loop de mensagens
(FIFO+archival) e retrieve via archival_memory_search + recall.
EN: MemGPT MemorySystem adapter — ingest through the message loop
(FIFO+archival) and retrieve via archival search + recall."""

from __future__ import annotations

import papers.p10_memgpt.prompts  # noqa: F401 — registra mock handlers
from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.memory_api import MemoryItem
from papers.p10_memgpt.method import MemGPTAgent


class MemGPTAdapter:
    name = "memgpt"

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self._llm = llm
        self._emb = embedder or HashingEmbedder()
        self.agent = MemGPTAgent(llm, self._emb)

    def reset(self) -> None:
        self.agent = MemGPTAgent(self._llm, self._emb)

    def add(self, item: MemoryItem) -> None:
        self.agent.step(item.text, item.timestamp)

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        hits = self.agent.archival.search(query, k)
        arc = [self.agent.arc_texts[d] for d, _ in hits]
        toks = set(query.lower().split())
        rec = [m.text for m in self.agent.fifo
               if toks & set(m.text.lower().split())][:k]
        return arc + rec
