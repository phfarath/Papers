"""PT: adapter MemorySystem do p14_memory_r1 — manager Memory-R1.
EN: Memory-R1 MemorySystem adapter."""

from __future__ import annotations

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.memory_api import MemoryItem
from papers.p14_memory_r1.method import MemoryR1Manager, extract_facts


class MemoryR1Adapter:
    name = "memory_r1"

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self._llm = llm
        self._emb = embedder or HashingEmbedder()
        self.manager = MemoryR1Manager(self._emb)

    def reset(self) -> None:
        self.manager = MemoryR1Manager(self._emb)

    def add(self, item: MemoryItem) -> None:
        for f in extract_facts(item.text) or [item.text]:
            self.manager.ingest(f)

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        return self.manager.retrieve(query, k)
