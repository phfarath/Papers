"""PT: adapter MemorySystem do p11_a_mem. EN: A-MEM MemorySystem adapter."""

from __future__ import annotations

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.memory_api import MemoryItem
from papers.p11_a_mem.method import AMemMemory


class AMemAdapter:
    name = "a_mem"

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self._llm = llm
        self._emb = embedder or HashingEmbedder()
        self.mem = AMemMemory(llm, self._emb)

    def reset(self) -> None:
        self.mem = AMemMemory(self._llm, self._emb)

    def add(self, item: MemoryItem) -> None:
        self.mem.add(item.text, item.timestamp)

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        return [n.content for n in self.mem.retrieve(query, k)]
