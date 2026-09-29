"""PT: adapters MemorySystem do p13_mem0 — Mem0 e Mem0g.
EN: Mem0 and Mem0g MemorySystem adapters."""

from __future__ import annotations

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.memory_api import MemoryItem
from papers.p13_mem0.method import Mem0gMemory, Mem0Memory


class Mem0Adapter:
    name = "mem0"

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self._llm = llm
        self._emb = embedder or HashingEmbedder()
        self.mem = Mem0Memory(llm, self._emb)

    def reset(self) -> None:
        self.mem = Mem0Memory(self._llm, self._emb)

    def add(self, item: MemoryItem) -> None:
        self.mem.add(item.text, item.timestamp)

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        return self.mem.retrieve(query, k)


class Mem0gAdapter:
    name = "mem0g"

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self._llm = llm
        self._emb = embedder or HashingEmbedder()
        self.mem = Mem0gMemory(llm, self._emb)

    def reset(self) -> None:
        self.mem = Mem0gMemory(self._llm, self._emb)

    def add(self, item: MemoryItem) -> None:
        self.mem.add(item.text, item.timestamp)

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        return self.mem.retrieve(query, k)
