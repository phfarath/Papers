"""PT: adapter MemorySystem do p12_zep. EN: Zep MemorySystem adapter."""

from __future__ import annotations

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.memory_api import MemoryItem
from papers.p12_zep.method import ZepGraph


class ZepAdapter:
    name = "zep"

    def __init__(self, llm: LLM, embedder: Embedder | None = None,
                 use_mmr: bool = False) -> None:
        self._llm = llm
        self._emb = embedder or HashingEmbedder()
        self._mmr = use_mmr
        self.graph = ZepGraph(self._emb)

    def reset(self) -> None:
        self.graph = ZepGraph(self._emb)

    def add(self, item: MemoryItem) -> None:
        self.graph.add(item.text, item.timestamp)

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        return [e.fact for e in self.graph.search(query, k,
                                                use_mmr=self._mmr, now=now)]
