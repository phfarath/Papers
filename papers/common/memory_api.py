"""Interface MemorySystem + baselines de comparacao.

PT: MemorySystem e a interface comum para comparar sistemas de memoria nos
benchmarks (p15-p17). Tres baselines sao fornecidos: FullContextMemory (janela
truncada de tokens), BM25Memory e EmbeddingRAGMemory.

EN: MemorySystem is the common interface for comparing memory systems on the
benchmarks (p15-p17). Three baselines are provided: FullContextMemory
(truncated token window), BM25Memory and EmbeddingRAGMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import count_tokens
from papers.common.retrieval import BM25Index, VectorIndex


@dataclass
class MemoryItem:
    """PT: um item de memoria (turno/chunk). EN: one memory item (turn/chunk)."""

    text: str
    timestamp: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class MemorySystem(Protocol):
    """PT: protocolo de ingestao incremental + recuperacao textual.

    EN: incremental-ingestion + text-retrieval protocol.
    """

    name: str

    def reset(self) -> None: ...
    def add(self, item: MemoryItem) -> None: ...
    def retrieve(
        self, query: str, *, k: int = 5, now: datetime | None = None
    ) -> list[str]: ...


class FullContextMemory:
    """PT: devolve os ultimos `max_tokens` tokens do historico (janela truncada).

    EN: returns the last `max_tokens` tokens of history (truncated window).
    """

    name = "full-context"

    def __init__(self, max_tokens: int = 2000) -> None:
        self.max_tokens = max_tokens
        self._items: list[MemoryItem] = []

    def reset(self) -> None:
        self._items = []

    def add(self, item: MemoryItem) -> None:
        self._items.append(item)

    def retrieve(
        self, query: str, *, k: int = 5, now: datetime | None = None
    ) -> list[str]:
        out: list[str] = []
        budget = self.max_tokens
        for item in reversed(self._items):
            t = count_tokens(item.text)
            if t > budget:
                break
            out.append(item.text)
            budget -= t
        return list(reversed(out))


class BM25Memory:
    """PT: recuperacao top-k por BM25. EN: top-k retrieval via BM25."""

    name = "bm25-rag"

    def __init__(self) -> None:
        self._index = BM25Index()
        self._texts: dict[str, str] = {}
        self._n = 0

    def reset(self) -> None:
        self._index = BM25Index()
        self._texts = {}
        self._n = 0

    def add(self, item: MemoryItem) -> None:
        doc_id = f"doc{self._n}"
        self._n += 1
        self._index.add(doc_id, item.text)
        self._texts[doc_id] = item.text

    def retrieve(
        self, query: str, *, k: int = 5, now: datetime | None = None
    ) -> list[str]:
        return [self._texts[d] for d, _ in self._index.search(query, k)]


class EmbeddingRAGMemory:
    """PT: recuperacao top-k por similaridade de embedding. EN: embedding RAG."""

    name = "embedding-rag"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or HashingEmbedder()
        self._index = VectorIndex(self._embedder)
        self._texts: dict[str, str] = {}
        self._n = 0

    def reset(self) -> None:
        self._index = VectorIndex(self._embedder)
        self._texts = {}
        self._n = 0

    def add(self, item: MemoryItem) -> None:
        doc_id = f"doc{self._n}"
        self._n += 1
        self._index.add(doc_id, item.text)
        self._texts[doc_id] = item.text

    def retrieve(
        self, query: str, *, k: int = 5, now: datetime | None = None
    ) -> list[str]:
        return [self._texts[d] for d, _ in self._index.search(query, k)]
