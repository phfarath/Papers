"""Indices de recuperacao: BM25, vetorial, RRF e MMR.

PT: BM25Index implementa Okapi BM25 (k1=1.5, b=0.75). VectorIndex usa um
Embedder e cosine similarity. `reciprocal_rank_fusion` e `mmr` sao usados por
varios papers (Zep, MemoryAgentBench, etc.).

EN: BM25Index implements Okapi BM25 (k1=1.5, b=0.75). VectorIndex uses an
Embedder and cosine similarity. `reciprocal_rank_fusion` and `mmr` are reused
by several papers (Zep, MemoryAgentBench, etc.).
"""

from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

from papers.common.embeddings import Embedder, HashingEmbedder


def _toks(text: str) -> list[str]:
    norm = "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.findall(r"[a-z0-9_]+", norm)


class BM25Index:
    """PT: indice BM25 em memoria. EN: in-memory BM25 index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = _toks(text)
        self.docs[doc_id] = toks
        for t in set(toks):
            self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = sum(len(d) for d in self.docs.values()) / max(len(self.docs), 1)

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, doc_id: str, query: str) -> float:
        toks = self.docs.get(doc_id)
        if toks is None:
            return 0.0
        dl = len(toks)
        s = 0.0
        for qt in set(_toks(query)):
            tf = toks.count(qt)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
            s += self._idf(qt) * (tf * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scored = sorted(
            ((doc_id, self.score(doc_id, query)) for doc_id in self.docs),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:k]


class VectorIndex:
    """PT: indice vetorial com Embedder + cosine. EN: vector index via Embedder."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._mat: np.ndarray | None = None

    def add(self, doc_id: str, text: str) -> None:
        self._ids.append(doc_id)
        self._texts.append(text)
        self._mat = self.embedder.embed(self._texts)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._mat is None or not self._ids:
            return []
        q = self.embedder.embed([query])
        sims = (self._mat @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in order]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """PT: fusao RRF: score(doc) = sum 1/(k + rank). EN: RRF fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def mmr(
    query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_: float, k: int
) -> list[int]:
    """PT: Maximal Marginal Relevance — equilibra relevancia e diversidade.

    EN: Maximal Marginal Relevance — trades off relevance vs diversity.
    Returns indices into cand_vecs.
    """
    n = cand_vecs.shape[0]
    if n == 0 or k <= 0:
        return []
    sim_q = cand_vecs @ query_vec
    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < min(k, n):
        best_i, best_s = -1, -float("inf")
        for i in remaining:
            red = max((float(cand_vecs[i] @ cand_vecs[j]) for j in selected), default=0.0)
            s = lambda_ * float(sim_q[i]) - (1 - lambda_) * red
            if s > best_s:
                best_s, best_i = s, i
        selected.append(best_i)
        remaining.discard(best_i)
    return selected
