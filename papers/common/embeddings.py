"""Embeddings determinísticos (hashing) + OpenAI opcional.

PT: HashingEmbedder projeta tokens (minusculos, com stemming leve) e bigramas em
um vetor de dim=512 usando blake2b — NUNCA o hash() do Python, que é salgado por
processo e quebraria a determinismo. Linhas sao L2-normalizadas para cosine = dot.

EN: HashingEmbedder projects tokens (lowercased, lightly stemmed) and bigrams
into a dim=512 vector using blake2b — NEVER Python's hash(), which is salted per
process and would break determinism. Rows are L2-normalized so cosine = dot.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

import numpy as np

DIM = 512

_STOP = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "em", "no", "na", "e",
    "the", "an", "of", "to", "in", "on", "and", "or", "is", "are", "be", "it",
}


def _stem(tok: str) -> str:
    """PT: stemming leve e deterministico (plurais/sufixos comuns EN).

    EN: light deterministic stemming (common EN plural/suffix rules).
    """
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def _feats(text: str) -> list[str]:
    """PT: features = tokens + bigramas adjacentes.

    EN: features = tokens + adjacent bigrams (captures tiny word order).
    """
    norm = "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )
    toks = [_stem(t) for t in re.findall(r"[a-z0-9_]+", norm) if t not in _STOP]
    bigrams = [f"{a}~{b}" for a, b in zip(toks, toks[1:], strict=False)]
    return toks + bigrams


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return (n, d) float32, L2-normalized rows."""
        ...


class HashingEmbedder:
    """PT: embedder deterministico por hashing de features (blake2b).

    EN: deterministic feature-hashing embedder (blake2b).
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _index(self, feat: str) -> int:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in _feats(text):
                out[i, self._index(feat)] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class OpenAIEmbedder:
    """PT: embeddings reais (text-embedding-3-small) — opcional.

    EN: real embeddings (text-embedding-3-small) — optional extra.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder needs the 'openai' package: pip install -e '.[openai]'"
            ) from e
        self._client = openai.OpenAI()
        self.model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_embedder(kind: str | None = None) -> Embedder:
    """PT: fabrica. `kind` ou env PAPERS_EMBEDDER ∈ {hashing, openai}.

    EN: factory. `kind` or env PAPERS_EMBEDDER ∈ {hashing, openai}.
    """
    kind = kind or os.environ.get("PAPERS_EMBEDDER", "hashing")
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"unknown embedder kind {kind!r}")


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PT: matriz (n_a, n_b) de cossenos entre linhas (assumindo normalizadas).

    EN: (n_a, n_b) cosine matrix between rows (assuming normalized).
    """
    return a @ b.T
