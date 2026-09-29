"""Utilidades gerais / General utilities.

PT: Funcoes pequenas usadas em todo o repo: normalizacao de texto, tokens,
parse de bloco JSON tolerante a cercas ```json.
EN: Small helpers used across the repo: text normalization, tokenization,
JSON-block parsing tolerant of ```json fences.
"""

from __future__ import annotations

import json
import re
import unicodedata

_STOPWORDS = {
    # PT + EN basic stopwords / palavras vazias basicas
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "em", "no", "na", "e", "ou",
    "the", "an", "of", "to", "in", "on", "and", "or", "is", "are", "be", "with",
    "for", "it", "its", "that", "this", "you", "your", "i", "we", "they", "at",
}


def strip_accents(text: str) -> str:
    """PT: remove acentos (util para matching tolerante). EN: strip accents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def tokens(text: str) -> list[str]:
    """PT: tokens minusculos, sem acento, sem stopwords.

    EN: lowercase, accent-free, stopword-free tokens.
    """
    raw = re.findall(r"[a-z0-9_]+", strip_accents(text.lower()))
    return [t for t in raw if t not in _STOPWORDS]


def token_set(text: str) -> set[str]:
    """PT: conjunto de tokens. EN: set of tokens."""
    return set(tokens(text))


def overlap(a: str, b: str) -> float:
    """PT: coeficiente de Jaccard entre os tokens de dois textos (0..1).

    EN: Jaccard similarity between the token sets of two texts (0..1).
    """
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_json_block(text: str) -> object:
    """PT: extrai o primeiro bloco JSON do texto, tolerando cercas ```json.

    EN: extract the first JSON block from text, tolerating ```json fences.
    Raises ValueError when nothing parses.
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    # PT: tenta tambem o primeiro {...} ou [...] balanceado por regex simples.
    # EN: also try the first {...} or [...] span as a fallback.
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    candidates.append(text)
    for cand in candidates:
        try:
            return json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"no JSON block found in: {text[:200]!r}")


def zscore(values: list[float]) -> list[float]:
    """PT: normalizacao z-score; desvio 0 -> zeros. EN: z-score; std 0 -> zeros."""
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var**0.5
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]
