"""PT: método do p13_mem0 — Mem0 (+Mem0g) (arXiv 2504.19413).

EN: p13_mem0 method — Mem0 (+Mem0g).

Pipeline (§3.1-3.2): extração usa (resumo corrente, últimas 10 mensagens,
par novo user/assistant) → fatos; depois top-10 memórias similares são
recuperadas e o "update phase" decide **ADD / UPDATE / DELETE / NOOP** por
fato. `Mem0gMemory` é a variante em grafo: fatos viram arestas
(entidade, relação, objeto) com invalidação por conflito (a aresta antiga é
marcada `invalid`, não apagada). Métricas por query: latência (wall time) e
tokens-em-contexto.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.facts import SLOT_SYNONYMS
from papers.common.llm import LLM, task_prompt
from papers.common.retrieval import VectorIndex
from papers.common.utils import token_set
from papers.p13_mem0 import prompts  # noqa: F401

LAST_N_MSGS = 10
TOP_K_SIMILAR = 10


@dataclass
class MemStats:
    """PT: métricas do paper — latência e tokens em contexto por query.
    EN: paper metrics — latency and tokens-in-context per query."""
    adds: int = 0
    updates: int = 0
    deletes: int = 0
    noops: int = 0
    latencies: list[float] = field(default_factory=list)
    tokens_in: list[int] = field(default_factory=list)


class Mem0Memory:
    name = "mem0"
    """PT: Mem0 — extração + update phase com ADD/UPDATE/DELETE/NOOP.
    EN: Mem0 — extraction + update phase with ADD/UPDATE/DELETE/NOOP."""

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self.llm = llm
        self._index = VectorIndex(embedder or HashingEmbedder())
        self.memories: dict[str, str] = {}  # id -> fact text
        self._n = 0
        self.summary = ""
        self._recent: list[str] = []
        self.stats = MemStats()

    def _ingest_pair(self, user: str, assistant: str, ts) -> None:
        """PT: extrai fatos do par com (summary, últimas 10 msgs, par novo).
        EN: extract facts from (summary, last-10 msgs, new pair)."""
        last10 = "\n".join(self._recent[-LAST_N_MSGS:])
        prompt = (f"Summary: {self.summary}\nLast messages:\n{last10}\n"
                  f"New pair:\nuser: {user}\nassistant: {assistant}")
        self.stats.tokens_in.append(len(prompt.split()))
        out = self.llm.complete(task_prompt("mem0.extract", prompt))
        facts = [ln.strip() for ln in out.splitlines()
                 if ln.strip() and ln.strip() != "none"]
        if not facts:
            return
        # PT: o índice pode reter ids já removidos por UPDATE/DELETE.
        # EN: the index may keep ids already removed by UPDATE/DELETE.
        sim = [self.memories[d]
               for d, _ in self._index.search(" ".join(facts),
                                              TOP_K_SIMILAR)
               if d in self.memories]
        ops = self.llm.complete(task_prompt(
            "mem0.ops",
            "Facts:\n" + "\n".join(facts) +
            "\nSimilar:\n" + ("\n".join(sim) or "none")))
        self._apply_ops(ops, ts)

    def _apply_ops(self, ops: str, ts) -> None:
        for ln in ops.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("ADD "):
                self._n += 1
                fid = f"m{self._n}"
                self.memories[fid] = ln[4:]
                self._index.add(fid, ln[4:])
                self.stats.adds += 1
            elif ln.startswith("UPDATE "):
                m = re.match(r"UPDATE (.*?) => (.*)", ln)
                if m:
                    self._n += 1
                    fid = f"m{self._n}"
                    old, new = m.group(1), m.group(2)
                    # PT: substitui a memória similar conflitante.
                    # EN: replace the conflicting similar memory.
                    for i, t in list(self.memories.items()):
                        if t == old:
                            del self.memories[i]
                    self.memories[fid] = new
                    self._index.add(fid, new)
                    self.stats.updates += 1
            elif ln.startswith("DELETE "):
                tgt_words = ln[7:].split()
                for i, t in list(self.memories.items()):
                    # PT: match por palavras — "user pet" casa "user's pet".
                    # EN: word-level match — "user pet" hits "user's pet".
                    if all(w in t for w in tgt_words):
                        del self.memories[i]
                        self.stats.deletes += 1
            elif ln.startswith("NOOP"):
                self.stats.noops += 1

    def add(self, text: str, ts: datetime | None = None) -> None:
        """PT: trata o texto como turno de usuário e passa pelo pipeline.
        EN: treat text as a user turn through the pipeline."""
        t0 = time.perf_counter()
        self._ingest_pair(text, "", ts)
        self._recent.append(text)
        self.stats.latencies.append(time.perf_counter() - t0)
        # PT: resumo incremental simples (últimos fatos). EN: light summary.
        vals = list(self.memories.values())
        self.summary = "; ".join(vals[-5:])[:240]

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        return [self.memories[d]
                for d, _ in self._index.search(query, k * 2)
                if d in self.memories][:k]


@dataclass
class Triple:
    """PT: aresta do Mem0g — (entidade, relação, objeto, válida?) com
    intervalo de validade. EN: Mem0g edge — (entity, relation, object,
    valid?) with a validity interval."""
    subj: str
    rel: str
    obj: str
    valid: bool = True
    since: datetime | None = None
    until: datetime | None = None

    @staticmethod
    def _fmt(d: datetime | None) -> str:
        return d.strftime("%Y-%m-%d") if d else "?"

    def text(self) -> str:
        # PT: render legível que o leitor entende (fato + validade).
        # EN: readable render the reader understands (fact + validity).
        end = self._fmt(self.until) if self.until else "present"
        return (f"{self.subj}'s {self.rel} is {self.obj} "
                f"(valid: {self._fmt(self.since)} → {end})")

    def key(self) -> str:
        # PT: texto indexado (sem o intervalo). EN: indexed text (no range).
        return f"{self.subj}'s {self.rel} is {self.obj}"


class Mem0gMemory:
    name = "mem0g"
    """PT: variante em grafo do Mem0 — fatos viram arestas; conflitos
    invalidam a aresta antiga (não apagam). EN: Mem0 graph variant —
    facts become edges; conflicts invalidate the old edge (not delete)."""

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self._inner = Mem0Memory(llm, embedder)
        self._emb = embedder or HashingEmbedder()
        self._index = VectorIndex(self._emb)
        self.edges: list[Triple] = []
        self.invalidated = 0

    @property
    def stats(self) -> MemStats:
        return self._inner.stats

    def add(self, text: str, ts: datetime | None = None) -> None:
        self._inner.add(text, ts)
        # PT: projeta fatos do Mem0 em arestas; conflito de chave → invalida.
        # EN: project Mem0 facts into edges; key conflict → invalidate.
        for fact in list(self._inner.memories.values()):
            m = re.match(r"(.*)'s (\w+) is (.*)", fact)
            if not m:
                continue
            subj, rel, obj = m.group(1), m.group(2).strip(), m.group(3)
            if any(t.subj == subj and t.rel == rel and t.obj == obj and
                   t.valid for t in self.edges):
                continue
            for t in self.edges:
                if t.valid and t.subj == subj and t.rel == rel:
                    t.valid = False
                    t.until = ts
                    self.invalidated += 1
            edge = Triple(subj, rel, obj, True, ts, None)
            self.edges.append(edge)
            self._index.add(f"e{len(self.edges)}", edge.key())

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """PT: recuperação entidade-cêntrica + semântica de triplas (paper
        §3.2 graph): identifica nome+slot na query via sinônimos e devolve
        TODAS as arestas daquele (entidade, slot) — válidas e invalidadas
        com seus intervalos (necessário para perguntas temporais) — mais o
        top-k semântico sobre o texto renderizado das triplas.
        EN: entity-centric + semantic triplet retrieval (paper §3.2):
        resolve name+slot in the query via synonyms and return ALL edges
        of that (entity, slot) — valid and invalidated with intervals
        (needed for temporal questions) — plus semantic top-k over the
        rendered triplet text."""
        qt = token_set(query)
        qslots = {s for s, sy in SLOT_SYNONYMS.items()
                  if sy & qt or s in qt}
        m = re.search(r"\b([A-Z][a-zA-Z]+)\b", query)
        name = m.group(1) if m else None
        wants_past = bool(re.search(
            r"\b(before|previous|previously|used to|originally|earlier|"
            r"did|had|was|were|still|no longer)\b", query.lower()))
        picked: list[Triple] = []
        # PT: arestas entidade-cêntricas do slot perguntado (todas).
        # EN: entity-centric edges of the asked slot (all of them).
        centric = [t for t in self.edges
                   if (name is None or t.subj == name)
                   and (not qslots or t.rel in qslots)
                   and (t.valid or wants_past)]
        picked += centric
        # PT: top-k semântico sobre o texto da tripla (query expandida por
        # sinônimos de slot). EN: semantic top-k over triplet text
        # (slot-synonym expanded query).
        extra = " ".join(sorted({w for sy in SLOT_SYNONYMS.values()
                                 for w in sy if sy & qt}))
        expanded = f"{query} {extra}" if extra else query
        for d, _ in self._index.search(expanded, k * 2):
            t = self.edges[int(d[1:]) - 1]
            if t.valid or wants_past:
                if t not in picked:
                    picked.append(t)
        picked = picked[:max(k, len(centric))]
        return [t.text() for t in picked] or [
            t.text() for t in self.edges if t.valid][:k]
