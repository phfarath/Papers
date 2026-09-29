"""PT: método do p12_zep — Zep/Graphiti (arXiv 2501.13956).

EN: p12_zep method — Zep/Graphiti.

Grafo temporal bitemporal: cada aresta (fato) tem 4 campos temporais —
t_valid/t_invalid (tempo de mundo) e t_created/t_expired (tempo de
transação). Contradições NÃO apagam a aresta antiga: recebem t_invalid
(§3.2, "edge invalidation"). Busca híbrida: cosine + BM25 + BFS no grafo,
fundidas por RRF; opção MMR para diversidade (§3.3). Contexto = string com
intervalos de validade.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.facts import SLOT_SYNONYMS, extract_facts
from papers.common.retrieval import VectorIndex, mmr
from papers.common.retrieval import reciprocal_rank_fusion as rrf
from papers.common.utils import token_set


def _parse(f: str) -> tuple[str, str, str] | None:
    """PT: "{Nome}'s {slot} is {valor}" → (nome, slot, valor).
    EN: parse a normalized fact."""
    m = re.match(r"(.+)'s (\w+) is (.*)", f)
    return (m.group(1), m.group(2), m.group(3)) if m else None


_REMOVAL = re.compile(r"remove: (\w+) (\w+)")


@dataclass
class Edge:
    """PT: aresta bitemporal. EN: bitemporal edge."""
    eid: str
    entity_a: str
    entity_b: str
    fact: str
    slot: str
    t_valid: datetime | None
    t_invalid: datetime | None = None
    t_created: datetime | None = None
    t_expired: datetime | None = None
    neighbors: list[str] = field(default_factory=list)
    _src: str = ""


class ZepGraph:
    """PT: grafo temporal do Zep. EN: Zep temporal graph."""
    name = "zep-graph"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._index = VectorIndex(embedder or HashingEmbedder())
        self.edges: dict[str, Edge] = {}
        self._n = 0

    def _extract(self, text: str) -> list[tuple[str, str, str, str]]:
        """PT: via extrator compartilhado → (nome, slot, valor, fato) ou
        ('note',...). EN: shared extractor → (name, slot, value, fact)."""
        out = []
        for f in extract_facts(text):
            if f.startswith("remove:"):
                continue
            p = _parse(f)
            if p:
                out.append((p[0], p[1], p[2], f))
        if not out and not _REMOVAL.search(" ".join(extract_facts(text))):
            out.append(("user", "note", "", text))
        return out

    def _invalidate_slot(self, name: str, slot: str, keep: str | None,
                         now: datetime | None) -> int:
        """PT: invalida arestas do mesmo (nome, slot) com outro valor (não
        apaga). EN: invalidate same (name, slot) edges — no delete."""
        n = 0
        for e in self.edges.values():
            if (e.entity_a == name and e.slot == slot and e.eid != keep
                    and e.t_invalid is None):
                e.t_invalid = now
                n += 1
        return n

    def add(self, text: str, ts: datetime | None = None) -> list[Edge]:
        """PT: ingere um turno: cria arestas, linka por slot, invalida
        contradições/retratações. EN: ingest a turn: create edges, link by
        slot, invalidate contradictions/retractions."""
        made: list[Edge] = []
        for f in extract_facts(text):
            m = _REMOVAL.match(f)
            if m:
                self._invalidate_slot(m.group(1), m.group(2), None, ts)
        for name, slot, val, fact in self._extract(text):
            self._n += 1
            e = Edge(f"e{self._n}", name, val, fact, slot, ts, None, ts,
                     None, [], text)
            self.edges[e.eid] = e
            self._index.add(e.eid, f"{fact} {text}")
            for prev in self.edges.values():
                if (prev.eid != e.eid and prev.slot == slot
                        and prev.entity_a == name):
                    prev.neighbors.append(e.eid)
                    e.neighbors.append(prev.eid)
            if slot != "note":
                self._invalidate_slot(name, slot, e.eid, ts)
            made.append(e)
        return made

    # ----- busca híbrida / hybrid search -----
    def _bm25_scores(self, query: str) -> dict[str, float]:
        q = token_set(query)
        # PT: BM25 sobre o documento indexado (fato + texto original).
        # EN: BM25 over the indexed doc (fact + original text).
        return {eid: len(q & token_set(e.fact + " " + e._src))
                for eid, e in self.edges.items()}

    def _bfs(self, seed_ids: list[str], hops: int = 1) -> list[str]:
        seen = set(seed_ids)
        dq = deque(seed_ids)
        out: list[str] = []
        while dq:
            for _ in range(len(dq)):
                cur = dq.popleft()
                if len(out) >= 6:
                    return out
                for nb in self.edges[cur].neighbors:
                    if nb not in seen:
                        seen.add(nb)
                        dq.append(nb)
                        out.append(nb)
        return out

    def search(self, query: str, k: int = 5, *, use_mmr: bool = False,
               now: datetime | None = None) -> list[Edge]:
        """PT: cosine + BM25 + BFS fundidos por RRF; válidos primeiro.
        EN: cosine + BM25 + BFS fused by RRF; currently-valid edges first."""
        # PT: expande a query com os sinônimos de slot — "live" deve
        # alcançar fatos "X's city is Y". EN: expand the query with slot
        # synonyms so "live" reaches "X's city is Y" facts.
        qt = token_set(query)
        extra = " ".join(sorted({w for syns in SLOT_SYNONYMS.values()
                                 for w in syns | {next(iter(syns))}
                                 if syns & qt}))
        expanded = f"{query} {extra}" if extra else query
        cos_hits = self._index.search(expanded, k * 2)
        cos = [d for d, _ in cos_hits]
        bm_scores = self._bm25_scores(expanded)
        bm = sorted(bm_scores, key=bm_scores.get, reverse=True)[:k * 2]
        bfs = self._bfs(cos[:2] + bm[:2])
        # PT: relevância real = cosine>0 ou BM25>0; os demais vão para o fim.
        # EN: real relevance = cosine>0 or BM25>0; the rest rank last.
        rel = {d for d, s in cos_hits if s > 0} | {
            d for d, s in bm_scores.items() if s > 0}
        # PT: recência temporal entra como ranking na fusão (Graphiti ordena
        # por recency). EN: temporal recency as a fused ranking.
        recent = sorted(self.edges, key=lambda d: (
            self.edges[d].t_valid or datetime.min), reverse=True)[:k * 2]
        ranked = rrf([cos, bm, bfs, recent])
        if use_mmr:
            emb = self._index.embedder
            qv = emb.embed([query])
            cand_ids = ranked[:k * 2]
            cv = emb.embed([self.edges[e].fact for e in cand_ids])
            sel = mmr(qv[0], cv, lambda_=0.7, k=k)
            ranked = [cand_ids[i] for i in sel]
        order = {eid: i for i, eid in enumerate(ranked)}
        # PT: o conjunto de saída vem de ranked + arestas do slot da query
        # (precisam estar rankeáveis mesmo sem overlap). EN: the output pool
        # is ranked + same-slot edges (must be selectable even w/o overlap).
        qt0 = token_set(query)
        qslots0 = {s for s, sy in SLOT_SYNONYMS.items()
                   if sy & qt0 or s in qt0}
        pool = set(ranked) | {eid for eid, e in self.edges.items()
                              if e.slot in qslots0 and e.slot != "note"}
        def pri(eid: str) -> tuple[bool, bool, bool, int]:
            e = self.edges[eid]
            # PT: fatos estruturados (slot!="note") acima de conversa bruta;
            # relevante & válido primeiro; inválidos caem mas não somem.
            # EN: structured facts above raw chatter; relevant & valid
            # first; invalid sinks but stays reachable.
            return (eid in rel and e.t_invalid is None,
                    eid in rel and e.slot != "note",
                    e.slot != "note", -order.get(eid, 10 ** 9))
        # PT: ordena TODAS as arestas pela prioridade — fatos relevantes
        # não podem ser expelidos por chatter bem rankeado (rankeados fora
        # do corte recebem o pior rank). EN: pri-sort ALL edges so relevant
        # facts aren't crowded out (out-of-cutoff ids get worst rank).
        top = sorted(pool, key=pri, reverse=True)[:k]
        return [self.edges[e] for e in top]

    def context(self, query: str, k: int = 5,
                now: datetime | None = None) -> str:
        """PT: string de contexto com intervalos de validade.
        EN: context string with validity ranges."""
        def fmt(d: datetime | None) -> str:
            return d.strftime("%Y-%m-%d") if d else "?"
        top = self.search(query, k, now=now)
        # PT: para slots mencionados na query, inclui TODAS as arestas
        # (válidas E invalidadas) — é o que permite responder "before" e
        # "still have a pet?". EN: for slots mentioned in the query, include
        # ALL edges (valid AND invalidated) — that's what answers "before"
        # and "still have a pet?".
        qt = token_set(query)
        qslots = {slot for slot, syns in SLOT_SYNONYMS.items()
                  if syns & qt or slot in qt}
        extra = [e for e in self.edges.values()
                 if e.slot in qslots and e not in top]
        merged = top + sorted(extra, key=lambda e: e.t_valid or datetime.min)
        lines = []
        for e in merged:
            end = fmt(e.t_invalid) if e.t_invalid else "present"
            lines.append(f"- {e.fact} (valid: {fmt(e.t_valid)} → {end})")
        return "\n".join(lines)
