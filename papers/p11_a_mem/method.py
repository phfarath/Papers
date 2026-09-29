"""PT: método do p11_a_mem — A-MEM (arXiv 2502.12110).

EN: p11_a_mem method — A-MEM.

Notas agentic (§3.2): cada memória vira uma nota com campos (content,
timestamp, keywords, tags, context, links). Link generation (§3.3): a nota
nova é linkada a vizinhos top-k por embedding. Neighbor evolution (§3.3): os
vizinhos atualizam seus próprios campos (tags/contexto) à luz da nota nova.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM, task_prompt
from papers.common.retrieval import VectorIndex
from papers.p11_a_mem import prompts  # noqa: F401

TOP_K_LINK = 5


@dataclass
class Note:
    """PT: nota A-MEM com os campos do paper. EN: A-MEM note fields."""
    nid: str
    content: str
    ts: datetime | None
    keywords: list[str]
    tags: list[str]
    context: str
    links: list[str] = field(default_factory=list)


class AMemMemory:
    """PT: memória A-MEM: construção de notas + links + evolução.
    EN: A-MEM memory: note construction + links + neighbor evolution."""

    def __init__(self, llm: LLM, embedder: Embedder | None = None) -> None:
        self.llm = llm
        self._index = VectorIndex(embedder or HashingEmbedder())
        self.notes: dict[str, Note] = {}
        self._n = 0
        self.n_links = 0
        self.n_evolutions = 0

    def add(self, text: str, ts: datetime | None = None) -> Note:
        """PT: constrói a nota (campos via LLM), indexa, linka top-k vizinhos,
        evolui vizinhos. EN: construct the note, index, link top-k neighbors,
        evolve neighbors."""
        out = self.llm.complete(task_prompt("amem.note", f"Content: {text}"))
        fields = dict(re.findall(r"(keywords|tags|context): (.*)", out))
        self._n += 1
        note = Note(f"n{self._n}", text, ts,
                    [k.strip() for k in fields.get("keywords", "").split(",")
                     if k.strip()],
                    [t.strip() for t in fields.get("tags", "").split(",")
                     if t.strip()],
                    fields.get("context", ""))
        self.notes[note.nid] = note
        self._index.add(note.nid, text)
        if len(self.notes) == 1:
            return note
        # PT: link generation entre vizinhos top-k. EN: link to top-k neighbors.
        neigh = [(d, s) for d, s in self._index.search(text, TOP_K_LINK + 1)
                 if d != note.nid][:TOP_K_LINK]
        listing = "\n".join(
            f"- {d}: {self.notes[d].content} [tags: "
            f"{', '.join(self.notes[d].tags)}]" for d, _ in neigh)
        chosen = self.llm.complete(task_prompt(
            "amem.link",
            f"New note: {text} {' '.join(note.tags)}\nNeighbors:\n{listing}"))
        for nid in re.findall(r"n\d+", chosen):
            if nid in self.notes and nid != note.nid:
                note.links.append(nid)
                self.notes[nid].links.append(note.nid)
                self.n_links += 1
                # PT: neighbor evolution — o vizinho pode ganhar tag nova.
                # EN: neighbor may gain a new tag.
                ev = self.llm.complete(task_prompt(
                    "amem.evolve",
                    f"New note: {text} {' '.join(note.tags)}\n"
                    f"Neighbor: {self.notes[nid].content} "
                    f"{' '.join(self.notes[nid].tags)}"))
                m = re.search(r"\+(\w+)", ev)
                if m and m.group(1) not in self.notes[nid].tags:
                    self.notes[nid].tags.append(m.group(1))
                    self.n_evolutions += 1
        return note

    def retrieve(self, query: str, k: int = 5) -> list[Note]:
        """PT: retrieval por embedding + 1 hop de links. EN: embedding
        retrieval plus 1-hop link expansion."""
        hits = [d for d, _ in self._index.search(query, k)]
        extra = [ln for d in hits for ln in self.notes[d].links][:2]
        seen: list[str] = []
        for d in hits + extra:
            if d not in seen:
                seen.append(d)
        return [self.notes[d] for d in seen[:k]]
