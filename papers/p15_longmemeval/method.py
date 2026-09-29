"""PT: p15 LongMemEval (arXiv 2410.10813) — benchmark de memória de longo
prazo + o "unified view" de design de memória do paper (§4):

- CP1 **Value**: o que se guarda — granularidade por SESSÃO vs por ROUND
  (turno) vs fatos decompostos.
- CP2 **Key**: como se indexa — texto cru vs *key expansion* (fatos
  extraídos anexados ao índice).
- CP3 **Query**: expansão de query *time-aware* para razão temporal (TR).
- CP4 **Reading**: leitura direta vs **Chain-of-Note** (uma nota por
  evidência antes de responder).

As 5 habilidades do benchmark mapeiam para os tipos do dataset compartilhado:
IE (information extraction)=single-hop, MR (multi-session reasoning)=
multi-hop, TR (temporal reasoning)=temporal, KU (knowledge update)=update,
ABS (abstention)=abstention.

EN: p15 LongMemEval — long-term memory benchmark + the paper's unified
memory-design view (§4). Value granularity (session/round/facts), key
expansion, time-aware query expansion for TR, Chain-of-Note reading
(CP4). Ability map: IE=single-hop, MR=multi-hop, TR=temporal, KU=update,
ABS=abstention.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta

import papers.p15_longmemeval.prompts  # noqa: F401 — registra handlers
from papers.common.conv_data import ConvSet, Question, Turn
from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.facts import extract_facts
from papers.common.llm import LLM, task_prompt
from papers.common.memory_api import MemoryItem
from papers.common.reader import answer_with_evidence
from papers.common.retrieval import VectorIndex

# PT: mapeamento habilidade do paper → qtype interno. EN: paper ability →
# internal qtype.
ABILITIES = {"IE": "single-hop", "MR": "multi-hop", "TR": "temporal",
             "KU": "update", "ABS": "abstention"}
ABILITY_ORDER = ["IE", "MR", "TR", "KU", "ABS"]


def scale_history(data: ConvSet, extra_filler: int, seed: int) -> ConvSet:
    """PT: cria uma história mais longa: por persona, insere
    `extra_filler` sessões de enchimento entre as sessões reais (timestamps
    monotônicos). EN: longer history via extra filler sessions per persona,
    monotonic timestamps."""
    rng = random.Random(seed)
    filler_lines = [
        "How is the weather today?", "Tell me a fun fact.",
        "What time is it?", "Thanks, that helps.", "Nice.", "I see.",
        "Interesting, go on.", "Ok great.", "Hmm, alright.",
        "What do you think about the news?", "Just checking in.",
    ]
    new_sessions: list[list[Turn]] = []
    by_persona: dict[str, list[Turn]] = {}
    for name, turns in data.by_persona.items():
        t0 = turns[0].ts
        # PT: espaça os turnos reais e intercala enchimento no tempo.
        # EN: spread real turns and interleave filler in time.
        new_turns: list[Turn] = []
        for i, t in enumerate(turns):
            new_turns.append(Turn(t.speaker, t.text,
                                  t0 + timedelta(hours=i * (extra_filler + 2))))
            for j in range(extra_filler):
                new_turns.append(Turn(
                    "user", f"{rng.choice(filler_lines)}",
                    t0 + timedelta(hours=i * (extra_filler + 2) + j + 1)))
        by_persona[name] = new_turns
        chunk = max(1, len(new_turns) // (4 + extra_filler))
        new_sessions += [new_turns[i:i + chunk]
                         for i in range(0, len(new_turns), chunk)]
    return ConvSet(new_sessions, list(data.questions), by_persona)


def load_official(path: str) -> ConvSet:
    """PT: loader opcional do JSON oficial LongMemEval (schema: lista de
    itens com ``question``, ``answer``, ``haystack_sessions`` — lista de
    sessões, cada uma lista de turnos {"role","content"} — e
    ``haystack_dates``). EN: optional official-JSON loader."""
    items = json.load(open(path, encoding="utf-8"))
    sessions: list[list[Turn]] = []
    questions: list[Question] = []
    for it in items:
        for i, sess in enumerate(it.get("haystack_sessions", [])):
            date = it.get("haystack_dates", ["2024-01-01"] * 100)[i]
            ts = datetime.fromisoformat(date[:10])
            sessions.append([Turn(t.get("role", "user"),
                                  t.get("content", ""), ts) for t in sess])
        qt = it.get("question_type", "single-hop")
        qtype = {"single-session-user": "single-hop",
                 "multi-session": "multi-hop",
                 "temporal-reasoning": "temporal",
                 "knowledge-update": "update",
                 "single-session-preference": "single-hop",
                 "abstention": "abstention"}.get(qt, "single-hop")
        gold = it.get("answer", "")
        if isinstance(gold, list):
            gold = str(gold[0]) if gold else ""
        questions.append(Question(qtype, it["question"], str(gold)))
    return ConvSet(sessions, questions, {})


class DesignRAG:
    """PT: baseline de RAG com os 4 knobs de design do paper (§4.2).
    EN: RAG baseline carrying the paper's 4 design knobs (§4.2).

    - ``value``: "round" (turno) | "session" | "facts" (decomposto).
    - ``key_expansion``: anexa fatos extraídos ao texto indexado.
    - ``time_aware``: expande queries temporais com sinônimos de slot.
    - ``chain_of_note``: lê por notas antes de responder.
    """

    name = "rag"

    def __init__(self, embedder: Embedder | None = None, *,
                 value: str = "round", key_expansion: bool = False,
                 time_aware: bool = False, chain_of_note: bool = False) -> None:
        self._emb = embedder or HashingEmbedder()
        self.value = value
        self.key_expansion = key_expansion
        self.time_aware = time_aware
        self.chain_of_note = chain_of_note
        self.index = VectorIndex(self._emb)
        self.docs: dict[str, str] = {}
        self._n = 0

    def reset(self) -> None:
        self.index = VectorIndex(self._emb)
        self.docs = {}
        self._n = 0

    # CP1 — value granularity
    def add(self, item: MemoryItem) -> None:
        self.add_doc(item.text, item.timestamp)

    def add_session(self, turns: list[Turn]) -> None:
        if self.value == "session":
            text = " ".join(t.text for t in turns)
            self.add_doc(text, turns[0].ts if turns else None)
        else:
            for t in turns:
                self.add_doc(t.text, t.ts)

    def add_doc(self, text: str, ts) -> None:
        docs = [text]
        if self.value == "facts":
            fs = [f for f in extract_facts(text)
                  if not f.startswith("remove:")]
            docs = fs or [text]
        for d in docs:
            self._n += 1
            did = f"d{self._n}"
            indexed = d
            # CP2 — key expansion: indexar fatos extraídos junto do texto.
            if self.key_expansion:
                extra = [f for f in extract_facts(d)
                         if not f.startswith("remove:")]
                if extra:
                    indexed = d + " || " + " ".join(extra)
            self.docs[did] = (d if ts is None else
                              f"{ts.strftime('%Y-%m-%d')} {d}")
            self.index.add(did, indexed)

    # CP3 — query
    def _expand_query(self, query: str) -> str:
        if not self.time_aware:
            return query
        q = query.lower()
        if re.search(r"\b(before|previous|earlier|ago|last|when)\b", q):
            # PT: TR — pede o valor MAIS ANTIGO. EN: TR — ask for the
            # OLDER value.
            return query + " previously earliest older before"
        if re.search(r"\b(now|current|still|today)\b", q):
            return query + " currently latest most recent now"
        return query

    def retrieve(self, query: str, *, k: int = 5, now=None) -> list[str]:
        q = self._expand_query(query)
        return [self.docs[d] for d, _ in self.index.search(q, k)]

    # CP4 — reading
    def answer(self, llm: LLM, question: str, k: int = 8) -> str:
        ev = self.retrieve(question, k=k)
        if not self.chain_of_note:
            return answer_with_evidence(llm, question, ev)
        body = (f"QUESTION: {question}\nEVIDENCE:\n"
                + "\n".join(f"- {e}" for e in ev)
                + "\nWrite one short note per evidence, then nothing else.")
        notes = llm.complete(task_prompt("qa.chain_of_note", body))
        note_lines = [ln[6:].strip() for ln in notes.splitlines()
                      if ln.startswith("Note: ") and "irrelevant" not in ln]
        return answer_with_evidence(llm, question, note_lines or ev)
