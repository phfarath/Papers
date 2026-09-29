"""PT: avaliação compartilhada do benchmark conversacional (p10–p14 e
batch 4): ingere todos os turnos e mede acurácia por tipo de pergunta via
qa.judge. Inclui linha "FullContext (oracle, untruncated)" como teto.

EN: shared evaluation of the conversational benchmark (p10–p14 and batch
4): ingest all turns and measure per-type accuracy via qa.judge. Includes
a "FullContext (oracle, untruncated)" upper-bound row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from papers.common.conv_data import ConvSet
from papers.common.llm import LLM
from papers.common.memory_api import (
    BM25Memory,
    EmbeddingRAGMemory,
    FullContextMemory,
    MemoryItem,
)
from papers.common.reader import answer_with_evidence, judge_answer

QTYPES = ["single-hop", "multi-hop", "temporal", "update", "abstention"]


@dataclass
class EvalResult:
    """PT: acurácia global + por tipo. EN: overall + per-type accuracy."""
    name: str
    overall: float
    per_type: dict[str, float] = field(default_factory=dict)
    extra: str = ""


def eval_memory(mem, data: ConvSet, llm: LLM, k: int = 8,
                name: str = "") -> EvalResult:
    """PT: ingere turnos e responde todas as perguntas; devolve acurácia por
    tipo. EN: ingest turns, answer all questions; per-type accuracy."""
    for t in data.all_turns():
        mem.add(MemoryItem(t.text, t.ts))
    by: dict[str, list[int]] = {qt: [] for qt in QTYPES}
    for q in data.questions:
        ev = mem.retrieve(q.question, k=k)
        pred = answer_with_evidence(llm, q.question, ev)
        by.setdefault(q.qtype, []).append(
            int(judge_answer(llm, q.question, q.gold, pred)))
    allv = [v for vs in by.values() for v in vs]
    return EvalResult(
        name or getattr(mem, "name", type(mem).__name__),
        sum(allv) / len(allv) if allv else 0.0,
        {qt: (sum(vs) / len(vs) if vs else 0.0) for qt, vs in by.items()})


def baseline_rows(data: ConvSet, llm: LLM, k: int = 8) -> list[EvalResult]:
    """PT: linhas de referência: oracle não-truncado, truncado, BM25, RAG.
    EN: reference rows: untruncated oracle, truncated, BM25, RAG."""
    return [
        eval_memory(FullContextMemory(max_tokens=100_000), data, llm, k,
                    "FullContext (oracle, untruncated)"),
        eval_memory(FullContextMemory(max_tokens=200), data, llm, k,
                    "FullContext (truncated)"),
        eval_memory(BM25Memory(), data, llm, k, "BM25"),
        eval_memory(EmbeddingRAGMemory(), data, llm, k, "EmbeddingRAG"),
    ]


def render_acc_table(rows: list[EvalResult],
                     extra_col: str | None = None) -> list[str]:
    """PT: linhas markdown da tabela de acurácia por tipo.
    EN: markdown lines for the per-type accuracy table."""
    head = ("| memory system | overall | " + " | ".join(QTYPES)
            + (" | " + extra_col if extra_col else "") + " |")
    sep = "|---|" * (3 + len(QTYPES) + (1 if extra_col else 0))
    out = [head, sep]
    for r in rows:
        cells = " | ".join(f"{r.per_type.get(qt, 0.0):.2f}" for qt in QTYPES)
        line = f"| {r.name} | {r.overall:.2f} | {cells}"
        if extra_col:
            line += f" | {r.extra}"
        out.append(line + " |")
    return out
