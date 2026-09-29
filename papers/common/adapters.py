"""PT: registro de MemorySystem — baselines + adapters dos papers.

EN: MemorySystem registry — baselines + paper adapters.

Direção dos imports (evita ciclo): o registry IMPORTA os papers; os papers
NUNCA importam o registry.
"""

from __future__ import annotations

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.memory_api import (
    BM25Memory,
    EmbeddingRAGMemory,
    FullContextMemory,
    MemorySystem,
)


def all_memory_systems(llm: LLM,
                       embedder: Embedder | None = None) -> list[MemorySystem]:
    """PT: 3 baselines + um adapter por paper de memória (p10–p13 + p14
    answer-agent). EN: 3 baselines + one adapter per memory paper."""
    emb = embedder or HashingEmbedder()
    systems: list[MemorySystem] = [
        FullContextMemory(max_tokens=800),  # PT: orçamento truncado. EN: budget.
        BM25Memory(),
        EmbeddingRAGMemory(emb),
    ]
    # PT: imports tardios aqui de propósito — o registry puxa os papers, e
    # nunca o contrário. EN: late imports on purpose — the registry pulls the
    # papers, never the other way around.
    from papers.p10_memgpt.adapter import MemGPTAdapter
    from papers.p11_a_mem.adapter import AMemAdapter
    from papers.p12_zep.adapter import ZepAdapter
    from papers.p13_mem0.adapter import Mem0Adapter, Mem0gAdapter
    from papers.p14_memory_r1.adapter import MemoryR1Adapter

    systems += [
        MemGPTAdapter(llm, emb),
        AMemAdapter(llm, emb),
        ZepAdapter(llm, emb),
        Mem0Adapter(llm, emb),
        Mem0gAdapter(llm, emb),
        MemoryR1Adapter(llm, emb),
    ]
    return systems
