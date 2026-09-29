# p16 — MemoryAgentBench (arXiv 2507.05257)

## Português

Replicação offline do **benchmark** MemoryAgentBench: ingestão incremental
em **chunks** (a pergunta só chega no fim — protocolo do paper, §3.3) e 4
competências (§3.1):

- **AR** — recuperação precisa de um fato do stream;
- **TTL** — aprendizado em-contexto de um mapeamento de rótulos
  (exemplos no stream → nova classificação);
- **LRU** — agregação sobre o stream inteiro, medida por cobertura de
  palavras-chave do ouro;
- **SF** — esquecimento seletivo: fatos posteriores sobrescrevem os
  anteriores (single-hop + multi-hop).

Avalia os 3 baselines (FullContext truncado, BM25, RAG) + todos os
adapters `all_memory_systems` + oráculo. Resultados em `RESULTS.md` —
não são os números do paper. Nota honesta: o Mem0g (arestas sujeito-
relação-objeto) perde o texto do fato e fica em 0.00.

## English

Offline replication of the **MemoryAgentBench** benchmark: incremental
**chunked** ingestion (the question only arrives at the end — the paper's
protocol, §3.3) and 4 competencies (§3.1): AR (accurate retrieval), TTL
(in-context label mapping learned from stream examples), LRU (whole-stream
aggregation scored by gold keyword coverage), SF (selective forgetting —
later facts overwrite earlier ones, single+multi-hop).

Evaluates the 3 baselines + all `all_memory_systems` adapters + oracle.
Results in `RESULTS.md` — not the paper's numbers. Honest note: Mem0g
(subject-relation-object edges) loses the fact text and sits at 0.00.
