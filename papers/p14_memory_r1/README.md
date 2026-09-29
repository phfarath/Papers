# p14 — Memory-R1 (arXiv 2508.19828)

## Português

Replicação offline do **Memory-R1**: RL treina a **política do gerente de
memória** (Memory Manager) e do **Answer Agent** — e não as memórias em si.

- **Memory Manager** escolhe **ADD / UPDATE / DELETE / NOOP** por fato
  ingerido, via política linear numpy (`LinearPolicy`) sobre features de
  overlap.
- **Answer Agent** faz **destilação de memória**: dos **≤60** candidatos
  recuperados, seleciona os ≤5 relevantes para a pergunta (`distill`,
  `MAX_CANDIDATES=60`).
- Treinamento com **PPO simplificado** (vantagem clipada `[-1,1]` vs
  baseline móvel) e **GRPO** (vantagem relativa ao **grupo de 8**
  ingestões), recompensa = exact-match/QA.
- Curvas de aprendizado vs **heurístico** (regra fixa) e **não-treinado**
  (sempre ADD); QA no dataset conversacional vs os 3 baselines.

### Memory-R1 vs MemRL (p03)

Em **MemRL** o aprendizado (valor Q) vive **nas memórias** — triplets
(intenção, memória, utilidade) — e o LLM agente fica congelado; a política de
uso muda porque o *valor estimado de cada memória* muda. Em **Memory-R1** o
LLM/retriever também é fixo, mas o que o RL treina é a **política do
gerente** (que ops aplicar) e do **respondedor** (que candidatos destilar):
o aprendizado vive nos *parâmetros das políticas*, não nas memórias.

### Mapeamento paper → código

| Paper | Código |
|---|---|
| Memory Manager com ops ADD/UPDATE/DELETE/NOOP | `MemoryR1Manager` |
| Answer Agent com distilação sobre ≤60 candidatos | `distill`, `MAX_CANDIDATES=60` |
| PPO e GRPO (group-relative) | `LinearPolicy.update`, `train` |
| Recompensa exact-match | `exact_match`, `_qa_reward` |

### Rodar / Run

```bash
python -m papers.p14_memory_r1.run --seed 0 --write-results
```

---

## English

Offline replication of **Memory-R1**: RL trains the **memory manager
policy** and the **answer agent** — not the memories themselves. The manager
picks **ADD/UPDATE/DELETE/NOOP** per ingested fact via a linear numpy
policy; the answer agent **distills** ≤5 evidence items from ≤60 retrieved
candidates. Training uses **simplified PPO** (clipped advantage vs moving
baseline) and **GRPO** (advantage relative to a group of 8 ingestions) on
exact-match reward, with learning curves vs a **heuristic** manager and an
**untrained** (always-ADD) baseline, plus conversational QA vs the 3 shared
baselines. **Contrast with MemRL (p03):** MemRL puts learned Q-values *on
the memories* with a frozen agent; Memory-R1 puts the learning in the
manager/answerer *policy parameters* — the README above explains this. A
`MemorySystem` adapter is included in the shared registry.
