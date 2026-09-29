# p17 — LongMemEval-V2 (arXiv 2605.12493v1)

## Português

Replicação offline do benchmark de **memória sobre trajetórias de agente
web**: um explorador scriptado (com ruído determinístico) executa as 5
tarefas do MiniWeb e grava trajetórias (ação → observação). As 5
competências do paper (§3.1): static state recall, dynamic state tracking,
workflow knowledge, environment gotchas, premise awareness (inclui
perguntas de **falsa premissa** que devem ser rejeitadas).

Formulação de *context gathering* (§3.3): a memória consome trajetórias e
devolve evidência compacta dentro de um **orçamento de 300 tokens** → o
leitor responde.

**Baselines do paper (§4):**
- `AgentRunbookR` — pools separados (observações de estado, eventos,
  notas de gotcha/estratégia destiladas na ingestão) fundidos no orçamento;
- `AgentRunbookC` — trajetórias como **arquivos** num diretório
  temporário; um "coding agent" usa chamadas restritas (`grep`/`read`)
  para coletar evidência;
- `truncated-history` e `plain-rag` como referências.

Métricas: acurácia por competência + latência por pergunta. Resultados em
`RESULTS.md` — não são os números do paper.

## English

Offline replication of the **web-agent trajectory memory** benchmark: a
scripted explorer (with deterministic noise) runs MiniWeb's 5 tasks and
records trajectories (action → observation). The paper's 5 competencies
(§3.1): static state recall, dynamic state tracking, workflow knowledge,
environment gotchas, premise awareness (including **false-premise**
questions that must be rejected).

Context-gathering formulation (§3.3): memory consumes trajectories and
returns compact evidence under a **300-token budget** → the reader
answers.

**Paper baselines (§4):** `AgentRunbookR` (separate pools: state obs /
action events / distilled gotcha-strategy notes, merged under budget),
`AgentRunbookC` (trajectories as **files** in a temp dir; a "coding agent"
gathers evidence via restricted `grep`/`read` calls), plus
`truncated-history` and `plain-rag`.

Metrics: per-competency accuracy + per-question latency. Results in
`RESULTS.md` — not the paper's numbers.
