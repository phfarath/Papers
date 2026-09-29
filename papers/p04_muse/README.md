# p04 — MUSE: Competence-Aware AI Agents with Metacognition

arXiv: https://arxiv.org/abs/2411.13537 — Valiente & Pilly.
**Não confundir** com o paper "MUSE" de outra linha (aqui é metacognição:
self-assessment + self-regulation).

## 🇧🇷 Português

### Resumo
O MUSE dá ao agente **consciência de competência**: um Self-Assessment Model
(M_sa = encoder + MLP sigmoid, treinado com BCE) aprende a prever a chance de
sucesso de uma trajetória. No deployment, a cada passo o Actor/World-Model gera
**5 rollouts hipotéticos** (temperatura 0.5), M_sa pontua cada um, e o agente
executa a **1ª ação do rollout com maior probabilidade de sucesso**
(self-regulation). Após cada episódio, g_η é atualizado online com replay.

### Ideia central em linguagem simples
É como um jogador que, antes de cada lance, imagina 5 sequências mentais de
jogadas e escolhe a primeira jogada da sequência que seu "instinto de
auto-avaliação" (treinado em jogos anteriores) acha que vai dar certo.

### Algoritmo passo a passo
```
Pré-deployment:
    1. roda Reflexion nas tarefas in-distribution → trajetórias
    2. treina g_η (BCE) em chunks de 4 pares ação-observação (Alg. 2, Eq. 6-7)
    3. datasets SFT (episódios de sucesso) e DPO (falha e_i → sucesso e_{i+1})
Deployment (a cada passo t):
    4. Actor gera 5 rollouts hipotéticos a partir de (o_t, c_{t-1}, I, rx)
    5. y_pred = M_sa(τ_i, P^e, I) para cada rollout i
    6. executa a 1ª ação do rollout com maior y_pred
    7. falha → nova reflexão; fim do episódio → replay → update de g_η (Alg. 4)
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| §4.1.3 M_sa = encoder + MLP, sigmoid, BCE (Eq. 6-7, Alg. 2) | `method.py: SelfAssessment` |
| chunks não sobrepostos de 4 pares ação-obs | `method.py: chunk_pairs` (CHUNK=4) |
| 5 rollouts, temp 0.5, 1ª ação do melhor | `prompts.py: muse.rollout`, `MuseAgent.act` (N_ROLLOUTS=5) |
| Reflexion p/ coleta pré-deployment | `run.py: collect_predeployment` (p02) |
| reflexão em falhas + reflexões também em sucesso | `MuseAgent.run_episode` |
| DPO (positivo: falha→sucesso; negativo: sucesso→falha) e SFT | `method.py: build_dpo_dataset`, `build_sft_dataset` |
| métricas metacog. acc + AUROC2, ttc com penalidade 100 | `run.py`, `method.py: auroc` |
| protocolo sem adaptação vs 5 adapt + 5 test | `run.py` |

### Como rodar
```bash
python -m papers.p04_muse.run --seed 0 --llm mock --write-results
```

### O que observar
- `sft.jsonl`/`dpo.jsonl` são gerados na pasta — o treino de pesos do Actor e do
  Reflection LLM **não roda offline** (com MockLLM não há pesos); só g_η treina.
- MUSE com adaptação deve superar MUSE sem adaptação; a acurácia metacognitiva e
  AUROC2 medem quão bem M_sa prevê sucesso.

### Diferenças vs paper e limitações
- O paper usa SentenceTransformer + MLP real e LoRA para SFT/DPO; aqui o encoder
  é o HashingEmbedder e g_η é uma regressão logística (MLP de 1 camada) em numpy.
- A versão "world model" (Dreamer-v3 + RSSM + cabeça de quantis N=5) é opcional
  no spec e **não** implementada — implementamos só a versão LLM (seção 4).
- Rollouts "hipotéticos" são gerados pelo mock com heurística; um LLM real
  imaginaria transições mais fiéis ao ambiente.

## 🇺🇸 English

### Summary
MUSE gives the agent **competence awareness**: a Self-Assessment Model (encoder
+ sigmoid MLP, BCE) learns to predict the probability of trajectory success. At
deployment, each step the Actor/World-Model generates **5 hypothetical
rollouts** (temperature 0.5), M_sa scores them, and the agent executes the
**first action of the best-scoring rollout** (self-regulation). After each
episode, g_η is updated online with a replay buffer.

### Core idea in plain words
Like a player who, before each move, imagines 5 possible continuations and
plays the first move of the continuation their trained gut feeling rates most
likely to succeed.

### Algorithm, mapping, how to run
See the Portuguese section (same content).

### Differences vs paper & limitations
- The paper uses a real SentenceTransformer + MLP and LoRA SFT/DPO; here the
  encoder is HashingEmbedder and g_η is logistic regression in numpy.
- The world-model variant (Dreamer-v3 + RSSM + N=5 quantile head) is optional
  in the spec and **not** implemented — only the LLM version (section 4).
- Hypothetical rollouts come from the mock heuristic.

### Referência / Reference
Valiente & Pilly. "Competence-Aware AI Agents with Metacognition for Unknown
Situations and Environments (MUSE)." 2024. https://arxiv.org/abs/2411.13537
