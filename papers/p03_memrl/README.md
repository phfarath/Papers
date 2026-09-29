# p03 — MemRL: non-parametric agent self-evolution via memory Q-values

arXiv: https://arxiv.org/abs/2601.03192 — MemTensor, 2026.
Code: https://github.com/MemTensor/MemRL

## 🇧🇷 Português

### Resumo
O MemRL organiza a memória em **triplas (z, e, Q)**: intent, experiência e
utilidade aprendida. A recuperação tem **duas fases**: (A) recall por
similaridade com limiar δ → C(s); (B) seleção top-k2 por score
`(1−λ)·ẑ(sim) + λ·ẑ(Q)` com z-score. Após cada trajetória, o Q das memórias
injetadas é atualizado por Monte Carlo `Q ← Q + α(r − Q)` (Eq. 4), e um resumo
da trajetória vira uma nova tripla `(z, e_new, Q_init)` — tudo sem tocar nos
pesos do LLM.

### Ideia central em linguagem simples
RAG comum é como perguntar ao amigo que "parece saber mais" (similaridade).
MemRL é perguntar a quem parece saber **E** já acertou antes — e ir atualizando
a reputação de cada conselho conforme ele funciona ou não na prática.

### Algoritmo passo a passo
```
para cada episódio:
    1. C(s) ← TopK_{k1}({i | cos(Emb(s), Emb(z_i)) > δ})     # Fase A
       se vazio → só o LLM, sem memória
    2. M_ctx ← TopK_{k2}(C(s)) por (1−λ)ẑ(sim) + λ ẑ(Q)      # Fase B
    3. a ~ π_LLM(· | s, M_ctx); r ← ambiente
    4. Q_i ← Q_i + α(r − Q_i) para i ∈ M_ctx                 # Eq. 4 (ou TD Eq. 3)
    5. e_new ← LLM.resume(τ); memória += (z, e_new, Q_init)
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| §4.1 tripla (z, e, Q) | `method.py: Triplet` |
| §4.2 Fase A, C(s) com δ | `method.py: MemRLMemory.phase_a` |
| §4.2 Fase B, score (1−λ)ẑ(sim)+λẑ(Q) | `method.py: phase_b` (+ `utils.zscore`) |
| §4.3 Eq. 4 Monte Carlo / Eq. 3 TD | `method.py: MemRLMemory.update(td=)` |
| §4.3 resumo da trajetória → nova tripla | `prompts.py: memrl.summarize`, `run.py` |
| §5.4.2 ablação λ (melhor λ=0.5, Fig. 5) e z-score/gating | `run.py` ablações |

### Como rodar
```bash
python -m papers.p03_memrl.run --seed 0 --llm mock --write-results
```

### O que observar
- MemRL (λ>0) deve superar RAG por similaridade (λ=0) — os distratores usam a
  redação exata da tarefa, então λ=0 os injeta sempre; com λ>0 o Q aprendido os
  rebaixa. O baseline sem memória fica próximo do MemRL: os distratores são um
  handicap que o MemRL precisa vencer, não uma vantagem gratuita.
- A ablação de λ deve ser NÃO-plana (λ=0 baixo, λ→1 melhor), e a correlação
  Q×sucesso deve ser positiva.
- A tabela de Q aprendidos deve ranquear experiências úteis acima de
  distratores — mas note que um distrator pode ter Q alto se funcionou em
  algumas tarefas (o Q estima utilidade empírica, não "verdade").
- Defaults δ=0.25, λ=0.5, α=0.3, Q_init=0.5, k1=4, k2=2, γ=0.9: o repo oficial
  não publica defaults; λ=0.5 segue a Fig. 5 do paper. Escolha documentada.

### Diferenças vs paper e limitações
- MockLLM + env-brinquedo: o Q aprende utilidade empírica no brinquedo, não em
  ALFWorld/HLE reais. Números não são comparáveis ao paper.
- O resumo da trajetória é gerado por regras (itens quebrados/portas vistas),
  não por um LLM real.

## 🇺🇸 English

### Summary
MemRL stores memory as **(z, e, Q) triplets** — intent, experience, learned
utility. Retrieval is **two-phase**: (A) similarity recall with threshold δ,
(B) top-k2 selection by `(1−λ)·ẑ(sim) + λ·ẑ(Q)` (z-scored). After each
trajectory, injected memories get the Monte Carlo update `Q ← Q + α(r − Q)`
(Eq. 4) and a trajectory summary becomes a new triplet — all without touching
LLM weights.

### Core idea in plain words
Plain RAG asks the friend who "sounds most similar". MemRL asks whoever sounds
similar **AND** has a good track record — and keeps updating each piece of
advice's reputation as it works or fails in practice.

### Algorithm, mapping, how to run, what to look at
See the Portuguese section (same content).

### Differences vs paper & limitations
MockLLM + toy env; Q learns empirical utility in the toy, not real ALFWorld/HLE
returns. Summaries are rule-generated, not LLM-written.

### Referência / Reference
MemTensor. "MemRL: non-parametric agent self-evolution." 2026.
https://arxiv.org/abs/2601.03192
