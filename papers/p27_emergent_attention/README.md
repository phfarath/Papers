# p27 — Emergent Self-Attention via Replicator Dynamics

Vivet & Arenas — 2026 (arXiv:2604.25481).

## 🇧🇷 Português

### Resumo
O paper propõe que a alocação tipo atenção pode **emergir de dinâmica
replicadora** sobre ganhos sinápticos — sem uma softmax hard-coded.
Cada item multiplica seu ganho por seu fitness e a glia fornece a
normalização global que mantém os ganhos no simplex. O resultado
analítico é exato: `g_t = softmax(β·t·f)` — a dinâmica local gera a
alocação softmax e, no limite, winner-take-all.

### Ideia central em linguagem simples
Num ecossistema, quem rende mais multiplica mais — a "atenção" é a
proporção populacional de cada tipo. A glia é quem garante que a
população total não explode: a divisão que mantém as frações somando 1.

### Algoritmo passo a passo
```
g uniforme no simplex; fitness f_i por item
repete:  g_i <- g_i · e^{β·f_i} / Z,   Z = Σ_j g_j·e^{β·f_j}  (glia)
→ g_t = softmax(β·t·f)  (analítico; entropia → 0, winner-take-all)
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| dinâmica replicadora nos ganhos | `method.py: ReplicatorAttention.step` |
| normalização global glial | o termo `Z` no `step` (ablação `glia=False`) |
| softmax emerge, não é imposta | `run.py` tabela g_t vs softmax(βt·f) |
| winner-take-all | `run.py` colapso de entropia |

### Como rodar
```bash
python -m papers.p27_emergent_attention.run --seed 0 --write-results
```

### O que observar
- `g_t` casa com `softmax(β·t·f)` a precisão de máquina para vários β.
- A entropia de `g` colapsa ao longo dos passos — a alocação concentra
  no item de maior fitness.
- Sem a normalização glial (Z=1) os ganhos explodem — a glia é o
  vínculo do simplex, não um detalhe.

### Diferenças vs paper e limitações
- O paper formula o replicator no contexto de camadas de atenção com
  fitness por query–key; usamos fitness genérico por item — a
  correspondência do mecanismo é a mesma.
- Mostramos o mapa exponenciado exato; variantes de tempo contínuo e
  com ruído não são exploradas.
- É uma correspondência matemática — não treinamos um Transformer.

## 🇺🇸 English

### Summary
The paper proposes that attention-like allocation can **emerge from
replicator dynamics** over synaptic gains — no hard-coded softmax.
Each item multiplies its gain by its fitness and glia supplies the
global normalization keeping gains on the simplex. The analytic result
is exact: `g_t = softmax(β·t·f)` — local dynamics generate the softmax
allocation and, in the limit, winner-take-all.

### Core idea in plain words
In an ecosystem, higher-yielding types multiply faster — "attention" is
each type's population share. Glia keeps the total population from
exploding: the division keeping fractions summing to 1.

### How to run
```bash
python -m papers.p27_emergent_attention.run --seed 0 --write-results
```

### What to look at
- `g_t` matches `softmax(β·t·f)` to machine precision across β.
- The entropy of `g` collapses over steps — allocation concentrates on
  the highest-fitness item.
- Without glial normalization (Z=1) gains explode — glia is the
  simplex constraint, not a detail.

### Differences vs the paper and limitations
- The paper frames the replicator in attention layers with query–key
  fitness; we use generic per-item fitness — same mechanism.
- We show the exact exponentiated map; continuous-time and noisy
  variants are not explored.
- A mathematical correspondence — no Transformer is trained.
