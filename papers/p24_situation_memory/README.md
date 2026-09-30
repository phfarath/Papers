# p24 — Situation-Based Neuromorphic Memory

Susanna Gordleeva et al. — Frontiers in Neuroscience 2025.
DOI: 10.3389/fnins.2025.1768235

## 🇧🇷 Português

### Resumo
O paper propõe memória organizada em **pools de padrões ligados por
situações**: a mesma entrada (cue) pode exigir respostas diferentes
conforme a situação/cenário corrente. Unidades de situação só disparam
quando cue E contexto casam — e a glia funciona como retenção lenta da
situação corrente, fazendo "latch" do contexto e segurando-o quando a
pista contextual some.

### Ideia central em linguagem simples
Um café "sem açúcar" num restaurante e o mesmo pedido em outro lugar têm
respostas diferentes — quem decide é o contexto. A glia é quem lembra
"em que restaurante você está" enquanto a conversa rola.

### Algoritmo passo a passo
```
unidades de situação s_{k,c}: preferem cue x_k E contexto c
codificar: glia integra o contexto (latch, decai com τ_ctx)
recall: ativação s ∝ x·x̂ + ctx·ĉ; só s>θ dispara → lê y_{k,c}
contexto removido? usa o latch glial como fonte do contexto
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| pools de padrões por situação | `method.py: SituationMemory.S/out` |
| gating por cue×contexto | `method.py: recall_gated` (θ-threshold) |
| glia retém a situação corrente | `method.py: astro` (latch, τ_ctx) |
| memória persiste sem pista | `run.py` context_persistence(gap) |

### Como rodar
```bash
python -m papers.p24_situation_memory.run --seed 0 --write-results
```

### O que observar
- Hebbian plano mistura respostas de contextos diferentes (~0.55 cos);
  o gating situacional recupera a resposta correta (~0.9).
- Com a pista contextual removida, o recall depende do latch glial:
  sobrevive a gaps da ordem de τ_ctx e decai depois.

### Diferenças vs paper e limitações
- É uma implementação conceitual de pools de padrões — o paper detalha
  arquitetura neuromórfica específica (memristiva/SNN).
- Usamos Hebbian one-shot; o paper discute aprendizado e reconsolidação.
- O "latch" glial é um traço exponencial — proxy da retenção contextual,
  não um modelo biofísico.

## 🇺🇸 English

### Summary
The paper proposes memory organized in **pattern pools bound by
situations**: the same cue can require different responses depending on
the current situation. Situation units fire only when cue AND context
match — and glia acts as a slow holder of the current situation,
latching the context and keeping it when the contextual cue disappears.

### Core idea in plain words
"Coffee, no sugar" means different things in different cafés — context
decides. Glia is what remembers "which café you're in" while the
conversation goes on.

### How to run
```bash
python -m papers.p24_situation_memory.run --seed 0 --write-results
```

### What to look at
- Flat Hebbian mixes the two contexts' responses (~0.55 cos); the
  situation gate retrieves the right one (~0.9).
- With the contextual cue removed, recall depends on the glial latch:
  it survives gaps on the order of τ_ctx and decays afterwards.

### Differences vs the paper and limitations
- A conceptual pattern-pool implementation — the paper details a
  specific neuromorphic (memristive/SNN) architecture.
- One-shot Hebbian storage; the paper discusses learning and
  reconsolidation.
- The glial "latch" is an exponential trace — a proxy for contextual
  retention, not a biophysical model.
