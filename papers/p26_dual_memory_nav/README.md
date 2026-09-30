# p26 — Dual-Timescale Memory for Navigation

Yana Tsybina et al. — 2026 (arXiv:2604.15391).

## 🇧🇷 Português

### Resumo
O paper argumenta que navegação eficiente precisa de **duas memórias em
escalas temporais distintas**: uma memória de longo prazo que consolida
trajetórias bem-sucedidas, e uma memória de curto prazo — implementada
pela glia — que marca células visitadas recentemente e as suprime,
dirigindo a exploração para o novo. Sem a glia, o agente vagueia; com
ela, a exploração é dirigida por novidade e a consolidação é mais rápida.

### Ideia central em linguagem simples
A glia é o "rastro de migalhas invertido": marca onde você JÁ passou
para você não perder tempo voltando — enquanto a memória neural lenta
grava o caminho que deu certo.

### Algoritmo passo a passo
```
a cada episódio:
    sup = 0 (supressão glial por célula)
    a cada passo:
        score(a) = Q[destino,a] − β·sup[destino]   # exploração dirigida
        sup *= decay; sup[célula atual] += 1
    ao atingir o objetivo:
        trace-back: Q[trajetória] += α(1−Q), α *= λ a cada passo atrás
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| memória de longo prazo | `method.py: Navigator.Q` + trace-back |
| memória de curto prazo glial | `method.py: sup` (decay por episódio) |
| exploração dirigida por novidade | `score = Q − β·sup` no `episode()` |
| duas escalas cooperando | `run.py` curva de aprendizado e β sweep |

### Como rodar
```bash
python -m papers.p26_dual_memory_nav.run --seed 0 --write-results
```

### O que observar
- Só com Q o agente falha em ~1/5 dos episódios (ciclos, vagueio); com
  glia converge a ~40-50 passos e não falha.
- O sweep β mostra o balanço: sem supressão (β=0) = controle; β forte
  demais interfere na consolidação.

### Diferenças vs paper e limitações
- Grid determinístico 10×10 com um objetivo; o paper trata ambientes
  mais ricos (obstáculos, múltiplas recompensas).
- A "consolidação" é um trace-back Hebbiano simplificado — não é TD(λ)
  completo nem um modelo biofísico de astrócito.
- A supressão é por célula e decai exponencialmente — proxy da fadiga
  glial, não um modelo de Ca²⁺.

## 🇺🇸 English

### Summary
The paper argues efficient navigation needs **two memories at distinct
timescales**: a long-term memory consolidating successful trajectories,
and a short-term one — implemented by glia — marking recently visited
cells and suppressing them, driving exploration toward the novel.
Without glia the agent wanders; with it exploration is novelty-driven
and consolidation is faster.

### Core idea in plain words
Glia is the "inverted breadcrumb trail": it marks where you've ALREADY
been so you don't waste time backtracking — while slow neural memory
records the path that worked.

### How to run
```bash
python -m papers.p26_dual_memory_nav.run --seed 0 --write-results
```

### What to look at
- With Q only, the agent fails ~1/5 of episodes (loops, wandering); with
  glia it converges to ~40-50 steps and never fails.
- The β sweep shows the balance: no suppression (β=0) = control; too
  strong interferes with consolidation.

### Differences vs the paper and limitations
- Deterministic 10×10 grid with one goal; the paper treats richer
  environments (obstacles, multiple rewards).
- "Consolidation" is a simplified Hebbian trace-back — not full TD(λ)
  nor a biophysical astrocyte model.
- Suppression is per-cell exponential decay — a proxy for glial fatigue,
  not a Ca²⁺ model.
