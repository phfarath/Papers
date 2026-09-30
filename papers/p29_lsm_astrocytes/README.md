# p29 — Liquid State Machine with Astrocyte-Like Units

Christopher S. Yang et al. — Neurocomputing 673:132805 (2026).
Preprint arXiv:2503.06798

## 🇧🇷 Português

### Resumo
O paper adiciona a um Liquid State Machine **unidades astrocíticas
lentas** — elementos com constante de tempo muito maior que a dos
neurônios que integram a atividade do reservatório e enriquecem o
readout com memória de longa escala. O resultado central é um
**óptimo na razão astrócito:neurônio**: com poucas unidades a memória
longa falta; com demais, o readout dilui/overfita.

### Ideia central em linguagem simples
Neurônios rápidos são como anotações em post-its: detalhe agora, mas o
papel voa em segundos. Os astrocitos são o caderno lento: guardam o que
importou no passado longo — mas cadernos demais só repetem a mesma
nota e atrapalham a leitura.

### Algoritmo passo a passo
```
reservatório: r <- tanh(W r + w_in·x)
astrocitos:   a += (−a + tanh(Wna·r))/τ_a    (τ_a = 80 >> 1)
readout ridge sobre [r | a] → predição s[t+60]
sweep da razão M/N de unidades lentas
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| unidades lentas tipo astrócito | `method.py: LiquidAstro.a` (τ_a=80) |
| LSM + glia para séries temporais | `method.py: step` features [r|a] |
| razão astro:neurônio ótima | `run.py` sweep de `ratio` |

### Como rodar
```bash
python -m papers.p29_lsm_astrocytes.run --seed 0 --write-results
```

### O que observar
- `ratio=0` (sem glia): NRMSE alto — o reservatório esquece o atraso
  τ=17 da série.
- `ratio≈0.5`: NRMSE mínimo — memória longa suficiente sem diluir.
- `ratio≥2`: NRMSE volta a subir — features redundantes overfitam.

### Diferenças vs paper e limitações
- O paper detalha a implementação em LSM de spikes; usamos unidades
  tanh rate-based com a mesma separação de escalas temporais.
- Os astrocitos não realimentam o reservatório (só leem) — o feedback
  deles na dinâmica é uma variante que o paper também explora.
- Uma única série (Mackey-Glass) e um horizonte; o paper avalia outras
  tarefas.

## 🇺🇸 English

### Summary
The paper adds **slow astrocyte-like units** to a Liquid State
Machine — elements with a much longer time constant than the neurons
that integrate reservoir activity and enrich the readout with
long-timescale memory. The central result is an **optimum in the
astrocyte:neuron ratio**: too few units leave long memory missing; too
many dilute/overfit the readout.

### Core idea in plain words
Fast neurons are like sticky notes: detail now, gone in seconds.
Astrocytes are the slow notebook: they keep what mattered in the long
past — but too many notebooks just repeat the same note and clutter
the reading.

### How to run
```bash
python -m papers.p29_lsm_astrocytes.run --seed 0 --write-results
```

### What to look at
- `ratio=0` (no glia): high NRMSE — the reservoir forgets the series'
  τ=17 delay.
- `ratio≈0.5`: minimum NRMSE — enough long memory without dilution.
- `ratio≥2`: NRMSE rises again — redundant features overfit.

### Differences vs the paper and limitations
- The paper details a spiking LSM implementation; we use rate-based
  tanh units with the same timescale separation.
- Astrocytes only read the reservoir (no feedback) — their feedback
  into the dynamics is a variant the paper also explores.
- A single series (Mackey-Glass) and horizon; the paper evaluates
  other tasks.
