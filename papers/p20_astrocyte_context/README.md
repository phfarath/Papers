# p20 — Astrocytes as a Mechanism for Contextually-Guided Network Dynamics and Function

Lulu Gong, Fabio Pasqualetti, Thomas Papouin, ShiNung Ching — PLOS
Computational Biology, 2024. DOI: 10.1371/journal.pcbi.1012186

## 🇧🇷 Português

### Resumo
Modelo de **laços aninhados em escalas temporais separadas**: pesos
rápidos f aprendem pela regra delta da tarefa; astrócitos a_c, mais
lentos, **consolidam** f enquanto o contexto c está ativo; e a cada trial
a restauração gateada `f ← f + η_r(a_c − f)` reintroduz a configuração
aprendida daquele contexto. É o modelo do paper de **metaplasticidade**:
a glia regula a própria plasticidade por contexto. Avaliado em reversão
de regras (B = −A), o paradigma do paper.

### Ideia central em linguagem simples
Aprender a regra A, depois a regra B inversa, faz a rede "brigar"
consigo mesma. Os astrócitos funcionam como um arquivo por contexto:
guardam "como eu resolvia isso da última vez" e devolvem quando o
contexto volta — sem recomeçar do zero.

### Algoritmo passo a passo
```
f ← 0; a_c ← 0 para cada contexto c
a cada trial (contexto c ativo):
    err ← (w*_c·x) − (f·x)
    f  ← f + η_f·err·x/N           # plasticidade rápida da tarefa
    f  ← f + η_r·(a_c − f)          # restauração gateada pelo contexto
    a_c← a_c + η_a·(f − a_c)        # consolidação lenta
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| laços neurônio–sinapse–astrócito em escalas separadas | `method.py: ContextGatedNet` (η_f ≫ η_a) |
| modulação contextual da dinâmica/plasticidade | `method.py: trial` (restauração gateada `eta_r`) |
| aprendizado com parâmetros de tarefa flutuando devagar | `run.py` sequência de reversões A→B→C |
| rede recebe sinal de contexto (limitação declarada) | ablação `context_signal=False` |

### Como rodar
```bash
python -m papers.p20_astrocyte_context.run --seed 0 --write-results
```

### O que observar
- Nas primeiras exposições os métodos são iguais — não há nada
  consolidado ainda (honesto: o astrócito não é vantagem gratuita).
- Na re-exposição, a restauração gateada corta o erro de readaptação
  ~pela metade vs a rede sem astrócitos.
- A ablação **sem sinal de contexto** perde o benefício — o caveat
  central do paper: a rede recebe informação contextual; ela não descobre
  a mudança sozinha (ver p30 para a descoberta a partir de recompensas).

### Diferenças vs paper e limitações
- Implementamos o esquema de duas escalas com regra delta linear — não a
  rede contínua detalhada do paper nem sua análise de bifurcação.
- O "sinal de contexto" é um rótulo discreto perfeito; no paper ele chega
  por covariáveis fisiológicas.
- Números demonstram o mecanismo, não reproduzem os valores do paper.

## 🇺🇸 English

### Summary
A **nested-loops-over-separated-timescales** model: fast weights f learn
via the task delta rule; slower astrocytes a_c **consolidate** f while
context c is active; each trial the gated restoration
`f ← f + η_r(a_c − f)` reinstates that context's learned configuration.
The paper's **metaplasticity** model: glia regulate plasticity itself
per context. Evaluated on rule reversal (B = −A), the paper's paradigm.

### Core idea in plain words
Learning rule A then the inverse rule B makes a network fight itself.
Astrocytes act as a per-context filing system: they store "how I solved
this last time" and hand it back when the context returns — no
relearning from scratch.

### How to run
```bash
python -m papers.p20_astrocyte_context.run --seed 0 --write-results
```

### What to look at
- First exposures are identical across methods — nothing consolidated yet
  (honest: the astrocyte is no free advantage).
- On re-exposure, gated restoration halves the re-adaptation error vs the
  no-astrocyte network.
- The **no-context-signal** ablation loses the benefit — the paper's own
  caveat: the network receives contextual information rather than
  discovering the switch itself (see p30 for reward-driven discovery).

### Differences vs the paper and limitations
- We implement the two-timescale scheme with a linear delta rule — not
  the paper's detailed continuous network or its bifurcation analysis.
- The "context signal" is a perfect discrete label; in the paper it
  arrives through physiological covariates.
- Numbers demonstrate the mechanism, not the paper's values.
