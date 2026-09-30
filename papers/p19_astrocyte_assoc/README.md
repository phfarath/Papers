# p19 — Neuron-Astrocyte Associative Memory

Leo Kozachkov, Jean-Jacques Slotine, Dmitry Krotov — PNAS 2025 (preprint 2023).
Paper: https://arxiv.org/abs/2311.08135 · DOI: 10.1073/pnas.2417788122

## 🇧🇷 Português

### Resumo
Rede **tripartite** neurônio–sinapse–astrócito para memória associativa.
Como cada astrócito contacta muitas sinapses, o caminho
sinapse→astrócito→sinapse produz **interações efetivas de ordem superior**
às de um Hopfield clássico: a energia E = −Σ_μ F(m_μ) com F = m^{K+1} dá
a família Dense Associative Memory, e quanto maior o grau de acoplamento
astroglial (K), maior a capacidade de memória — o paper demonstra leis de
escala **supralineares** em N, superando implementações biológicas
conhecidas do DenseAM. A recuperação suave com ganhos softmax(β·m) é o
caso limite que se conecta à atenção de Transformers.

### Ideia central em linguagem simples
Hopfield guarda padrões só nas sinapses entre neurônios — como uma agenda
com N contatos por pessoa. Astrócitos agrupam várias sinapses e devolvem
um sinal conjunto (não-linear), como se grupos de contatos votassem juntos:
o mesmo cérebro comporta bem mais lembranças recuperáveis de uma pista
incompleta.

### Algoritmo passo a passo
```
armazena P padrões ξ^μ ∈ {±1}^N
cue x: 30% dos bits invertidos
repete ≤30×:                                  # passo síncrono
    m_μ ← (ξ^μ·x)/N                           # overlaps
    g_μ ← sign(m_μ)|m_μ|^K                    # feedback astroglial (K=1: Hopfield)
    x   ← sign(Σ_μ ξ_i^μ g_μ)                 # campo local
sucesso se |cos(x, ξ_alvo)| ≥ 0.8
modo suave: g_μ ← softmax(β·m_μ)              # limite "Transformer"
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| modelo NAM neurônio–sinapse–astrócito | `method.py: NeuronAstrocyteMemory` |
| energia E = −Σ F(m), família Hopfield→DenseAM→Transformer | `method.py: gains` (`order`=K, `beta`) |
| capacidade supralinear vs Hopfield | `run.py` tabelas load e scaling |
| ponte com atenção (softmax sobre similaridades) | `method.py: beta`, `run.py` tabela β |

### Como rodar
```bash
python -m papers.p19_astrocyte_assoc.run --seed 0 --write-results
```

### O que observar
- Hopfield (K=1) colapsa acima de ~0.2·N padrões; com astrócitos (K≥2) a
  recuperação se mantém em cargas muito maiores, e a vantagem cresce com N.
- No modo suave, β pequeno devolve uma mistura difusa dos padrões (atenção
  "mole"); β grande converge para winner-take-all — a ponte com softmax
  attention.
- A capacidade superior depende da ordem K — não é "memória extra grátis":
  é a mesma rede, mas com interações mais ricas mediadas pela glia.

### Diferenças vs paper e limitações
- Implementamos a **família de dinâmicas** (campo local ∝ g_μ = f'(m_μ)),
  não o modelo compartimental completo de processos astrogliais do paper;
  a correspondência K↔nº de sinapses por astrócito é qualitativa.
- Padrões aleatórios i.i.d. — o paper também analisa padrões estruturados.
- Números não comparáveis ao paper: demonstram o mecanismo, não reproduzem
  suas constantes/capacidades exatas.

## 🇺🇸 English

### Summary
A **tripartite** neuron–synapse–astrocyte network for associative memory.
Because each astrocyte contacts many synapses, the
synapse→astrocyte→synapse path yields **higher-order effective
interactions** than a classic Hopfield net: energy E = −Σ_μ F(m_μ) with
F = m^{K+1} spans the Dense Associative Memory family, and larger
astrocytic coupling (K) gives larger capacity — the paper proves
**supralinear** scaling laws in N, beating known biological DenseAM
implementations. Soft readout with softmax(β·m) gains is the limiting
case connecting to Transformer attention.

### Core idea in plain words
Hopfield stores patterns only in neuron–neuron synapses — an address book
with N contacts per person. Astrocytes pool many synapses and return a
joint nonlinear signal — groups of contacts voting together — so the same
brain holds far more memories retrievable from a partial cue.

### How to run
```bash
python -m papers.p19_astrocyte_assoc.run --seed 0 --write-results
```

### What to look at
- Hopfield collapses beyond ~0.2·N; astrocyte models keep recalling at
  several-fold higher load, and the gap widens as N grows.
- Small β returns a diffuse mixture (soft attention); large β converges
  to winner-take-all — the bridge to softmax attention.

### Differences vs the paper and limitations
- We implement the **dynamics family** (local field ∝ g_μ = f'(m_μ)), not
  the paper's full compartmental astrocyte-process model; the
  K↔synapses-per-astrocyte mapping is qualitative.
- i.i.d. random patterns — the paper also analyzes structured patterns.
- Numbers are not comparable to the paper: they demonstrate the
  mechanism, not its exact constants/capacities.
