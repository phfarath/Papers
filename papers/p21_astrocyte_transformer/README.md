# p21 — Building Transformers from Neurons and Astrocytes

Leo Kozachkov, Ksenia V. Kastanenka, Dmitry Krotov — PNAS 2023.
DOI: 10.1073/pnas.2219150120 · bioRxiv 2022.10.12.511910

## 🇧🇷 Português

### Resumo
O paper propõe que a operação central de um Transformer —
`softmax(QKᵀ/√d)·V` — pode ser realizada por uma dinâmica
neurônio–astrócito em duas escalas temporais. Neurônios rápidos
carregam os scores `s_i = q·k_i/√d`; um astrócito lento **integra a
atividade neural total** (pooling dos processos astrogliais) e devolve
normalização divisiva `y_i = exp(n_i)/A`. No ponto fixo, `A = Σexp(s)` —
o astrócito é o **denominador da softmax** e a saída é exatamente a
atenção.

### Ideia central em linguagem simples
Numa votação por pesos exponenciais, alguém precisa somar todos os votos
para normalizar. Neurônios rápidos só conhecem o próprio voto; o
astrócito, lento e conectado a todos, faz essa soma — e aí a rede
inteira "descobre" que estava calculando atenção.

### Algoritmo passo a passo
```
s_i ← (q·k_i)/√d para cada key i
n_i ← 0, A ← 0
repete até convergir:
    dn_i/dt = (−n_i + s_i)/τ_n        # neurônios carregam scores
    dA/dt   = (−A + Σ_j e^{n_j})/τ_a  # astrócito integra a soma (lento)
y = (e^{n}/A)·V                       # = softmax(s)·V no ponto fixo
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| neurônios × astrócitos realizam atenção | `method.py: NeuronAstrocyteAttention` |
| pooling astroglial = denominador softmax | `method.py: run` (termo `−A + Σe^{n}`) |
| duas escalas temporais neurônio/glia | `tau_n`, `tau_a` (tabela de convergência) |
| não é "Transformer no cérebro" — é correspondência | ablação `astrocyte=False` e RESULTS |

### Como rodar
```bash
python -m papers.p21_astrocyte_transformer.run --seed 0 --write-results
```

### O que observar
- Erro final ~1e−9: a dinâmica converge para a atenção softmax exata.
- τ_a maior só estica o transiente — a escala temporal glial dita o tempo
  de convergência, não o resultado.
- Sem astrócito (A≡1) sai atenção linear não normalizada — o erro é
  enorme: a normalização é exatamente o papel proposto para a glia.

### Diferenças vs paper e limitações
- É uma **correspondência computacional** — o paper não afirma que o
  cérebro "roda" Transformers nem que adicionar astrócitos a uma LLM
  melhore desempenho; nós também não.
- Implementamos o esquema mínimo de normalização divisiva; o paper usa
  construções com populações e não-linearidades específicas.
- q,k,v aleatórios — não treinamos nem avaliamos tarefas de linguagem.

## 🇺🇸 English

### Summary
The paper proposes that a Transformer's core operation —
`softmax(QKᵀ/√d)·V` — can be performed by a two-timescale
neuron–astrocyte dynamics. Fast neurons carry scores
`s_i = q·k_i/√d`; a slow astrocyte **integrates total neural activity**
(astrocytic-process pooling) and feeds back divisive normalization
`y_i = exp(n_i)/A`. At the fixed point `A = Σexp(s)` — the astrocyte is
the **softmax denominator** and the output is exactly attention.

### Core idea in plain words
In exponential-weighted voting, someone must sum all votes to normalize.
Fast neurons only know their own vote; the slow all-connected astrocyte
computes the sum — and the whole network turns out to be computing
attention.

### How to run
```bash
python -m papers.p21_astrocyte_transformer.run --seed 0 --write-results
```

### What to look at
- Final error ~1e−9: the dynamics converge to exact softmax attention.
- Larger τ_a only stretches the transient — the glial timescale sets
  convergence time, not the outcome.
- Ablated astrocyte (A≡1) yields unnormalized linear attention — huge
  error: normalization is exactly the proposed glial role.

### Differences vs the paper and limitations
- This is a **computational correspondence** — the paper does not claim
  brains run Transformers or that adding astrocytes improves an LLM;
  neither do we.
- We implement the minimal divisive-normalization scheme; the paper uses
  specific populations and nonlinearities.
- Random q,k,v — no language task is trained or evaluated.
