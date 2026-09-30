# p28 — Astromorphic Self-Repair

Yi Han et al. — AAAI 2023 (DOI 10.1609/aaai.v37i6.25947; arXiv:2209.07428).

## 🇧🇷 Português

### Resumo
O paper mostra que mecanismos inspirados em astrócitos dão a redes não
supervisionadas a capacidade de **auto-reparo**: quando neurônios morrem
(injeção de falha), a glia — que acompanha o domínio de entrada de cada
unidade — realoca o domínio registrado para unidades reserva, preservando
a representação sem precisar re-aprender do zero.

### Ideia central em linguagem simples
Se um funcionário sai de férias para sempre, a glia é o caderno de
instruções que ele deixou: o substituto começa já sabendo qual era o
domínio do antecessor — em vez de aprender do zero.

### Algoritmo passo a passo
```
WTA-Hebbian: vencedor j aprende w_j += η(x − w_j)
glia registra domínio: m_j += ρ(x − m_j) quando j vence
falha: mata neurônios ativos
reparo: reserva s recebe W_s = m_morto + ruído e entra na competição
medir cobertura = cos do melhor neurônio vivo por cluster
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| glia monitora domínio dos neurônios | `method.py: SelfRepairNet.m` |
| realocação pós-falha | `method.py: inject_fault` (graft em reserva) |
| degradação graciosa vs falha | `run.py` cobertura pre/post + sweep n_kill |

### Como rodar
```bash
python -m papers.p28_astromorphic_repair.run --seed 0 --write-results
```

### O que observar
- Sem reparo, matar 3/6 neurônios derruba a cobertura a ~0.54 e a rede
  leva ~200 passos para se recuperar parcialmente.
- Com reparo glial, os reservas assumem o domínio e a cobertura fica
  ~0.95 — quase sem solavanco.

### Diferenças vs paper e limitações
- O paper usa camadas de "astromorphic neurons" em aprendizado não
  supervisionado com arquitetura própria; aqui é WTA-Hebbian + memória
  de domínio — mesmo mecanismo de realocação.
- A falha é morte instantânea de neurônios; o paper varia tipos de dano.
- Spares ilimitados por disponibilidade; em hardware seriam recursos
  fixos.

## 🇺🇸 English

### Summary
The paper shows astrocyte-inspired mechanisms give unsupervised
networks **self-repair**: when neurons die (fault injection), glia —
which tracks each unit's input domain — reallocates the recorded domain
to spare units, preserving the representation without relearning from
scratch.

### Core idea in plain words
If an employee quits for good, glia is the instruction manual they left
behind: the replacement starts already knowing the predecessor's
domain — instead of learning from zero.

### How to run
```bash
python -m papers.p28_astromorphic_repair.run --seed 0 --write-results
```

### What to look at
- Without repair, killing 3/6 neurons drops coverage to ~0.54 and the
  net needs ~200 steps to partially recover.
- With glial repair, spares take over the domain and coverage stays
  ~0.95 — barely a hiccup.

### Differences vs the paper and limitations
- The paper uses "astromorphic neuron" layers in unsupervised learning
  with its own architecture; here it's WTA-Hebbian + domain memory —
  same reallocation mechanism.
- The fault is instant neuron death; the paper varies damage types.
- Spares are bounded by availability; in hardware they'd be fixed
  resources.
