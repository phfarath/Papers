# p22 — Artificial Neuron–Glia Networks

Alberto Porto-Pazos, Nati Varela, Alejandro Pazos — PLOS ONE 2011.
DOI: 10.1371/journal.pone.0019109

## 🇧🇷 Português

### Resumo
O paper introduz as **Artificial Neuron–Glia Networks (ANGN)**: MLPs
clássicas em que cada neurônio tem astrócitos associados que monitoram a
atividade sináptica e podem modular os pesos — incluindo o mecanismo de
**"postsynaptic noise"**: quando um neurônio estagna (saturado, com
gradiente ~0 e erro ainda alto), o astrócito injeta ruído nos seus pesos
aferentes, ajudando a rede a sair de ótimos locais. O paper reportou que
as ANGN resolvem tarefas onde a NN pura falha, com ganho dependente da
dificuldade.

### Ideia central em linguagem simples
Quando um neurônio "trava" num platô do gradiente, o astrócito percebe
que ele parou de aprender e dá um "empurrão" aleatório nos seus pesos —
como sacudir uma peça encaixada no lugar errado.

### Algoritmo passo a passo
```
para cada época de gradiente:
    forward + backward da MLP sigmoide
    para cada unidade oculta j (astrócito j):
        se |∂E/∂w_[:,j]| médio < ε: stall_j += 1  senão stall_j = 0
        se stall_j > limiar e erro alto:
            w_[:,j] += N(0, mag)   # postsynaptic noise
            stall_j = 0
    atualiza pesos com o gradiente
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| astrócito monitora a sinapse | `method.py: NeuronGliaNet.stall` |
| postsynaptic noise | `method.py: step` (kick `N(0, kick_mag)`) |
| ANGN vs NN em tarefas booleanas | `run.py: TASKS` (xor, parity3, ...) |
| ganho dependente da dificuldade | sucesso ↑ em xor nh=2; ~neutro em tarefas fáceis |

### Como rodar
```bash
python -m papers.p22_neuron_glia_nets.run --seed 0 --write-results
```

### O que observar
- Em XOR com só 2 unidades ocultas, a NN pura estagna em ~4/10 dos seeds;
  a ANGN resgata parte deles via kicks do astrócito (sucesso ↑ ~13pp).
- Em tarefas fáceis o efeito é quase neutro — o paper também reporta
  benefício não uniforme (maior quanto mais difícil o problema).
- `mean kicks` mostra a frequência com que o mecanismo dispara.

### Diferenças vs paper e limitações
- O paper testa vários algoritmos de treino e arquiteturas; usamos uma
  MLP sigmoide com GD full-batch — o mecanismo do astrócito é o mesmo.
- Tarefas booleanas com todas as amostras (sem held-out); o objetivo é
  mostrar o escape de platôs, não generalização.
- No paper o astrócito pode potenciar E deprimir sinapses; implementamos
  só o kick de ruído por estagnação — a assinatura central do resultado.

## 🇺🇸 English

### Summary
The paper introduces **Artificial Neuron–Glia Networks (ANGN)**: classic
MLPs where each neuron has associated astrocytes monitoring synaptic
activity and able to modulate weights — including the **postsynaptic
noise** mechanism: when a neuron stalls (saturated, ~0 gradient while
error is still high), its astrocyte injects noise into the incoming
weights, helping the network escape local optima. The paper reports NGNs
solving tasks where plain NNs fail, with task-dependent gains.

### Core idea in plain words
When a neuron gets stuck on a gradient plateau, the astrocyte notices it
stopped learning and gives its weights a random "push" — like shaking a
piece stuck in the wrong slot.

### How to run
```bash
python -m papers.p22_neuron_glia_nets.run --seed 0 --write-results
```

### What to look at
- On XOR with only 2 hidden units the plain NN stalls in ~4/10 seeds;
  the NGN rescues some via astrocyte kicks (success ↑ ~13pp).
- On easy tasks the effect is nearly neutral — matching the paper's
  non-uniform benefit (larger on harder problems).

### Differences vs the paper and limitations
- The paper tests several training algorithms and architectures; we use
  one sigmoid MLP with full-batch GD — the astrocyte mechanism is the same.
- Full boolean truth tables (no held-out set); the goal is showing the
  plateau escape, not generalization.
- Their astrocytes can both potentiate and depress synapses; we
  implement only the stagnation noise kick — the central signature.
