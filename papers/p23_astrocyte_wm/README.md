# p23 — Working Memory: Spiking Network + Astrocytes

Susanna Gordleeva et al. — Frontiers in Cellular Neuroscience 2021.
DOI: 10.3389/fncel.2021.631485

## 🇧🇷 Português

### Resumo
O paper mostra, numa rede neural de disparos com astrócitos, que a
dinâmica lenta do Ca²⁺ astroglial **prolonga a memória de trabalho** e a
torna robusta a distratores: sinapses entre neurônios persistentemente
co-ativos são potencializadas pelos astrócitos (a "sinapse tripartida"),
sustentando o padrão memorizado além do estímulo — enquanto um distrator
breve não atinge o limiar de Ca²⁺ e não ganha persistência.

### Ideia central em linguagem simples
O astrócito funciona como um "temporizador de relevância": só o que fica
ativo tempo suficiente para encher o reservatório de cálcio ganha
reforço. Um flash rápido (distrator) passa — um padrão sustentado
(item memorizado) fica.

### Algoritmo passo a passo
```
rede recorrente com inibição global; assembly = item de WM
cue no assembly (sustentado) → delay → distrator breve → leitura
a cada passo:
    r = max(0, tanh(u)); u integra W_eff·r + entrada − inibição + ruído
    C += (−C + r)/τ_C                    # traço lento tipo Ca²⁺
    g = max(0, C − C_thr)
    W_eff = W·(1 + α·min(g_i, g_j))      # sinapse tripartida
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| Ca²⁺ astroglial lento | `method.py: WMNetwork.C` (τ_C = 1s) |
| potenciação de sinapses co-ativas | `method.py: step` (min(g_i,g_j)) |
| WM robusta a distrator | `run.py` protocolo cue→distractor→readout |
| ganho astroglial controla persistência | `run.py` sweep α |

### Como rodar
```bash
python -m papers.p23_astrocyte_wm.run --seed 0 --write-results
```

### O que observar
- Controle (sem glia): o item decai ao nível do distrator — a
  recorrência sub-crítica não sustenta o padrão.
- Com astrócitos: o item persiste saturado e o distrator é suprimido —
  `held` (item>2·distr) vai a ~1.0 nos seeds.
- O sweep α mostra a persistência emergindo conforme o ganho glial cresce.

### Diferenças vs paper e limitações
- O paper usa neurônios de disparo Izhikevich e modelo biofísico de Ca²⁺;
  usamos dinâmica rate-based (SNN-lite) com traço lento — mesma
  assinatura funcional.
- Apenas um item e um distrator; o paper varia cargas e protocolos.
- Potenciação instantânea em mínimo bilateral — uma simplificação do
  mecanismo de gliotransmissão.

## 🇺🇸 English

### Summary
The paper shows, in a spiking neural network with astrocytes, that slow
astrocytic Ca²⁺ dynamics **extends working memory** and makes it robust
to distractors: synapses between persistently co-active neurons are
potentiated by astrocytes (the tripartite synapse), sustaining the
memorized pattern beyond the stimulus — while a brief distractor never
reaches the Ca²⁺ threshold and gains no persistence.

### Core idea in plain words
The astrocyte acts as a "relevance timer": only what stays active long
enough to fill the calcium reservoir gets reinforced. A quick flash
(distractor) passes — a sustained pattern (memorized item) stays.

### How to run
```bash
python -m papers.p23_astrocyte_wm.run --seed 0 --write-results
```

### What to look at
- Control (no glia): the item decays to distractor level — sub-critical
  recurrence cannot sustain the pattern.
- With astrocytes: the item persists saturated and the distractor is
  suppressed — `held` (item>2·distr) reaches ~1.0 across seeds.
- The α sweep shows persistence emerging with glial gain.

### Differences vs the paper and limitations
- The paper uses Izhikevich spiking neurons and a biophysical Ca²⁺ model;
  we use rate-based dynamics (SNN-lite) with a slow trace — same
  functional signature.
- Only one item and one distractor; the paper varies loads and protocols.
- Instantaneous bilateral-minimum potentiation — a simplification of the
  gliotransmission mechanism.
