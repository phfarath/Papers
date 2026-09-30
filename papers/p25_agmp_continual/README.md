# p25 — Astrocyte-Gated Metaplasticity (AGMP)

Zhengshan Dong, Wude He — Frontiers in Neuroscience 19:1768235 (2026).
DOI: 10.3389/fnins.2025.1768235

## 🇧🇷 Português

### Resumo
O paper propõe que astrócitos implementam **metaplasticidade** — a
"plasticidade da plasticidade": cada sinapse tem seu nível de
plasticidade modulado pela glia conforme seu histórico de importância.
Sinapses críticas para conhecimento já adquirido tornam-se rígidas; as
restantes continuam plásticas. Isso ataca o dilema
estabilidade–plasticidade do aprendizado contínuo — e κ grande demais
trava a rede, mostrando o custo do excesso de consolidação.

### Ideia central em linguagem simples
Cada conexão carrega um "termômetro de importância" mantido pela glia:
quanto mais aquela sinapse contribuiu para reduzir erros, menos ela pode
mudar — como endurecer o verniz das engrenagens que já funcionam.

### Algoritmo passo a passo
```
para cada passo de treino na tarefa corrente:
    grad = ∂L/∂w
    A += |grad · w|           # astrócito integra contribuição sináptica
    w -= η · [1/(1+κA)] · grad   # gate de plasticidade
medir MSE de todas as tarefas após cada fase
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| astrócito regula plasticidade por sinapse | `method.py: MetaplasticLinear.A/gate` |
| metaplasticidade reduz esquecimento | `run.py` matriz fase×tarefa + forgetting |
| dilemma estabilidade–plasticidade | `run.py` sweep κ (κ=50 trava o novo) |

### Como rodar
```bash
python -m papers.p25_agmp_continual.run --seed 0 --write-results
```

### O que observar
- Sem gate: a T1 é praticamente apagada depois de aprender T2/T3.
- Com AGMP (κ=5): a T1 é parcialmente retida com custo pequeno na nova.
- κ=50: T1 muito retida, mas T3 quase não aprende — plasticidade morta:
  o mecanismo tem um ponto ótimo, não "quanto mais melhor".

### Diferenças vs paper e limitações
- Implementação mínima em regressor linear; o paper avalia redes
  profundas e benchmarks de CL (Split-MNIST e afins).
- A medida de importância é um integral tipo "synaptic intelligence" —
  proxy da consolidação glial, não o modelo biofísico do paper.
- Apenas 3 tarefas sequenciais; não há re-exposição nem replay.

## 🇺🇸 English

### Summary
The paper proposes astrocytes implement **metaplasticity** — the
"plasticity of plasticity": each synapse's learning rate is modulated by
glia according to its importance history. Synapses critical to acquired
knowledge stiffen; the rest stay plastic. This targets the
stability–plasticity dilemma of continual learning — and too-large κ
freezes the network, showing the cost of over-consolidation.

### Core idea in plain words
Each connection carries a glial "importance thermometer": the more a
synapse contributed to reducing errors, the less it may change — like
varnishing the gears that already work.

### How to run
```bash
python -m papers.p25_agmp_continual.run --seed 0 --write-results
```

### What to look at
- Without the gate, T1 is almost erased after learning T2/T3.
- With AGMP (κ=5), T1 is partly retained at a small cost to the new task.
- κ=50 retains T1 but barely learns T3 — dead plasticity: the mechanism
  has an optimum, not "more is better".

### Differences vs the paper and limitations
- Minimal implementation on a linear regressor; the paper evaluates
  deep networks and CL benchmarks (Split-MNIST-like).
- The importance measure is a synaptic-intelligence-style integral — a
  proxy for glial consolidation, not the paper's biophysical model.
- Only 3 sequential tasks; no re-exposure or replay.
