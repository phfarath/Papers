# p30 — Neural-Astrocyte Hybrid Automaton

Giacomo Vedovati, Ilya E. Monosov, Thomas J. Papouin, ShiNung Ching —
preprint, setembro 2026.

## 🇧🇷 Português

### Resumo
O paper propõe uma arquitetura neurônio–astrócito que funciona como
**autômato híbrido**: modos discretos de operação (memórias por contexto
latente) + um acumulador contínuo glial de evidência. A dinâmica
astroglial integra sinais de que a regra corrente deixou de valer —
recompensa abaixo do esperado — e dispara a transição de modo, sem
precisar re-aprender (cada modo guarda sua memória de valores).

### Ideia central em linguagem simples
Trocar de estratégia não deveria exigir re-aprender tudo: a glia é o
contador de "quantas vezes o plano falhou"; estourando o limite, você
muda de modo — e cada modo já tinha seu próprio caderno de notas.

### Algoritmo passo a passo
```
modos discretos m ∈ {0,1}, cada um com Q-table própria
a cada tentativa:
    agir pelo Q do modo; atualizar Q do modo
    surpresa = max(0, max Q_modo − r)   # regra quebrada?
    A += taxa·surpresa − vazamento
    se A > θ: troca de modo; A = 0
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| estados discretos + variável contínua | `method.py: HybridAutomaton` (mode + A) |
| evidência acumulada de falha de regra | `method.py: accumulate` |
| transição de modo pelo limiar glial | `accumulate → mode flip` |
| memória preservada por contexto | `Q[modo]` separadas |

### Como rodar
```bash
python -m papers.p30_hybrid_automaton.run --seed 0 --write-results
```

### O que observar
- Com aprendizado lento (η=0.05), o bandit de regra latente é intratável
  para um Q único; o autômato híbrido detecta a quebra em ~5 tentativas
  e reusa a memória do outro contexto — recompensa ~35% maior.
- Com η rápido a baseline rastreia bem e o ganho some — a glia vale
  quando inferir é mais rápido que reaprender.
- Algumas trocas falsas são o preço da sensibilidade (leak/θ regulam).

### Diferenças vs paper e limitações
- O paper treina a arquitetura completa numa tarefa hierárquica; aqui a
  estrutura híbrida (modos + acumulador) é o foco, em bandit com regra
  latente.
- O acumulador usa surpresa de recompensa — proxy da evidência do paper.
- Apenas 2 modos e comutação conhecida simétrica; o paper trata
  transições entre contextos mais complexos.

## 🇺🇸 English

### Summary
The paper proposes a neural-astrocyte architecture acting as a
**hybrid automaton**: discrete operation modes (per-latent-context
memories) + a continuous glial evidence accumulator. The astrocyte
dynamics integrate signals that the current rule stopped working —
reward below expectation — and trigger the mode transition without
relearning (each mode keeps its own value memory).

### Core idea in plain words
Switching strategies shouldn't require relearning everything: glia is
the "how many times has the plan failed" counter; past the threshold
you change modes — and each mode already had its own notebook.

### How to run
```bash
python -m papers.p30_hybrid_automaton.run --seed 0 --write-results
```

### What to look at
- With slow learning (η=0.05) a single-Q agent cannot track the latent
  rule flips; the hybrid detects the break in ~5 trials and reuses the
  other context's memory — ~35% more reward.
- With fast η the baseline tracks well and the gain vanishes — glia
  matters when inference must be faster than relearning.
- Some false switches are the price of sensitivity (leak/θ tune it).

### Differences vs the paper and limitations
- The paper trains the full architecture on a hierarchical task; here
  the hybrid structure (modes + accumulator) is the focus, on a
  latent-rule bandit.
- The accumulator uses reward surprise — a proxy for the paper's
  evidence signal.
- Only 2 modes and symmetric switching; the paper treats transitions
  between richer contexts.
