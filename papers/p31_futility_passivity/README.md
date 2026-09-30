# p31 — Glia Accumulate Evidence that Actions Are Futile

Yu Mu et al. — Cell 2019 (DOI 10.1016/j.cell.2019.05.050).

## 🇧🇷 Português

### Resumo
**Paper biológico** (não-computacional): em larvas de zebrafish,
astrócitos radiais **acumulam evidência de que as ações são fúteis** —
sinais ligados a tentativas de natação sem o feedback esperado — e
participam da mudança para um estado de **passividade** (supressão
comportamental). É o fundamento experimental para a ideia de um
"acumulador de ineficácia" que precede a mudança de modo de agir.

Nossa implementação é um **modelo abstrato do achado**, não uma
replicação da biologia: um agente tenta agir; falhas sobem a evidência
glial E (sucesso zera); E acima do limiar suprime o comportamento por um
período.

### Ideia central em linguagem simples
A glia conta "quantas vezes você bateu na mesma porta sem ela abrir":
passado um ponto, ela diz "pare de gastar energia" — e só tenta de novo
depois de descansar.

### Algoritmo passo a passo
```
a cada período:
    se passivo: descansa; E decai rápido
    senão: tenta (sucesso com p)
        sucesso -> E = 0
        falha   -> E += α − vazamento
        E > θ   -> passivo por `cool` períodos
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| evidência glial de futilidade | `method.py: FutilityAgent.E` |
| supressão ao cruzar limiar | `method.py: passive` (cool-off) |
| eficácia restaurada → retomada | `run.py` regime "phases" |

### Como rodar
```bash
python -m papers.p31_futility_passivity.run --seed 0 --write-results
```

### O que observar
- Regime fútil: o agente sem glia desperdiça as 400 tentativas; com
  glia ele suprime após acumular evidência suficiente.
- Regime viável (p=0.3): mantém a maioria dos acertos — sequências
  longas de falhas disparam passividade ocasional (falso positivo).
- Regime fases: após o descanso, o agente retoma na janela viável —
  a passividade é reversível, como no animal.

### Diferenças vs paper e limitações
- Este é um modelo **conceitual**: o paper mede atividade glial e faz
  intervenções celulares em zebrafish; aqui só testamos a dinâmica
  funcional (acumular → limiar → supressão → retomada).
- A "passividade" é um cool-off determinístico; no animal é um estado
  neural com recuperação própria.
- Não há aprendizado — só a lógica de evidência, que é a contribuição
  do achado para a linha de glia artificial.

## 🇺🇸 English

### Summary
**Biological paper** (non-computational): in zebrafish larvae, radial
astrocytes **accumulate evidence that actions are futile** — signals
linked to swimming attempts without the expected feedback — and take
part in switching to a **passivity** state (behavioral suppression). It
is the experimental foundation for the idea of a "futility accumulator"
preceding a change of behavioral mode.

Our implementation is an **abstract model of the finding**, not a
biological replication: an agent acts; failures raise glial evidence E
(success resets); E past the threshold suppresses behavior for a period.

### Core idea in plain words
Glia counts "how many times you knocked on the door without it opening":
past a point, it says "stop wasting energy" — and only tries again
after resting.

### How to run
```bash
python -m papers.p31_futility_passivity.run --seed 0 --write-results
```

### What to look at
- Futile regime: the no-glia agent wastes all 400 attempts; with glia
  it suppresses after enough accumulated evidence.
- Viable regime (p=0.3): keeps most wins — long failure streaks
  trigger occasional passivity (false positive).
- Phases regime: after resting, the agent resumes in the viable
  window — passivity is reversible, as in the animal.

### Differences vs the paper and limitations
- This is a **conceptual** model: the paper measures glial activity and
  runs cellular interventions in zebrafish; here we only test the
  functional dynamics (accumulate → threshold → suppression → resume).
- "Passivity" is a deterministic cool-off; in the animal it is a neural
  state with its own recovery.
- No learning — only the evidence logic, which is the finding's
  contribution to the artificial-glia line.
