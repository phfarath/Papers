# p02 — Reflexion: Language Agents with Verbal Reinforcement Learning

arXiv: https://arxiv.org/abs/2303.11366 — Shinn et al., 2023.

## 🇧🇷 Português

### Resumo
O Reflexion transforma o feedback binário do ambiente em **feedback verbal**:
após cada tentativa, um LLM escreve uma *reflexão* sobre o que deu errado, que é
guardada numa memória de longo prazo limitada (Ω = 1–3 reflexões) e reutilizada
nas próximas tentativas — sem treinar peso nenhum.

### Ideia central em linguagem simples
É como um estudante que, ao errar uma prova, escreve num caderninho "da próxima
vez, lembre: o fogão estava quebrado, use o micro-ondas". Na prova seguinte ele
lê o caderninho antes de responder. O "caderninho" é a memória; a anotação é a
reflexão.

### Algoritmo passo a passo (Algoritmo 1 do paper)
```
Inicialize Actor (M_a), Evaluator (M_e), Self-Reflection (M_sr); mem ← []
repita até o Evaluator aprovar OU esgotar max_trials:
    1. τ_t ← trajetória do Actor (ReAct) no ambiente
    2. r_t ← M_e(τ_t)            # recompensa do env + heurística
    3. sr_t ← M_sr(τ_t, r_t)     # reflexão verbal
    4. mem ← mem + [sr_t]        # limitado a Ω reflexões
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| §3 Actor (ReAct) | `method.py: ReflexionAgent.act` + handler `household.act` (`envs/household.py`) |
| §3 Evaluator (heurística ALFWorld) | `method.py: ReflexionAgent.evaluate` |
| §3 Self-Reflection / mem | `prompts.py: _reflect` (`reflexion.reflect`), `mem` limitado a Ω em `run_task` |
| §4.1 ALFWorld | `envs/household.py` (tarefas + armadilhas) |
| §4.3 Programação + testes auto-gerados | `method.py: run_tests`, `derive_internal_tests`, `reflexion_code_loop`; `prompts.py: CODE_PROBLEMS`, handler `code.tests` (testes internos derivados SÓ dos exemplos da docstring) + testes escondidos (pass@1 real) |

### Como rodar
```bash
python -m papers.p02_reflexion.run --seed 0 --llm mock --write-results
```

### O que observar nos resultados
- **Household**: a coluna `post-1` (sucesso nas tentativas 2+, após a primeira
  reflexão) deve superar o baseline — é onde a memória verbal age.
- **Programação**: o baseline repete o mesmo bug; o Reflexion corrige após a
  reflexão citar o teste que falhou. Pass@1 é medido em testes ESCONDIDOS —
  as linhas com `internal→hidden FP` mostram os falsos positivos dos testes
  auto-gerados (bug passa no interno, falha em edge case), como no paper.
- `llm.calls` conta os prompts — o custo do Reflexion é ~uma reflexão por trial.

### Diferenças vs paper e limitações
- O "LLM" é um MockLLM determinístico: as reflexões são geradas por regras sobre
  o TEXTO da trajetória (portas fechadas, dispositivos quebrados), não por um
  modelo real. O mecanismo (loop, Ω, tipos de memória) é fiel.
- O evaluator interno de código usa testes derivados dos exemplos da docstring
  (handler `code.tests`), como os testes auto-gerados do paper; o sucesso real é
  medido em testes escondidos separados. O mock "gera" código consultando um
  mini-corpus embutido.
- Números do RESULTS.md são do brinquedo, não do paper (paper: +22% no ALFWorld).

## 🇺🇸 English

### Summary
Reflexion turns binary environment feedback into **verbal feedback**: after each
trial an LLM writes a *reflection* on what went wrong, stored in a bounded
long-term memory (Ω = 1–3 reflections) and reused in later trials — no weight
training at all.

### Core idea in plain words
Like a student who, after failing a test, writes in a notebook "next time,
remember: the stove was broken, use the microwave". Before the next attempt they
re-read the notebook. The notebook is memory; the note is the reflection.

### Step-by-step algorithm (paper's Algorithm 1)
See the pseudocode above (same algorithm).

### Paper → code mapping, How to run, What to look at
Same tables/commands as in the Portuguese section.

### Differences vs paper & limitations
- The "LLM" is a deterministic MockLLM: reflections are produced by rules over
  the trajectory TEXT, not a real model. The mechanism (loop, Ω, memory types)
  is faithful.
- The internal code evaluator uses fixed per-task tests; the mock "writes" code
  from a small built-in corpus.
- RESULTS.md numbers come from the toy setup, not the paper (paper: +22% on
  ALFWorld).

### Referência / Reference
Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement
Learning." NeurIPS 2023. https://arxiv.org/abs/2303.11366
