# p01 — CLIN: A Continually Learning Language Agent

arXiv: https://arxiv.org/abs/2310.10134 — Majumdar et al., 2023.

## 🇧🇷 Português

### Resumo
O CLIN aprende **continuamente** entre tentativas: após cada trial, um *memory
generator* reflete sobre a trajetória e reescreve uma memória de **abstrações
causais** no formato "X [should/may] be necessary to Y" e "X does not
contribute to Y". `should` marca baixa incerteza, `may` alta. Para generalizar,
uma *meta-memória* é gerada a partir das melhores memórias por episódio
(auto-curriculum).

### Ideia central em linguagem simples
É um caderno de receitas que o cozinheiro atualiza após cada prato: "abrir o
armário *pode* ser necessário para achar a chaleira", "o fogão *não contribui*
para esquentar (está quebrado)". Na cozinha nova ele carrega só as receitas
genéricas — a meta-memória.

### Algoritmo passo a passo
```
para cada trial k de um episódio:
    repita até done ou max_steps:
        g_t ← Controller(m, e, T_<t, S_{k-1})   # próximo sub-goal
        a_t ← Executor(g_t, ações válidas)      # goal → ação
        o_t ← env.step(a_t)
    S_{k+1} ← MemGen(τ_k, r_k, {S_{k-2}, S_{k-1}, S_k})
para generalizar (Gen-Env / Gen-Task):
    S_meta ← MetaGen(memórias das melhores trials por episódio, tarefa-alvo)
```

### Mapeamento paper → código
| Paper | Código |
|---|---|
| §3 controller / executor | `method.py: ClinAgent._goal` / `._act` (handlers `clin.goal`, `clin.act`) |
| §3.2 memory generator + formato "necessary"/"does not contribute", should/may | `prompts.py: _memgen` (`clin.memgen`) |
| §3.2 input = última trial + r_k + 3 memórias | `method.py: update_memory` |
| §3.3 meta-memória + auto-curriculum (arquivo = 10, melhor trial/episódio) | `method.py: build_meta_memory`, `archive` |
| §4 settings Adapt / Gen-Env / Gen-Task / ablação | `run.py` |
| Nota: "causal" ≠ causalidade provada | docstring de `prompts.py` |

### Como rodar
```bash
python -m papers.p01_clin.run --seed 0 --llm mock --write-results
```

### O que observar
- **Adapt**: `post-1` do CLIN deve superar o baseline sem memória.
- **Gen-Env / Gen-Task**: meta-memória carrega só conselhos "de ambiente"
  (portas, containers, dispositivos) — generalização fraca mas real no mock.
- **Ablação**: abstrações estruturadas vs conselho livre.

### Diferenças vs paper e limitações
- MockLLM: memórias são extraídas por regras do TEXTO da trajetória — nada de
  raciocínio causal real. "Causal" é só o formato linguístico, como no paper.
- O ScienceWorld real é muito mais rico; usamos o HouseholdEnv de 5 cômodos.

## 🇺🇸 English

### Summary
CLIN learns **continually** across trials: after each trial a *memory
generator* rewrites a memory of **causal abstractions** in the form "X
[should/may] be necessary to Y" / "X does not contribute to Y". `should` marks
low uncertainty, `may` high. For generalization, a *meta-memory* is generated
from the best per-episode memories (auto-curriculum).

### Core idea, algorithm, mapping, how to run
See the Portuguese section (same content in English there for the algorithm
and mapping tables).

### Differences vs paper & limitations
- MockLLM extracts memories by rules over the trajectory TEXT — no real causal
  reasoning. "Causal" is only the linguistic format, as in the paper.
- Our HouseholdEnv (5 rooms) stands in for ScienceWorld.

### Referência / Reference
Majumdar, B.P., et al. "CLIN: A Continually Learning Language Agent for Rapid
Task Adaptation and Generalization." 2023. https://arxiv.org/abs/2310.10134
