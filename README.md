# Papers — replicação de papers de memória de agentes

## Português

Replicações **offline e determinísticas** dos mecanismos de 31 papers sobre
memória para agentes LLM e computação neural inspirada em glia. Como não há chave de API, cada paper é
reimplementado com um **MockLLM** (respostas determinísticas por seed) e
ambientes de brinquedo — as implementações são fiéis aos *mecanismos*
(fórmulas, pipelines, estruturas), não aos números dos papers.

### A regra de honestidade

Um mock handler recebe **apenas o texto do prompt** (mais um `rng`) — nunca
estado do ambiente ou gabarito. Os resultados são gerados por
`run.py --write-results` com seed fixa; nada é escrito à mão. Para trocar
por um LLM real compatível com OpenAI:

```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=... PAPERS_LLM_MODEL=...
python -m papers.p10_memgpt.run --llm openai
```

### Quickstart

```bash
pip install -U "setuptools>=68" && pip install -e ".[dev]"
ruff check . && pytest -q
python -m papers.p02_reflexion.run        # qualquer paper
python scripts/run_all.py                 # regenera todos os RESULTS.md
```

### Layout

- `papers/common/` — LLM (mock/openai), embeddings, retrieval, MemorySystem,
  reader (qa.answer/judge), dataset conversacional, catálogo;
- `papers/envs/` — HouseholdEnv, CraftWorld, MiniWeb (determinísticos);
- `papers/pNN_*/` — um pacote por paper: `method.py`, `prompts.py`,
  `run.py`, `README.md` (PT/EN com tabela paper→código), `RESULTS.md`;
- `docs/ARQUITETURA_ARCHITECTURE.md` — arquitetura e como adicionar papers.

### Os 31 papers (ordem de leitura sugerida = lista do usuário)

| # | Paper | arXiv | Pasta | Ideia (PT/EN) | Env |
|---|-------|-------|-------|----------------|-----|
| 1 | CLIN | [2310.10134](https://arxiv.org/abs/2310.10134) | `p01_clin` | abstrações causais guiam a próxima tentativa / causal abstractions guide the next trial | Household |
| 2 | Reflexion | [2303.11366](https://arxiv.org/abs/2303.11366) | `p02_reflexion` | reflexões verbais episódicas / episodic verbal reflections | Household + code |
| 3 | MemRL | [2411.13537v2](https://arxiv.org/abs/2411.13537v2) | `p03_memrl` | tripletos com utilidade Q / utility-Q triplets | Household |
| 4 | MUSE | [2601.03192](https://arxiv.org/abs/2601.03192) | `p04_muse` | metacognição decide quando refletir / metacognition decides when to reflect | Household |
| 5 | CER | [2506.06698](https://arxiv.org/abs/2506.06698) | `p05_cer` | destilação de dinâmicas e skills / dynamics+skill distillation | MiniWeb |
| 6 | Voyager | [2305.16291](https://arxiv.org/abs/2305.16291) | `p06_voyager` | currículo + biblioteca de skills / curriculum + skill library | CraftWorld |
| 7 | MUSE-Autoskill | [2605.27366](https://arxiv.org/abs/2605.27366) | `p07_muse_autoskill` | ciclo de vida de skills / skill lifecycle | SkillsBench-lite |
| 8 | CoALA | [2309.02427](https://arxiv.org/abs/2309.02427) | `p08_coala` | framework de memórias/ações/decisão / memory/action/decision framework | Household |
| 9 | Generative Agents | [2304.03442](https://arxiv.org/abs/2304.03442) | `p09_generative_agents` | recência+importância+relevância, reflexão, plano / retrieval+reflection+planning | Smallville-lite |
| 10 | MemGPT | [2310.08560](https://arxiv.org/abs/2310.08560) | `p10_memgpt` | memória em camadas tipo SO / OS-style layered memory | conv |
| 11 | A-MEM | [2502.12110](https://arxiv.org/abs/2502.12110) | `p11_a_mem` | notas com links e evolução / linked evolving notes | conv |
| 12 | Zep | [2501.13956](https://arxiv.org/abs/2501.13956) | `p12_zep` | grafo bitemporal / bitemporal graph | conv |
| 13 | Mem0 | [2504.19413](https://arxiv.org/abs/2504.19413) | `p13_mem0` | extração→ops ADD/UPDATE/DELETE/NOOP | conv |
| 14 | Memory-R1 | [2508.19828](https://arxiv.org/abs/2508.19828) | `p14_memory_r1` | RL treina o gerenciador / RL trains the manager | conv |
| 15 | LongMemEval | [2410.10813](https://arxiv.org/abs/2410.10813) | `p15_longmemeval` | benchmark de chat (5 habilidades) / chat benchmark | histories |
| 16 | MemoryAgentBench | [2507.05257](https://arxiv.org/abs/2507.05257) | `p16_memoryagentbench` | ingestão incremental AR/TTL/LRU/SF | streams |
| 17 | LongMemEval-V2 | [2605.12493v1](https://arxiv.org/abs/2605.12493v1) | `p17_longmemeval_v2` | memória sobre trajetórias web + AgentRunbook | MiniWeb |
| 18 | Survey | [2512.13564](https://arxiv.org/abs/2512.13564) | `p18_survey` | taxonomia forms/functions/dynamics | repo |
| 19 | Neuron–Astrocyte Memory | [2311.08135](https://arxiv.org/abs/2311.08135) | `p19_astrocyte_assoc` | acoplamentos ordem-3 / order-3 couplings | numpy |
| 20 | Astrocyte Context Dynamics | [10.1371/journal.pcbi.1012186](https://doi.org/10.1371/journal.pcbi.1012186) | `p20_astrocyte_context` | contexto liga pesos em duas escalas / context binds weights two-timescale | numpy |
| 21 | Neuron–Astrocyte Transformer | [10.1073/pnas.2219150120](https://doi.org/10.1073/pnas.2219150120) | `p21_astrocyte_transformer` | ODE converge a softmax / ODE converges to softmax | numpy |
| 22 | Neuron–Glia Networks | [10.1371/journal.pone.0019109](https://doi.org/10.1371/journal.pone.0019109) | `p22_neuron_glia_nets` | glia detecta estagnação e injeta ruído / stall detection + noise | numpy |
| 23 | WM SNN + Astrocytes | [10.3389/fncel.2021.631485](https://doi.org/10.3389/fncel.2021.631485) | `p23_astrocyte_wm` | Ca²⁺ sustenta WM / Ca²⁺ sustains WM | numpy |
| 24 | Situation Memory | [10.1109/TNNLS.2023.3335450](https://doi.org/10.1109/TNNLS.2023.3335450) | `p24_situation_memory` | cue E contexto / cue AND context recall | numpy |
| 25 | AGMP Continual | [10.3389/fnins.2025.1768235](https://doi.org/10.3389/fnins.2025.1768235) | `p25_agmp_continual` | gate metaplástico / metaplastic gating | numpy |
| 26 | Dual-Timescale Nav | [2604.15391](https://arxiv.org/abs/2604.15391) | `p26_dual_memory_nav` | supressão de visitados / visited suppression | numpy |
| 27 | Emergent Attention | [2604.25481](https://arxiv.org/abs/2604.25481) | `p27_emergent_attention` | replicador → softmax / replicator → softmax | numpy |
| 28 | Astromorphic Self-Repair | [10.1609/aaai.v37i6.25947](https://doi.org/10.1609/aaai.v37i6.25947) | `p28_astromorphic_repair` | enxerto de domínio glial / glial domain graft | numpy |
| 29 | LSM + Astrocytes | [2503.06798](https://arxiv.org/abs/2503.06798) | `p29_lsm_astrocytes` | unidades lentas, razão em U / slow units, U-shaped ratio | numpy |
| 30 | Hybrid Automaton | [2609.16217](https://arxiv.org/abs/2609.16217) | `p30_hybrid_automaton` | evidência → troca de modo / evidence → mode switch | numpy |
| 31 | Futility Evidence | [10.1016/j.cell.2019.05.050](https://doi.org/10.1016/j.cell.2019.05.050) | `p31_futility_passivity` | acumula futilidade → passividade / futility → passivity | numpy |

### Hipótese de pesquisa (do usuário)

A **representação explícita de validade das experiências** — quando uma
experiência era verdadeira e quando deixou de ser — é uma questão aberta.
Hipótese (não um resultado): no código, isso já existe nos intervalos
`t_valid`/`t_invalid` do Zep (`p12_zep/method.py`) e poderia ser estendido
aos tripletos do MemRL (`p03_memrl/method.py`, um campo de validade por
triplet filtrando a recuperação). Mem0 demonstra o contraste didático:
UPDATE/DELETE perdem os valores antigos, enquanto Zep os mantém com
intervalo fechado — por isso Zep acerta "antes/anteriormente" e Mem0 não.

## English

**Offline, deterministic replications** of the mechanisms of 18
agent-memory papers. With no API key, each paper is reimplemented with a
**MockLLM** (seed-deterministic answers) and toy environments — faithful
to the *mechanisms* (formulas, pipelines, structures), not the papers'
numbers.

### Honesty rule

A mock handler receives **only the prompt text** (plus an `rng`) — never
environment state or gold answers. Results come from
`run.py --write-results` at a fixed seed; nothing is hand-written. Switch
to a real OpenAI-compatible LLM:

```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=... PAPERS_LLM_MODEL=...
python -m papers.p10_memgpt.run --llm openai
```

### Quickstart

```bash
pip install -U "setuptools>=68" && pip install -e ".[dev]"
ruff check . && pytest -q
python -m papers.p02_reflexion.run        # any paper
python scripts/run_all.py                 # regenerate all RESULTS.md
```

### Layout

- `papers/common/` — LLM (mock/openai), embeddings, retrieval,
  MemorySystem, reader (qa.answer/judge), shared conversational dataset,
  catalog;
- `papers/envs/` — HouseholdEnv, CraftWorld, MiniWeb (deterministic);
- `papers/pNN_*/` — one package per paper: `method.py`, `prompts.py`,
  `run.py`, bilingual `README.md` (paper→code mapping), `RESULTS.md`;
- `docs/ARQUITETURA_ARCHITECTURE.md` — architecture + how to add a paper.

### The 31 papers (suggested reading order)

See the PT table above (bilingual idea column); reading order: CLIN →
Reflexion → MemRL → MUSE → CER → Voyager → MUSE-Autoskill → CoALA →
Generative Agents → MemGPT → A-MEM → Zep → Mem0 → Memory-R1 →
LongMemEval → MemoryAgentBench → LongMemEval-V2 → Survey →
Neuron–Astrocyte Memory → Astrocyte Context Dynamics →
Neuron–Astrocyte Transformer → Neuron–Glia Networks → WM SNN +
Astrocytes → Situation Memory → AGMP Continual → Dual-Timescale Nav →
Emergent Attention → Astromorphic Self-Repair → LSM + Astrocytes →
Hybrid Automaton → Futility Evidence.

### Research hypothesis (user's)

The **explicit validity representation of experiences** — when an
experience was true and when it stopped being — is an open question. This
is a hypothesis, not a result: in code it already exists as Zep's
`t_valid`/`t_invalid` intervals (`p12_zep/method.py`) and could extend to
MemRL triplets (`p03_memrl/method.py`, a per-triplet validity field
filtering retrieval). Mem0 gives the didactic contrast: UPDATE/DELETE
lose old values while Zep keeps them with closed intervals — hence Zep
answers "before/previously" correctly and Mem0 does not.
