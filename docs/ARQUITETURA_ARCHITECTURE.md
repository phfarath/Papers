# Arquitetura / Architecture

## 🇧🇷 Português

### Por que um MockLLM? (regra de honestidade)
Não há chave de API de LLM no ambiente. Então cada paper é reimplementado **fiel
ao mecanismo**, rodando 100% offline com um `MockLLM` determinístico
(`papers/common/llm.py`):

- Cada chamada de LLM carrega um marcador `[[task:nome]]` na 1ª linha do prompt
  (via `task_prompt`). O `MockLLM` roteia para um handler registrado por
  `@mock_handler("nome")`.
- Um handler recebe **apenas** `(prompt: str, rng: random.Random)`. Ele **não
  pode** ler estado do ambiente, gabarito ou variáveis globais do experimento —
  assim, qualquer efeito da memória nos resultados passa obrigatoriamente pelo
  TEXTO do prompt, exatamente como com um LLM real.
- Handlers imitam um "LLM razoável" com heurísticas legíveis (overlap de
  palavras, regex sobre o formato do prompt) e ruído controlado pelo rng —
  determinístico por seed.
- O "conhecimento" do mock morre DENTRO do handler (ex.: o mini-corpus de código
  do Reflexion), assim como um LLM real tem conhecimento próprio. Os resultados
  demonstram o mecanismo — **não reproduzem os números do paper**.

### Trocando para um LLM real
```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...            # e opcionalmente OPENAI_BASE_URL
export PAPERS_LLM=openai             # ou --llm openai
export PAPERS_LLM_MODEL=gpt-4o-mini  # default
# embeddings reais:
export PAPERS_EMBEDDER=openai        # default: hashing
```
Os prompts já são reais (instruções claras + formato de saída), então o mesmo
código funciona com GPT-compatíveis.

### Ambientes (papers/envs)
Todos seguem `reset() -> str`, `step(action) -> StepResult`,
`valid_actions()`, `task_description`, `score` (0..1), `max_steps`:
- `household.py` — ScienceWorld/ALFWorld-lite: 5 cômodos, portas/containers
  fechados, dispositivos quebrados, ~6 tipos de tarefa × 3 variantes, split
  in/out-of-distribution. O handler compartilhado `household.act` é a política
  usada por CLIN, Reflexion, MemRL e MUSE.
- `craft.py` — CraftWorld: árvore tecnológica Minecraft-lite + `run_skill_code`
  (exec em namespace restrito, com orçamento de operações).
- `web.py` — MiniWeb: shop/forum/gitlab-lite com busca, formulários, login e
  "pegadinhas" (admin não linkado, carrinho/issues mutáveis).

### MemorySystem (papers/common/memory_api.py)
Interface comum p/ benchmarks de memória (p15–p17):
```python
class MemorySystem(Protocol):
    name: str
    def reset(self) -> None: ...
    def add(self, item: MemoryItem) -> None: ...
    def retrieve(self, query, *, k=5, now=None) -> list[str]: ...
```
Baselines inclusos: `FullContextMemory` (janela truncada), `BM25Memory`,
`EmbeddingRAGMemory`.

### Como adicionar um paper (checklist)
1. `papers/pNN_nome/`: `__init__.py`, `README.md` (bilíngue, seções do spec),
   `method.py`, `prompts.py` (templates + `@mock_handler`), `run.py`
   (`python -m papers.pNN_nome.run --seed 0 --llm mock --write-results`).
2. Leia a seção de método do paper e cite seções/equações no mapeamento.
3. Registre handlers com nomes únicos (`pNN.algo`).
4. `run.py` deve imprimir a tabela e gerar `RESULTS.md` com seed fixa, offline
   < ~60 s.
5. `tests/test_pNN_nome.py`: invariantes do método + determinismo.
6. Adicione a pasta ao README raiz e ao `taxonomy.py` (p18).

---

## 🇺🇸 English

### Why a MockLLM? (honesty rule)
There is no LLM API key in the environment, so each paper is reimplemented
**faithful to the mechanism**, running 100% offline with a deterministic
`MockLLM` (`papers/common/llm.py`):

- Every LLM call carries a `[[task:name]]` marker on the prompt's first line
  (via `task_prompt`); `MockLLM` routes it to a handler registered with
  `@mock_handler("name")`.
- A handler receives **only** `(prompt: str, rng: random.Random)`. It must
  **never** read environment state, gold answers or experiment globals — so
  any effect of memory on results must flow through the prompt TEXT, exactly
  like a real LLM.
- Handlers imitate a "reasonable LLM" with readable heuristics (word overlap,
  regex over the prompt format) plus rng-controlled noise — deterministic per
  seed.
- The mock's "knowledge" lives INSIDE the handler (e.g., Reflexion's mini code
  corpus), like a real LLM's own knowledge. Results demonstrate the mechanism —
  they do **not** reproduce the papers' numbers.

### Switching to a real LLM
```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...            # optionally OPENAI_BASE_URL
export PAPERS_LLM=openai             # or --llm openai
export PAPERS_LLM_MODEL=gpt-4o-mini  # default
export PAPERS_EMBEDDER=openai        # default: hashing
```
Prompts are already real prompts (clear instructions + output format), so the
same code works with OpenAI-compatible endpoints.

### Environments (papers/envs)
All follow `reset() -> str`, `step(action) -> StepResult`,
`valid_actions()`, `task_description`, `score` (0..1), `max_steps`:
- `household.py` — ScienceWorld/ALFWorld-lite: 5 rooms, closed doors/containers,
  broken devices, ~6 task types × 3 variants, in/out-of-distribution split. The
  shared `household.act` handler is the policy used by CLIN, Reflexion, MemRL
  and MUSE.
- `craft.py` — CraftWorld: Minecraft-lite tech tree + `run_skill_code` (exec in
  a restricted namespace, with an operation budget).
- `web.py` — MiniWeb: shop/forum/gitlab-lite with search, forms, login and
  gotchas (unlinked admin, mutable cart/issues).

### MemorySystem (papers/common/memory_api.py)
Common interface for the memory benchmarks (p15–p17): `reset`, `add`,
`retrieve`. Bundled baselines: `FullContextMemory` (truncated window),
`BM25Memory`, `EmbeddingRAGMemory`.

### How to add a paper (checklist)
1. `papers/pNN_name/`: `__init__.py`, bilingual `README.md` (spec sections),
   `method.py`, `prompts.py` (templates + `@mock_handler`), `run.py`.
2. Read the paper's method section and cite sections/equations in the mapping.
3. Register handlers with unique names (`pNN.something`).
4. `run.py` prints the table and generates `RESULTS.md` with a fixed seed,
   offline < ~60 s.
5. `tests/test_pNN_name.py`: method invariants + determinism.
6. Add the folder to the root README and to `taxonomy.py` (p18).
