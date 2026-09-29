# p18 — Memory Survey (arXiv 2512.13564)

## Português

Resumo didático do survey **"Memory in the Age of AI Agents"** +
classificação de cada implementação do repo na taxonomia do paper.

**Forms (§3) — o que carrega a memória:**
- *token-level*: flat (1D — listas de textos), planar (2D — grafos como
  A-MEM/Zep), hierarchical (3D — camadas como MemGPT);
- *parametric* — nos pesos do modelo;
- *latent* — embeddings/KV escondidos.

**Functions (§4):** factual (fatos estáveis), experiential (episódios/
skills), working (estado transitório de raciocínio).

**Dynamics (§5):** formation (como se escreve), evolution (como se atualiza/
esquece/refina), retrieval (como se lê).

**Distinções (§2.3):** LLM memory = implícito nos pesos; RAG = store
estático de recuperação; context engineering = orquestração do contexto,
não store persistente; agent memory = as três dinâmicas juntas.

`taxonomy.py` classifica p01–p17 (fonte única: `papers/common/catalog.py`)
com justificativas; `RESULTS.md` renderiza as tabelas e o mapa entre
papers. Um teste garante cobertura total e rótulos válidos.

## English

Didactic summary of the **"Memory in the Age of AI Agents"** survey + a
classification of every repo implementation in the paper's taxonomy.

**Forms (§3):** token-level {flat 1D lists, planar 2D graphs (A-MEM, Zep),
hierarchical 3D layers (MemGPT)}, parametric (weights), latent
(embeddings/hidden state). **Functions (§4):** factual, experiential,
working. **Dynamics (§5):** formation, evolution, retrieval.

**Distinctions (§2.3):** LLM memory is implicit in weights; RAG is a
static retrieval store; context engineering orchestrates context, not a
persistent store; agent memory is all three dynamics together.

`taxonomy.py` classifies p01–p17 (single source:
`papers/common/catalog.py`) with justifications; `RESULTS.md` renders the
tables and the paper map. A test enforces full coverage and valid labels.
