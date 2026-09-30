"""PT: p18 — "Memory in the Age of AI Agents: A Survey" (arXiv 2512.13564).
Classificação de CADA implementação do repo (p01–p17) na taxonomia do
survey (§3–§5), como dados com justificativas:

- **Forms** (§3): token-level {flat(1D), planar(2D/grafo),
  hierarchical(3D)}, parametric, latent.
- **Functions** (§4): factual, experiential, working.
- **Dynamics** (§5): formation, evolution, retrieval.

Além disso, a tabela de distinção memory vs LLM-memory vs RAG vs context
engineering (§2.3). A lista de papers vem de papers/common/catalog.py —
fonte única de verdade.

EN: p18 — the agent-memory survey. Classifies every repo implementation
(p01–p17) along the survey taxonomy (forms/functions/dynamics) with
justifications, plus the memory-vs-RAG-vs-context-engineering distinction
table (§2.3). Paper list comes from papers/common/catalog.py — single
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from papers.common.catalog import CATALOG

# PT: rótulos válidos da taxonomia. EN: valid taxonomy labels.
FORMS = {"flat", "planar", "hierarchical", "parametric", "latent"}
FUNCTIONS = {"factual", "experiential", "working"}
DYNAMICS = {"formation", "evolution", "retrieval"}


@dataclass(frozen=True)
class Classification:
    folder: str
    title: str
    form: str                    # PT: um de FORMS. EN: one of FORMS.
    functions: str               # PT: subset de FUNCTIONS ("f+e+w").
    dynamics: str                # PT: subset de DYNAMICS ("f+e+r").
    justification_pt: str
    justification_en: str


_F, _E, _W = "factual", "experiential", "working"
_FM, _EV, _RT = "formation", "evolution", "retrieval"


def _c(folder: str, form: str, fun: str, dyn: str,
       pt: str, en: str) -> Classification:
    title = next((p.title for p in CATALOG if p.folder == folder), folder)
    return Classification(folder, title, form, fun, dyn, pt, en)


CLASSIFICATIONS: list[Classification] = [
    _c("p01_clin", "flat", f"{_E}+{_F}", f"{_FM}+{_RT}",
       "memória textual plana de abstrações causais; formada por reflexão, "
       "recuperada por recência",
       "flat textual memory of causal abstractions; formed by reflection, "
       "retrieved by recency"),
    _c("p02_reflexion", "flat", f"{_E}+{_W}", f"{_FM}+{_RT}",
       "reflexões verbais episódicas em buffer plano",
       "episodic verbal reflections in a flat buffer"),
    _c("p03_memrl", "flat", f"{_E}+{_F}", f"{_EV}+{_RT}",
       "tripletos com utilidade Q que EVOLUI via aprendizado",
       "triplets whose Q-utility EVOLVES via learning"),
    _c("p04_muse", "flat", f"{_E}+{_W}", f"{_FM}+{_RT}",
       "reflexões planas; a formação é decidida por metacognição (g_η)",
       "flat reflections; formation gated by metacognition (g_η)"),
    _c("p05_cer", "hierarchical", f"{_E}+{_F}", f"{_FM}+{_RT}",
       "memória em camadas (dinâmicas vs skills abstratas) — separação "
       "hierárquica por tipo",
       "layered memory (dynamics vs abstract skills) — hierarchical split "
       "by type"),
    _c("p06_voyager", "flat", "experiential", f"{_FM}+{_RT}",
       "biblioteca plana de skills indexadas por descrição",
       "flat skill library indexed by description"),
    _c("p07_muse_autoskill", "hierarchical",
       f"{_E}+{_F}+{_W}", f"{_FM}+{_EV}+{_RT}",
       "skills em diretórios + ciclo de vida (refinar/prunar = evolução)",
       "skill dirs + lifecycle (refine/prune = evolution)"),
    _c("p08_coala", "hierarchical", f"{_F}+{_E}+{_W}",
       f"{_FM}+{_RT}",
       "framework: memória working/episodic/semantic em camadas",
       "framework: working/episodic/semantic layered memory"),
    _c("p09_generative_agents", "hierarchical", f"{_E}+{_F}",
       f"{_FM}+{_EV}+{_RT}",
       "memória observacional + reflexões sintetizadas (evolução)",
       "observation memory + synthesized reflections (evolution)"),
    _c("p10_memgpt", "hierarchical", f"{_F}+{_W}",
       f"{_FM}+{_EV}+{_RT}",
       "camadas core/recall/archival com paginação (hierarquia de "
       "contexto)", "core/recall/archival layers with paging (context "
       "hierarchy)"),
    _c("p11_a_mem", "planar", f"{_F}+{_E}", f"{_FM}+{_EV}+{_RT}",
       "notas com links entre vizinhos — grafo 2D; evolução de vizinhos",
       "linked notes — 2D graph; neighbor evolution"),
    _c("p12_zep", "planar", "factual", f"{_FM}+{_EV}+{_RT}",
       "grafo bitemporal — arestas invalidadas por contradição (evolução)",
       "bitemporal graph — edges invalidated on contradiction (evolution)"),
    _c("p13_mem0", "flat", "factual", f"{_FM}+{_EV}+{_RT}",
       "store plano de fatos com ops ADD/UPDATE/DELETE (evolução)",
       "flat fact store with ADD/UPDATE/DELETE ops (evolution)"),
    _c("p14_memory_r1", "flat", "factual", f"{_EV}+{_RT}",
       "manager plano; a EVOLUÇÃO é aprendida por RL",
       "flat manager; EVOLUTION is RL-learned"),
    _c("p15_longmemeval", "flat", f"{_F}+{_E}", f"{_RT}",
       "benchmark — avalia forma/valor/chave/leitura do retrieval",
       "benchmark — evaluates retrieval's value/key/query/reading"),
    _c("p16_memoryagentbench", "flat", f"{_F}+{_E}", f"{_RT}+{_FM}",
       "benchmark de ingestão incremental — competências AR/TTL/LRU/SF",
       "incremental-ingestion benchmark — AR/TTL/LRU/SF"),
    _c("p17_longmemeval_v2", "hierarchical", f"{_E}+{_W}",
       f"{_FM}+{_RT}",
       "benchmark sobre trajetórias; runbook usa pools/arquivos "
       "estruturados",
       "trajectory benchmark; runbook uses structured pools/files"),
    _c("p19_astrocyte_assoc", "parametric", f"{_E}+{_F}",
       f"{_FM}+{_RT}",
       "memória associativa nos pesos de ordem 3 (astrócito)",
       "associative memory in order-3 (astrocyte) weights"),
    _c("p20_astrocyte_context", "parametric", f"{_E}+{_W}",
       f"{_FM}+{_EV}+{_RT}",
       "gates de contexto consolidam/restauram pesos (evolução)",
       "context gates consolidate/restore weights (evolution)"),
    _c("p21_astrocyte_transformer", "latent", "working",
       f"{_FM}+{_RT}",
       "normalização glial lenta age como atenção sobre a entrada",
       "slow glial normalization acts as attention over input"),
    _c("p22_neuron_glia_nets", "parametric", f"{_E}+{_F}",
       f"{_FM}+{_EV}",
       "ganhos gliais de co-atividade modulam aprendizado",
       "glial co-activity gains modulate learning"),
    _c("p23_astrocyte_wm", "latent", "working",
       f"{_FM}+{_EV}+{_RT}",
       "assembléias WM mantidas por Ca²⁺ glial (estado latente)",
       "WM assemblies held by glial Ca²⁺ (latent state)"),
    _c("p24_situation_memory", "flat", f"{_E}+{_W}",
       f"{_FM}+{_EV}+{_RT}",
       "matriz de situações cue+contexto; contexto glial evolui/decai",
       "cue+context situation matrix; glial context evolves/decays"),
    _c("p25_agmp_continual", "parametric", f"{_E}+{_F}",
       f"{_FM}+{_EV}",
       "importância glial metaplástica preserva pesos antigos",
       "glial metaplastic importance preserves old weights"),
    _c("p26_dual_memory_nav", "planar", f"{_E}+{_W}",
       f"{_FM}+{_EV}+{_RT}",
       "mapa 2D de supressão + traços longos de sucesso",
       "2D suppression map + long-term success traces"),
    _c("p27_emergent_attention", "latent", "working",
       f"{_EV}+{_RT}",
       "ganhos gliais no simplex; atenção emerge como roteamento",
       "glial gains on the simplex; attention as routing"),
    _c("p28_astromorphic_repair", "flat", f"{_F}+{_E}",
       f"{_FM}+{_RT}",
       "templates gliais de domínio servem de memória para reparo",
       "glial domain templates serve as memory for repair"),
    _c("p29_lsm_astrocytes", "latent", "working",
       f"{_FM}+{_EV}",
       "unidades lentas carregam história temporal do reservatório",
       "slow units carry the reservoir's temporal history"),
    _c("p30_hybrid_automaton", "hierarchical", f"{_E}+{_W}",
       f"{_FM}+{_RT}",
       "Q-tables por modo (contexto) + acumulador de evidência",
       "per-mode Q-tables + evidence accumulator"),
    _c("p31_futility_passivity", "flat", "experiential",
       f"{_FM}+{_EV}",
       "evidência escalar de futilidade suprime e decai",
       "scalar futility evidence suppresses and decays"),
]


DISTINCTION_ROWS = [
    ("LLM memory", "pesos do modelo / KV-cache (parametric/latent)",
     "parametric", "implícito nos pesos — não endereçável"),
    ("RAG", "recuperação de documentos externos estáticos",
     "retrieval-only", "sem formação/evolução; memória passiva"),
    ("Context engineering", "orquestração do que entra no contexto",
     "—", "opera sobre contexto, não sobre store persistente"),
    ("Agent memory (survey)", "store persistente que se forma, evolui e "
     "é recuperado", f"{_FM}+{_EV}+{_RT}", "todas as três dinâmicas"),
]
