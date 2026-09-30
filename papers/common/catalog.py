"""PT: catálogo único dos 31 papers replicados (fonte de verdade para o
índice, a taxonomia do p08 e o survey do p18). Cada entrada: pasta, id
arXiv/DOI, título, ideia em uma linha PT/EN e ambiente usado.

EN: single source of truth for the 31 replicated papers (index, p08
taxonomy, p18 survey). Each entry: folder, arXiv/DOI id, title, one-line
PT/EN idea and the environment used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperInfo:
    folder: str
    arxiv: str
    title: str
    idea_pt: str
    idea_en: str
    env: str


CATALOG: list[PaperInfo] = [
    PaperInfo("p01_clin", "2310.10134", "CLIN",
              "memória de abstrações causais que guia a próxima tentativa",
              "causal-abstraction memory guiding the next trial",
              "HouseholdEnv"),
    PaperInfo("p02_reflexion", "2303.11366", "Reflexion",
              "reflexões verbais episódicas após cada falha",
              "episodic verbal reflections after each failure",
              "HouseholdEnv + programming"),
    PaperInfo("p03_memrl", "2411.13537v2", "MemRL",
              "tripletos (intenção, experiência, utilidade-Q) recuperados em "
              "duas fases", "(intent, experience, Q-utility) triplets with "
              "two-phase retrieval", "HouseholdEnv"),
    PaperInfo("p04_muse", "2601.03192", "MUSE",
              "metacognição: auto-avaliação prevê quando buscar reflexão",
              "metacognition: self-assessment predicts when to reflect",
              "HouseholdEnv"),
    PaperInfo("p05_cer", "2506.06698", "CER",
              "destilação de dinâmicas e skills a partir de trajetórias",
              "dynamics + skill distillation from trajectories",
              "MiniWeb"),
    PaperInfo("p06_voyager", "2305.16291", "Voyager",
              "currículo automático + biblioteca de skills verificadas",
              "automatic curriculum + verified skill library",
              "CraftWorld"),
    PaperInfo("p07_muse_autoskill", "2605.27366", "MUSE-Autoskill",
              "ciclo de vida de skills (criar→testar→usar→refinar→prunar)",
              "skill lifecycle (create→test→use→refine→prune)",
              "SkillsBench-lite"),
    PaperInfo("p08_coala", "2309.02427", "CoALA",
              "framework: tipos de memória, ações internas/externas, ciclo "
              "de decisão", "framework: memory types, internal/external "
              "actions, decision cycle", "HouseholdEnv"),
    PaperInfo("p09_generative_agents", "2304.03442", "Generative Agents",
              "recuperação recência+importância+relevância, reflexão, "
              "planejamento", "recency+importance+relevance retrieval, "
              "reflection, planning", "Mini-Smallville"),
    PaperInfo("p10_memgpt", "2310.08560", "MemGPT",
              "memória em camadas tipo SO gerida por function calls",
              "OS-style layered memory managed via function calls",
              "conv dataset"),
    PaperInfo("p11_a_mem", "2502.12110", "A-MEM",
              "notas agentic com links e evolução de vizinhos",
              "agentic notes with links and neighbor evolution",
              "conv dataset"),
    PaperInfo("p12_zep", "2501.13956", "Zep / Graphiti",
              "grafo bitemporal com invalidação de contradições",
              "bitemporal graph with contradiction invalidation",
              "conv dataset"),
    PaperInfo("p13_mem0", "2504.19413", "Mem0",
              "pipeline extração→ops (ADD/UPDATE/DELETE/NOOP) + variante "
              "grafo", "extraction→ops (ADD/UPDATE/DELETE/NOOP) pipeline + "
              "graph variant", "conv dataset"),
    PaperInfo("p14_memory_r1", "2508.19828", "Memory-R1",
              "RL treina a política do gerenciador de memória (PPO/GRPO)",
              "RL trains the memory-manager policy (PPO/GRPO)",
              "conv dataset"),
    PaperInfo("p15_longmemeval", "2410.10813", "LongMemEval",
              "benchmark de memória de longo prazo em chat (5 habilidades)",
              "long-term chat memory benchmark (5 abilities)",
              "synthetic histories"),
    PaperInfo("p16_memoryagentbench", "2507.05257", "MemoryAgentBench",
              "benchmark de memória via interações incrementais (AR/TTL/"
              "LRU/SF)", "memory benchmark via incremental interactions "
              "(AR/TTL/LRU/SF)", "chunked streams"),
    PaperInfo("p17_longmemeval_v2", "2605.12493v1", "LongMemEval-V2",
              "benchmark de memória sobre trajetórias de agente web + "
              "AgentRunbook", "web-agent trajectory memory benchmark + "
              "AgentRunbook", "MiniWeb trajectories"),
    PaperInfo("p18_survey", "2512.13564", "Memory Survey",
              "taxonomia de memória de agentes: forms, functions, dynamics",
              "agent-memory taxonomy: forms, functions, dynamics",
              "repo-wide"),
    PaperInfo("p19_astrocyte_assoc", "2311.08135",
              "Neuron–Astrocyte Associative Memory",
              "acoplamentos de ordem 3 (astrócito) estabilizam memórias",
              "order-3 (astrocyte) couplings stabilize memories",
              "numpy offline"),
    PaperInfo("p20_astrocyte_context", "10.1371/journal.pcbi.1012186",
              "Astrocytes Contextually-Guided Dynamics",
              "modulação em duas escalas de tempo liga contexto a pesos",
              "two-timescale modulation binds context to weights",
              "numpy offline"),
    PaperInfo("p21_astrocyte_transformer", "10.1073/pnas.2219150120",
              "Transformers from Neurons and Astrocytes",
              "ODE neurônio–astrócito converge para atenção softmax",
              "neuron–astrocyte ODE converges to softmax attention",
              "numpy offline"),
    PaperInfo("p22_neuron_glia_nets", "10.1371/journal.pone.0019109",
              "Artificial Neuron–Glia Networks",
              "glia detecta estagnação do gradiente e injeta ruído",
              "glia detects gradient stalls and injects noise",
              "numpy offline"),
    PaperInfo("p23_astrocyte_wm", "10.3389/fncel.2021.631485",
              "Working Memory in SNN with Astrocytes",
              "Ca²⁺ glial sustenta assembléias WM contra distrator",
              "glial Ca²⁺ sustains WM assemblies against distractors",
              "numpy offline"),
    PaperInfo("p24_situation_memory", "10.1109/TNNLS.2023.3335450",
              "Situation-Based Neuromorphic Memory",
              "memória de situação exige cue E contexto da glia",
              "situation memory requires both cue AND glial context",
              "numpy offline"),
    PaperInfo("p25_agmp_continual", "10.3389/fnins.2025.1768235",
              "AGMP Continual Learning",
              "gate metaplástico glial regula estabilidade–plasticidade",
              "glial metaplastic gate tunes stability–plasticity",
              "numpy offline"),
    PaperInfo("p26_dual_memory_nav", "2604.15391",
              "Dual-Timescale Memory for Navigation",
              "supressão glial de locais visitados acelera navegação",
              "glial suppression of visited places speeds navigation",
              "numpy offline"),
    PaperInfo("p27_emergent_attention", "2604.25481",
              "Emergent Self-Attention (Replicator)",
              "dinâmica replicadora glial gera pesos de atenção softmax",
              "glial replicator dynamics yields softmax attention",
              "numpy offline"),
    PaperInfo("p28_astromorphic_repair", "10.1609/aaai.v37i6.25947",
              "Astromorphic Self-Repair",
              "domínios gliais guardam gabaritos para enxerto após falha",
              "glial domains keep templates for graft repair",
              "numpy offline"),
    PaperInfo("p29_lsm_astrocytes", "2503.06798",
              "LSM with Astrocyte-Like Units",
              "unidades lentas dão features temporais; razão ótima em U",
              "slow units give temporal features; U-shaped optimal ratio",
              "numpy offline"),
    PaperInfo("p30_hybrid_automaton", "2609.16217",
              "Neural-Astrocyte Hybrid Automaton",
              "acumulador glial de evidência troca de modo/contexto",
              "glial evidence accumulator switches mode/context",
              "numpy offline"),
    PaperInfo("p31_futility_passivity", "10.1016/j.cell.2019.05.050",
              "Glia Accumulate Futility Evidence",
              "evidência de futilidade suprime ação; retoma após descanso",
              "futility evidence suppresses action; resumes after rest",
              "numpy offline"),
]
