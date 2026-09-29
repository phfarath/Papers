"""PT: catálogo único dos 18 papers replicados (fonte de verdade para o
índice, a taxonomia do p08 e o survey do p18). Cada entrada: pasta, id
arXiv, título, ideia em uma linha PT/EN e ambiente usado.

EN: single source of truth for the 18 replicated papers (index, p08
taxonomy, p18 survey). Each entry: folder, arXiv id, title, one-line PT/EN
idea and the environment used.
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
]
