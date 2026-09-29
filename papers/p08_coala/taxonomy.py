"""PT: taxonomia CoALA — classifica cada módulo p01–p18 do repo (pelo nome,
incluindo os ainda não implementados) nas dimensões do paper: tipos de
memória, espaço de ações e procedimento de decisão.

EN: CoALA taxonomy — classifies every p01–p18 module in the repo (by name,
including not-yet-implemented ones) along the paper's dimensions: memory
types, action space, and decision procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

W = "working"
E = "episodic"
S = "semantic"
P = "procedural"


@dataclass
class PaperModule:
    folder: str
    method: str
    memories: str            # PT: tipos CoALA usados. EN: CoALA memory types.
    action_space: str        # PT: internal/external. EN: internal/external.
    decision: str            # PT: procedimento de decisão. EN: decision proc.
    implemented: bool


MODULES: list[PaperModule] = [
    PaperModule("p01_clin", "CLIN", f"{W}+{E}+{S}",
                "internal (memory gen) + external", "ReAct + causal memory", True),
    PaperModule("p02_reflexion", "Reflexion", f"{W}+{E}",
                "internal (reflect) + external", "ReAct + verbal reflection", True),
    PaperModule("p03_memrl", "MemRL", f"{S}+{E}",
                "internal (retrieve/learn)", "retrieval + Q-update", True),
    PaperModule("p04_muse", "MUSE (metacognition)", f"{W}+{E}",
                "internal (self-assess/regulate) + external",
                "MLP g_η + rollout control", True),
    PaperModule("p05_cer", "CER", f"{S}+{P}",
                "internal (distill/retrieve) + external",
                "ReAct + separate dyn/skill retrieval", True),
    PaperModule("p06_voyager", "Voyager", f"{P}",
                "internal (write/verify code) + external",
                "curriculum + iterative prompting", True),
    PaperModule("p07_muse_autoskill", "MUSE-Autoskill", f"{P}+{E}",
                "internal (skill lifecycle) + external",
                "catalog select + DAG compression", True),
    PaperModule("p08_coala", "CoALA", f"{W}+{E}+{S}+{P}",
                "internal (reason/retrieve/learn) + external",
                "propose→evaluate→select", True),
    PaperModule("p09_generative_agents", "Generative Agents", f"{E}+{S}",
                "internal (retrieve/reflect/plan) + external",
                "retrieval score + planning", True),
    PaperModule("p10_memgpt", "MemGPT", f"{W}+{E}+{S}",
                "internal (paging/function calls) + external",
                "LLM-OS paging", True),
    PaperModule("p11_a_mem", "A-MEM", f"{E}+{S}",
                "internal (link/evolve notes)",
                "agentic note construction", True),
    PaperModule("p12_zep", "Zep/Graphiti", f"{E}+{S}",
                "internal (graph updates/retrieval)",
                "temporal knowledge graph", True),
    PaperModule("p13_mem0", "Mem0", f"{E}+{S}",
                "internal (extract/update/retrieve)",
                "memory extraction + retrieval", True),
    PaperModule("p14_memory_r1", "Memory-R1", f"{E}+{S}",
                "internal (RL-trained memory ops)",
                "RL over memory operations", True),
    PaperModule("p15_longmemeval", "LongMemEval", f"{E}+{S}",
                "internal (retrieve from history)",
                "benchmark for chat memory", True),
    PaperModule("p16_memoryagentbench", "MemoryAgentBench", f"{E}+{S}+{P}",
                "internal (bench ops) + external",
                "benchmark for memory agents", True),
    PaperModule("p17_longmemeval_v2", "LongMemEval-V2", f"{E}+{S}",
                "internal (retrieve from history)",
                "benchmark v2", True),
    PaperModule("p18_survey", "Survey", "—", "—", "taxonomy survey", True),
]


def describe() -> list[PaperModule]:
    """PT: devolve a classificação de todos os módulos. EN: the table."""
    return MODULES


def render_table() -> str:
    lines = ["| module | method | memories | action space | decision | impl |",
             "|---|---|---|---|---|---|"]
    for m in describe():
        lines.append(f"| {m.folder} | {m.method} | {m.memories} | "
                     f"{m.action_space} | {m.decision} | "
                     f"{'yes' if m.implemented else 'soon'} |")
    return "\n".join(lines)
