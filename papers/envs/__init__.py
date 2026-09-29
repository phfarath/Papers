"""Ambientes-brinquedo deterministicos / Deterministic toy environments.

PT: Todos os envs seguem a mesma interface: `reset()`, `step(action)`,
`valid_actions()`, `task_description`, `score` (0..1, estilo ScienceWorld),
`max_steps`. Nenhum env usa LLM por dentro — a "inteligencia" fica no agente.

EN: All envs share one interface: `reset()`, `step(action)`,
`valid_actions()`, `task_description`, `score` (0..1, ScienceWorld-style),
`max_steps`. No env uses an LLM internally — the "intelligence" lives in the
agent.
"""

from dataclasses import dataclass, field


@dataclass
class StepResult:
    """PT: resultado de um passo. EN: result of one step."""

    observation: str
    reward: float
    done: bool
    info: dict[str, str] = field(default_factory=dict)
