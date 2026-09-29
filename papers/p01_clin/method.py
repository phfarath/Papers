"""CLIN (Majumdar et al., 2023) — controller + executor + memory generator.

PT: Implementa o Algoritmo 1 do paper: a cada passo, o controller propõe o
próximo goal g_t (condicionado pela tarefa, histórico e memória recuperada); o
executor converte g_t numa ação válida a_t. Ao fim de cada trial, o memory
generator produz a nova memória S_{k+1} usando a trial mais recente
(g_t, a_t, o_t), a recompensa final r_k e as memórias das 3 trials mais
recentes. Meta-memória (§3.3): resume as melhores memórias por episódio
(auto-curriculum, arquivo de 10) com prompts diferentes para Gen-Env/Gen-Task.

EN: implements the paper's Algorithm 1 (see PT). Meta-memory (§3.3): summarizes
the best per-episode memories (auto-curriculum, archive of 10) with different
prompts for Gen-Env vs Gen-Task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from papers.common.llm import LLM, task_prompt
from papers.envs.household import HouseholdEnv, TaskSpec, score_actions
from papers.p01_clin import prompts  # noqa: F401  (registra handlers / registers handlers)


@dataclass
class ClinTrial:
    success: bool
    score: float
    steps: int
    trajectory: list[tuple[str, str, str]]  # (goal, action, observation)
    reward: float


@dataclass
class ClinAgent:
    """PT: agente CLIN. EN: CLIN agent."""

    llm: LLM
    max_trials: int = 4
    use_memory: bool = True
    structured: bool = True     # False = ablação free-form / free-form ablation
    memory: list[str] = field(default_factory=list)
    meta: list[str] = field(default_factory=list)  # meta-memória / meta-memory
    memory_history: list[list[str]] = field(default_factory=list)  # S_k
    archive: list[tuple[list[str], float]] = field(default_factory=list)  # p/ meta

    # ---------- controller / executor ----------
    def _goal(self, spec: TaskSpec, obs: str, hist: str) -> str:
        mem = self.memory + self.meta
        body = (
            "You are the controller of a household agent. Output the next "
            "sub-goal (one line).\n"
            f"Task: {spec.description}\nObservation: {obs}\n"
            f"Trial so far:\n{hist}\n"
            + ("Memory:\n" + "\n".join(f"- {m}" for m in mem) if mem else "")
        )
        return self.llm.complete(task_prompt("clin.goal", body)).strip()

    def _act(self, spec: TaskSpec, goal: str, obs: str,
             actions: list[str], recent: list[str]) -> str:
        mem = self.memory + self.meta
        body = (
            "Map the goal to ONE valid action. Reply with ONLY the action.\n"
            f"Task: {spec.description}\nGoal: {goal}\nObservation: {obs}\n"
            + ("Memory:\n" + "\n".join(f"- {m}" for m in mem) + "\n" if mem else "")
            + ("Recent actions:\n" + "\n".join(f"- {r}" for r in recent[-8:]) + "\n"
               if recent else "")
            + "Valid actions:\n" + "\n".join(f"- {a}" for a in actions)
        )
        act = self.llm.complete(task_prompt("clin.act", body)).strip()
        if act not in actions:
            act = score_actions(goal + " " + spec.description, obs, actions,
                                mem, recent, random.Random(0))
        return act

    # ---------- trial / memory generator ----------
    def run_trial(self, spec: TaskSpec) -> ClinTrial:
        env = HouseholdEnv(spec)
        obs = env.reset()
        traj: list[tuple[str, str, str]] = []
        recent: list[str] = []
        while True:
            hist = "\n".join(f"({g}, {a}) -> {o[:60]}" for g, a, o in traj[-6:])
            goal = self._goal(spec, obs, hist or "(start)")
            act = self._act(spec, goal, obs, env.valid_actions(), recent)
            res = env.step(act)
            traj.append((goal, act, res.observation))
            recent.append(act)
            obs = res.observation
            if res.done:
                break
        return ClinTrial(env.done, env.score, env.steps, traj, res.reward)

    def update_memory(self, spec: TaskSpec, trial: ClinTrial) -> None:
        """PT: memory generator — usa trial + r_k + 3 memórias mais recentes.

        EN: memory generator — trial + final reward + 3 most recent memories.
        """
        traj_txt = "\n".join(f"({g}, {a}) -> {o.split(' You are in')[0]}"
                             for g, a, o in trial.trajectory)
        recent_mem = self.memory_history[-3:]
        body = (
            "Generate causal abstractions in the forms 'X [should/may] be "
            "necessary to Y' and 'X does not contribute to Y'.\n"
            f"Task: {spec.description}\nFinal reward: {trial.reward}\n"
            f"Latest trial:\n{traj_txt}\n"
            "Recent memories:\n"
            + ("\n".join("- " + m for m2 in recent_mem for m in m2) or "(none)")
        )
        task = "clin.memgen" if self.structured else "clin.freeform"
        out = self.llm.complete(task_prompt(task, body))
        self.memory = [ln[2:].strip() for ln in out.splitlines()
                       if ln.startswith("- ")]
        self.memory_history.append(self.memory)
        self.archive.append((self.memory, trial.reward))
        self.archive = sorted(self.archive, key=lambda x: x[1],
                              reverse=True)[:10]  # auto-curriculum

    def run_episode(self, spec: TaskSpec) -> list[ClinTrial]:
        trials = []
        for _ in range(self.max_trials):
            tr = self.run_trial(spec)
            trials.append(tr)
            if self.use_memory:
                self.update_memory(spec, tr)
            if tr.success:
                break
        return trials

    # ---------- meta-memória / meta-memory ----------
    def build_meta_memory(self, target: TaskSpec, mode: str) -> list[str]:
        """PT: meta-memória a partir das melhores memórias (§3.3).

        EN: meta-memory from the best memories; `mode` = 'gen-env'|'gen-task'.
        """
        best = [m for mem, _ in self.archive for m in mem]
        setting = ("to solve the same task in a new environment configuration"
                   if mode == "gen-env" else "to solve a different task")
        body = (
            f"Write meta-memory abstractions {setting}.\n"
            f"Target task: {target.description}\n"
            f"Setting tag: {mode}\nPast memories (with rewards):\n"
            + "\n".join(f"- {m}" for m in best)
        )
        out = self.llm.complete(task_prompt("clin.metagen", body))
        return [ln[2:].strip() for ln in out.splitlines()
                if ln.startswith("- ")]
