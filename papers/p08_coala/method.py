"""PT: método do p08_coala — agente CoALA (arXiv 2309.02427) no HouseholdEnv.

EN: p08_coala method — a CoALA-style agent on HouseholdEnv.

CoALA (§3): memories = working + episodic + semantic + procedural; internal
actions (reasoning=WM update, retrieval=LTM read, learning=LTM write) +
external grounding actions; decision cycle = planning (propose → evaluate →
select) then execute.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from papers.common.llm import LLM, task_prompt
from papers.envs.household import (
    EpisodeResult,
    HouseholdEnv,
    TaskSpec,
    score_actions,
)
from papers.p08_coala import prompts  # noqa: F401  (registra handlers)


@dataclass
class WorkingMemory:
    """PT: memória de trabalho — tarefa corrente + plano (scratchpad).
    EN: working memory — current task + plan (scratchpad)."""
    task: str = ""
    plan: str = ""
    recent: list[str] = field(default_factory=list)


@dataclass
class LongTermMemory:
    """PT: LTM = episodic (episódios passados) + semantic (regras extraídas).
    EN: LTM = episodic (past episodes) + semantic (extracted rules)."""
    episodic: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)

    def all_lines(self) -> list[str]:
        return self.episodic + self.semantic


class CoALAAgent:
    """PT: agente com ciclo de decisão CoALA. EN: CoALA decision-cycle agent."""

    def __init__(self, llm: LLM, seed: int = 0) -> None:
        self.llm = llm
        self.rng = random.Random(seed)
        self.wm = WorkingMemory()
        self.ltm = LongTermMemory()

    # ---------- ações internas (§3.2) ----------
    def _internal_retrieve(self) -> list[str]:
        """PT: retrieval — lê a LTM. EN: retrieval — read LTM."""
        if not self.ltm.all_lines():
            return []
        out = self.llm.complete(task_prompt(
            "coala.retrieve",
            f"Task: {self.wm.task}\nLong-term memory:\n"
            + "\n".join(f"- {m}" for m in self.ltm.all_lines())))
        return [ln[2:] for ln in out.splitlines() if ln.startswith("- ")
                and ln != "- none"]

    def _internal_reason(self, obs: str) -> None:
        """PT: reasoning — atualiza a WM com o plano. EN: reasoning — update WM."""
        self.wm.plan = self.llm.complete(task_prompt(
            "coala.reason", f"Task: {self.wm.task}\nObservation:\n{obs}"))

    def _internal_learn(self, traj: list[tuple[str, str]],
                        success: bool) -> None:
        """PT: learning — escreve na LTM (episodic sempre; semantic se a regra
        for extraível). EN: learning — write to LTM (episodic always; semantic
        when a rule is extractable)."""
        traj_txt = " ; ".join(a for a, _ in traj)
        # PT: nota episódica neutra (sem marcas de conselho bom/ruim — é só
        # um registro de que o episódio aconteceu).
        # EN: neutral episodic note (no good/bad-advice markers — it is just
        # a record that the episode happened).
        self.ltm.episodic.append(
            f"episode '{self.wm.task[:40]}' at step "
            f"{len(traj)}: outcome={'solved' if success else 'unsolved'}")
        rule = self.llm.complete(task_prompt(
            "coala.learn",
            f"Task: {self.wm.task}\nTrajectory: {traj_txt}")).strip()
        if success and rule and rule not in self.ltm.semantic:
            self.ltm.semantic.append(rule)
        # PT: da falha, registra em semantic só o que o ENV disse que não
        # funcionou (observação com erro) — nunca condena um device que só
        # falhou por ordem de subgoals. EN: on failure, record only what the
        # env explicitly rejected (error observation) — never blame a device
        # that just ran out of steps.
        if not success:
            for act, obs in traj:
                m = re.match(
                    r"(?:heat|cool|wash|plant) (\w+) (?:with|in) (\w+)",
                    act.lower())
                if m and ("cannot" in obs.lower() or "does not work"
                          in obs.lower()):
                    line = (f"{m.group(2)} does not contribute to the "
                            f"{m.group(1)} task")
                    if line not in self.ltm.semantic:
                        self.ltm.semantic.append(line)

    # ---------- ciclo de decisão (§3.3): plan → execute ----------
    def _select(self, obs: str, va: list[str], mem: list[str]) -> str:
        """PT: planning = propose (valid actions) → evaluate (score) → select
        (argmax). EN: propose → evaluate → select."""
        return score_actions(self.wm.task, obs, va, mem, self.wm.recent,
                             self.rng)

    def run_episode(self, task: TaskSpec) -> EpisodeResult:
        env = HouseholdEnv(task)
        obs = env.reset()
        self.wm = WorkingMemory(task=task.description)
        traj: list[tuple[str, str]] = []
        while True:
            # PT: internal actions primeiro — reason (WM) + retrieve (LTM).
            # EN: internal actions first — reason (WM) + retrieve (LTM).
            self._internal_reason(obs)
            mem = self._internal_retrieve()
            va = env.valid_actions()
            act = self._select(obs, va, mem)
            if act not in va:
                act = va[0]
            res = env.step(act)
            traj.append((act, res.observation))
            self.wm.recent.append(act)
            obs = res.observation
            if res.done:
                break
        # PT: learning — escreve o episódio na LTM. EN: write episode to LTM.
        self._internal_learn(traj, env.done)
        return EpisodeResult(task.task_id, env.done, env.score, env.steps,
                             traj)


def run_react(llm: LLM, task: TaskSpec, seed: int = 0) -> EpisodeResult:
    """PT: baseline ReAct — mesmo env, sem LTM nem ciclo interno.
    EN: ReAct baseline — same env, no LTM, no internal cycle."""
    from papers.envs.household import run_episode
    return run_episode(llm, task, seed_rng=random.Random(seed))
