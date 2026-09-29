"""PT: método do p06_voyager — Voyager (arXiv 2305.16291) no CraftWorld.

EN: p06_voyager method — Voyager on CraftWorld.

Componentes (paper §3): (a) currículo automático, (b) skill library com
embedding por descrição e retrieval top-5, (c) iterative prompting com
feedback do env + erros de execução + self-verification (critic, ≤4 rounds),
(d) skill só entra na biblioteca após sucesso verificado.

Components (paper §3): (a) automatic curriculum, (b) skill library indexed by
description embeddings with top-5 retrieval, (c) iterative prompting with
environment feedback + execution errors + self-verification (≤4 rounds),
(d) skills stored only on verified success.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from papers.common.embeddings import HashingEmbedder
from papers.common.llm import LLM, task_prompt
from papers.common.retrieval import VectorIndex
from papers.envs.craft import CraftWorld, run_skill_code
from papers.p06_voyager.prompts import CHAIN

MAX_ROUNDS = 4  # PT: §3.3 — até 4 rounds de self-correction. EN: ≤4 rounds.
TOP_K = 5       # PT: §3.3 — top-5 skills recuperadas. EN: top-5 retrieved.


@dataclass
class Skill:
    """PT: skill verificada — nome, descrição (indexada) e código.
    EN: verified skill — name, description (indexed), and code."""
    name: str
    description: str
    code: str


class SkillLibrary:
    """PT: biblioteca de skills (§3.3): busca vetorial top-k pela descrição.
    EN: skill library (§3.3): top-k vector retrieval over descriptions."""

    def __init__(self, seed: int = 0) -> None:
        self._index = VectorIndex(HashingEmbedder())
        self._skills: dict[str, Skill] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._skills)

    def add(self, skill: Skill) -> None:
        """PT: adiciona skill (chamado só após verificação do critic).
        EN: add a skill (called only after critic verification)."""
        if skill.name not in self._skills:
            self._order.append(skill.name)
            self._index.add(skill.name, skill.description)
        self._skills[skill.name] = skill

    def retrieve(self, llm: LLM, query: str, k: int = TOP_K) -> list[Skill]:
        """PT: top-k skills por embedding da descrição × consulta (a própria
        tarefa, como no paper). EN: top-k skills by description×query."""
        hits = self._index.search(query, k)
        return [self._skills[n] for n, _ in hits if n in self._skills]

    def names(self) -> list[str]:
        return list(self._order)


@dataclass
class Attempt:
    task: str
    success: bool
    rounds: int


@dataclass
class Iteration:
    """PT: resultado de uma iteração do currículo. EN: one curriculum iteration."""
    i: int
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    items: list[str] = field(default_factory=list)


class VoyagerAgent:
    """PT: agente Voyager (currículo + skills + iterative prompting + critic).
    EN: Voyager agent (curriculum + skills + iterative prompting + critic)."""

    def __init__(self, llm: LLM, use_library: bool = True,
                 use_curriculum: bool = True, use_critic: bool = True,
                 seed: int = 0) -> None:
        self.llm = llm
        self.use_library = use_library
        self.use_curriculum = use_curriculum
        self.use_critic = use_critic
        self.rng = random.Random(seed)
        self.library = SkillLibrary(seed=seed)
        self.completed: list[str] = []
        self.failed: list[str] = []

    # ---------- §3.2 currículo ----------
    def propose_task(self, env: CraftWorld) -> str:
        """PT: próxima tarefa do currículo (ou aleatória na ablação).
        EN: next curriculum task (or random in the no-curriculum ablation)."""
        if not self.use_curriculum:
            return f"{self.rng.choice(['mine', 'craft'])} " \
                f"{self.rng.choice(CHAIN)}"
        body = ("Propose the next task for the agent.\n"
                f"Episode target: {env.target}\n"
                f"Inventory: {env.inv}\n"
                f"Completed tasks: {', '.join(self.completed) or 'none'}\n"
                f"Failed tasks: {', '.join(self.failed) or 'none'}")
        return self.llm.complete(task_prompt("voyager.curriculum", body)).strip()

    # ---------- §3.3 execução iterativa ----------
    def _attempt(self, env: CraftWorld, task: str, item: str,
                 error: str) -> tuple[str, bool]:
        skills = self.library.retrieve(self.llm, task) if self.use_library else []
        skl_txt = "\n".join(f"- {s.name} — {s.description}\n```{s.code}```"
                            for s in skills)
        body = (f"Task: {task}\nInventory: {env.inv}\n"
                + ("Relevant skills:\n" + skl_txt + "\n" if skl_txt else "")
                + f"Last execution error: {error or 'none'}\n"
                "Write python code calling bot.mine/bot.craft/bot.smelt.")
        code = self.llm.complete(task_prompt("voyager.write_code", body)).strip()
        out = run_skill_code(env, code)
        ok = env.inv.get(item, 0) > 0 and "error:" not in out
        new_err = "" if ok else (out if "error:" in out else f"{item} missing")
        return code, ok, new_err

    def solve_task(self, env: CraftWorld, task: str) -> Attempt:
        """PT: até MAX_ROUNDS rodadas: gerar → executar → critic verifica →
        (sucesso: registra skill). EN: up to MAX_ROUNDS rounds: generate →
        execute → critic verifies → on success, register the skill."""
        item = task.split()[-1]
        error = ""
        code = ""
        ok = False
        rounds = 0
        for r in range(1, MAX_ROUNDS + 1):
            rounds = r
            code, ok, error = self._attempt(env, task, item, error)
            if ok:
                if self.use_critic:
                    verdict = self.llm.complete(task_prompt(
                        "voyager.critic",
                        f"Task: {task}\nInventory: {env.inv}\n"
                        f"Last execution error: {error or 'none'}")).strip()
                    ok = verdict.startswith("success")
                if ok:
                    break
        if ok:
            self.completed.append(task)
            desc = self.llm.complete(task_prompt(
                "voyager.suggest", f"Task: {task}")).strip()
            self.library.add(Skill(item, desc, code))
        else:
            self.failed.append(task)
        return Attempt(task, ok, rounds)


def run_episode(llm: LLM, env: CraftWorld, iterations: int,
                use_library: bool = True, use_curriculum: bool = True,
                use_critic: bool = True, seed: int = 0,
                library: SkillLibrary | None = None) -> tuple[list[Iteration],
                                                              SkillLibrary]:
    """PT: roda `iterations` tarefas do currículo; registra itens descobertos.
    Devolve também a biblioteca (para reuso zero-shot).

    EN: runs `iterations` curriculum tasks; records discovered items. Also
    returns the skill library (for zero-shot reuse)."""
    agent = VoyagerAgent(llm, use_library, use_curriculum, use_critic, seed)
    if library is not None:
        agent.library = library
    seen: set[str] = set()
    hist: list[Iteration] = []
    for i in range(1, iterations + 1):
        task = agent.propose_task(env)
        att = agent.solve_task(env, task)
        seen.update(k for k, v in env.inv.items() if v > 0)
        hist.append(Iteration(i, task, [att], sorted(seen)))
    return hist, agent.library


def milestones(hist: list[Iteration], chain: list[str]) -> dict[str, int]:
    """PT: índice da iteração em que cada marco foi obtido (0 = nunca).
    EN: iteration index where each milestone item was first obtained."""
    out = {m: 0 for m in chain}
    for it in hist:
        for m in chain:
            if out[m] == 0 and m in it.items:
                out[m] = it.i
    return out
