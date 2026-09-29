"""Reflexion (Shinn et al., 2023) — loop Actor/Evaluator/Self-Reflection.

PT: Implementa o Algoritmo 1 do paper: o Actor (política ReAct, aqui o handler
"household.act") gera uma trajetória; o Evaluator pontua (recompensa do env +
heurística tipo ALFWorld: ações repetidas / passos demais); o Self-Reflection
model gera uma reflexão verbal guardada na memória de longo prazo `mem`,
limitada a Ω (default 3) reflexões, e o ciclo repete até sucesso ou max_trials.
Também incluído: mini setting de programação com testes unitários auto-gerados
como evaluator interno.

EN: implements the paper's Algorithm 1: the Actor (ReAct policy, the
"household.act" handler) generates a trajectory; the Evaluator scores it (env
reward + an ALFWorld-style heuristic: repeated actions / too many steps); the
Self-Reflection model writes a verbal reflection into long-term memory `mem`,
bounded by Ω (default 3), and the loop repeats until success or max_trials.
Also included: a mini programming setting with self-generated unit tests as
the internal evaluator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from papers.common.llm import LLM, task_prompt
from papers.envs.household import HouseholdEnv, TaskSpec, score_actions
from papers.p02_reflexion.prompts import CodeProblem, reflection_body


@dataclass
class TrialResult:
    """PT: resultado de uma trial. EN: result of one trial."""

    success: bool
    score: float
    steps: int
    trajectory: list[tuple[str, str]]
    eval_note: str = ""


@dataclass
class ReflexionAgent:
    """PT: agente Reflexion para o HouseholdEnv.

    EN: Reflexion agent for HouseholdEnv.
    """

    llm: LLM
    omega: int = 3          # PT: limite Ω da memória. EN: memory bound Ω.
    max_trials: int = 4
    use_reflection: bool = True
    mem: list[str] = field(default_factory=list)  # reflexões / reflections

    # ---------- Evaluator ----------
    def evaluate(self, env: HouseholdEnv, traj: list[tuple[str, str]]) -> tuple[bool, str]:
        """PT: recompensa do env + heurística (repetição/estourar passos).

        EN: env reward + heuristic (repetition / exceeding steps).
        """
        if env.done:
            return True, "success"
        acts = [a for a, _ in traj]
        for i in range(len(acts) - 2):
            if acts[i] == acts[i + 1] == acts[i + 2]:
                return False, "failed: same action repeated 3x"
        if env.steps >= env.max_steps:
            return False, "failed: exceeded max steps"
        return False, "failed: task incomplete"

    # ---------- Actor (ReAct-style) ----------
    def act(self, task: str, obs: str, actions: list[str], recent: list[str]) -> str:
        body = (
            "You are a household agent (ReAct). Think briefly, then reply with "
            "ONLY the best action from the valid list.\n"
            f"Task: {task}\nObservation: {obs}\n"
            + ("Reflections:\n" + "\n".join(f"- {m}" for m in self.mem) + "\n"
               if self.mem else "")
            + ("Recent actions:\n" + "\n".join(f"- {r}" for r in recent[-8:]) + "\n"
               if recent else "")
            + "Valid actions:\n" + "\n".join(f"- {a}" for a in actions)
        )
        act = self.llm.complete(task_prompt("household.act", body)).strip()
        return act

    # ---------- Self-Reflection ----------
    def reflect(self, traj: list[tuple[str, str]], note: str, task: str) -> str:
        body = reflection_body(task, traj, note, self.mem)
        return self.llm.complete(task_prompt("reflexion.reflect", body)).strip()

    # ---------- loop de trials / trial loop ----------
    def run_task(self, spec: TaskSpec) -> list[TrialResult]:
        """PT: loop completo sobre uma tarefa. EN: full loop on one task."""
        results: list[TrialResult] = []
        for _ in range(self.max_trials):
            env = HouseholdEnv(spec)
            obs = env.reset()
            traj: list[tuple[str, str]] = []
            recent: list[str] = []
            while True:
                va = env.valid_actions()
                act = self.act(spec.description, obs, va, recent)
                if act not in va:
                    act = score_actions(spec.description, obs, va, self.mem,
                                        recent, random.Random(0))
                res = env.step(act)
                traj.append((act, res.observation))
                recent.append(act)
                obs = res.observation
                if res.done:
                    break
            ok, note = self.evaluate(env, traj)
            results.append(TrialResult(ok, env.score, env.steps, traj, note))
            if ok:
                break
            if self.use_reflection:
                new = self.reflect(traj, note, spec.description)
                if new not in self.mem:  # PT: evita reflexões idênticas.
                    self.mem = (self.mem + [new])[-self.omega:]
        return results


# ---------------------------------------------------------------------------
# Setting de programação / programming setting
# ---------------------------------------------------------------------------

def run_tests(code: str, tests: list[str]) -> tuple[bool, str]:
    """PT: evaluator — executa testes num namespace e devolve o 1º FAIL.

    EN: evaluator — runs tests in a namespace and returns the first FAIL.
    """
    ns: dict[str, object] = {}
    try:
        exec(code, ns)  # noqa: S102 - toy sandbox
        for t in tests:
            if not eval(t, ns):  # noqa: S307
                got = eval(t.split("==")[0].strip(), ns)
                return False, f"FAIL: {t} (got {got!r})"
    except Exception as e:
        return False, f"FAIL: {e}"
    return True, "PASS"


@dataclass
class CodeLoopResult:
    """PT: resultado do loop de programação. EN: code-loop result.

    `internal_ok` = passou nos testes auto-gerados; `hidden_ok` = pass@1 nos
    testes escondidos (a medida real de sucesso — paper §4.2, internal tests
    podem dar falsos positivos).
    """

    internal_ok: bool
    hidden_ok: bool
    n_trials: int


def derive_internal_tests(llm: LLM, problem: CodeProblem) -> list[str]:
    """PT: testes internos derivados SOMENTE dos exemplos da docstring
    (task "code.tests"). EN: internal tests derived ONLY from the docstring
    examples — like the paper's self-generated unit tests."""
    body = (
        "Generate unit tests (one per line, 'f(args) == expected') using ONLY "
        "the examples in the docstring.\n"
        f"Function: {problem.name}\nSignature: {problem.sig}\n"
        f"Doc: {problem.doc}"
    )
    out = llm.complete(task_prompt("code.tests", body))
    return [ln.strip() for ln in out.splitlines() if "==" in ln]


def reflexion_code_loop(llm: LLM, problem: CodeProblem,
                        use_reflection: bool,
                        max_trials: int = 4) -> CodeLoopResult:
    """PT: loop Reflexion de código: escreve → testes INTERNOS → reflexão.
    Sucesso real medido nos testes ESCONDIDOS ao final.
    EN: Reflexion code loop: write → INTERNAL tests → reflection. True success
    measured on the HIDDEN tests at the end."""
    internal = derive_internal_tests(llm, problem)
    mem: list[str] = []
    code = ""
    internal_ok = False
    n_trials = 0
    for _ in range(max_trials):
        n_trials += 1
        body = (
            "Implement the function below in Python. Reply with ONLY code.\n"
            f"Function: {problem.name}\nSignature: {problem.sig}\n"
            f"Doc: {problem.doc}\n"
            + ("Reflections:\n" + "\n".join(f"- {m}" for m in mem) + "\n"
               if mem else "")
        )
        code = llm.complete(task_prompt("code.write", body))
        internal_ok, report = run_tests(code, internal)
        if internal_ok:
            break
        if not use_reflection:
            break  # PT: sem reflexão não há aprendizado. EN: no learning.
        rbody = (
            "Reflect briefly on this failed submission.\n"
            f"Task: {problem.name} — {problem.doc}\n{report}\nCode:\n{code}"
        )
        mem = (mem + [llm.complete(task_prompt("code.reflect", rbody))])[-3:]
    hidden_ok, _ = run_tests(code, problem.hidden)
    return CodeLoopResult(internal_ok, hidden_ok, n_trials)
