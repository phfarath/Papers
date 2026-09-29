"""CLIN — experimento executável / runnable experiment.

PT: `python -m papers.p01_clin.run [--seed 0] [--llm mock|openai]
[--write-results]` — avalia os 4 settings do paper sobre 20 seeds: Adapt
(memória cresce ao longo das trials na mesma tarefa/ambiente — reportamos a
curva trial-a-trial, a figura principal do paper), Gen-Env (meta-memória → novo
ambiente), Gen-Task (meta-memória → nova tarefa) e a ablação structured vs
free-form memory.

EN: evaluates the paper's 4 settings over 20 seeds: Adapt (with the
trial-by-trial success curve — the paper's main figure), Gen-Env, Gen-Task and
the structured vs free-form ablation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from papers.common.llm import get_llm
from papers.envs.household import TASKS, TaskSpec
from papers.p01_clin.method import ClinAgent, ClinTrial

SEEDS = tuple(range(20))

ID_TASKS = [t for t in TASKS if t.split == "id"]
OOD_TASKS = [t for t in TASKS if t.split == "ood"]


def _ep(llm_kind: str, seed: int, spec: TaskSpec,
        meta: list[str] | None = None, structured: bool = True,
        use_memory: bool = True) -> tuple[ClinAgent, list[ClinTrial]]:
    ag = ClinAgent(get_llm(llm_kind, seed), use_memory=use_memory,
                   structured=structured)
    ag.meta = list(meta or [])
    return ag, ag.run_episode(spec)


def _by_trial(trials: list[ClinTrial]) -> list[int]:
    """PT: sucesso acumulado na trial k (episódio já resolvido conta como 1).

    EN: cumulative success at trial k (an already-solved episode counts 1).
    """
    return [int(any(t.success for t in trials[:k + 1]))
            for k in range(4)]


@dataclass
class ExperimentResult:
    adapt: list[tuple[str, str, str, str, str, str, str]]
    curve_clin: list[float]      # sucesso médio nas trials 1..4
    curve_base: list[float]
    gen_env: list[tuple[str, str, str, str]]
    gen_task: list[tuple[str, str, str, str]]
    ablation: list[tuple[str, str]] = field(default_factory=list)


def run_experiment(seed: int, llm_kind: str) -> ExperimentResult:
    rows_adapt = []
    curve_c = [0.0] * 4
    curve_b = [0.0] * 4
    n_eps = 0
    for task in TASKS:
        cum_c = cum_b = p_c = p_b = n_c = n_b = 0
        for s in SEEDS:
            sd = seed * 100 + s
            _, tr_c = _ep(llm_kind, sd, task)
            _, tr_b = _ep(llm_kind, sd, task, use_memory=False)
            cum_c += any(x.success for x in tr_c)
            cum_b += any(x.success for x in tr_b)
            p_c += sum(x.success for x in tr_c[1:])
            n_c += len(tr_c) - 1
            p_b += sum(x.success for x in tr_b[1:])
            n_b += len(tr_b) - 1
            for k in range(4):
                curve_c[k] += _by_trial(tr_c)[k]
                curve_b[k] += _by_trial(tr_b)[k]
            n_eps += 1
        rows_adapt.append((task.task_id, task.task_type, task.variant,
                           f"{cum_c/len(SEEDS):.2f}", f"{cum_b/len(SEEDS):.2f}",
                           f"{p_c/max(n_c,1):.2f}", f"{p_b/max(n_b,1):.2f}"))
    curve_c = [v / n_eps for v in curve_c]
    curve_b = [v / n_eps for v in curve_b]

    # Gen-Env: treina no ID de mesmo tipo, testa no OOD com meta-memória.
    rows_ge = []
    for ood in OOD_TASKS:
        id_same = next(t for t in ID_TASKS if t.task_type == ood.task_type)
        cum_m = cum_b = 0
        for s in SEEDS:
            sd = seed * 100 + s
            trainer, _ = _ep(llm_kind, sd, id_same)
            meta = trainer.build_meta_memory(ood, "gen-env")
            _, tr = _ep(llm_kind, sd, ood, meta=meta)
            cum_m += any(x.success for x in tr)
            _, trb = _ep(llm_kind, sd, ood, meta=None, use_memory=False)
            cum_b += any(x.success for x in trb)
        rows_ge.append((ood.task_id, ood.task_type,
                        f"{cum_m/len(SEEDS):.2f}", f"{cum_b/len(SEEDS):.2f}"))

    # Gen-Task: treina em TODOS os ID de outros tipos, testa em cada OOD.
    rows_gt = []
    for ood in OOD_TASKS:
        others = [t for t in ID_TASKS if t.task_type != ood.task_type]
        cum_m = cum_b = 0
        for s in SEEDS:
            sd = seed * 100 + s
            trainer = ClinAgent(get_llm(llm_kind, sd))
            for t in others:
                trainer.run_episode(t)
            meta = trainer.build_meta_memory(ood, "gen-task")
            _, tr = _ep(llm_kind, sd, ood, meta=meta)
            cum_m += any(x.success for x in tr)
            _, trb = _ep(llm_kind, sd, ood, meta=None, use_memory=False)
            cum_b += any(x.success for x in trb)
        rows_gt.append((ood.task_id, ood.task_type,
                        f"{cum_m/len(SEEDS):.2f}", f"{cum_b/len(SEEDS):.2f}"))

    # Ablação structured vs free-form (Adapt, post-1).
    row_abl = []
    p_s = p_f = n_s = n_f = 0
    for task in TASKS:
        for s in SEEDS:
            sd = seed * 100 + s
            _, tr_s = _ep(llm_kind, sd, task, structured=True)
            _, tr_f = _ep(llm_kind, sd, task, structured=False)
            p_s += sum(x.success for x in tr_s[1:])
            n_s += len(tr_s) - 1
            p_f += sum(x.success for x in tr_f[1:])
            n_f += len(tr_f) - 1
    row_abl.append(("structured (causal abstractions)", f"{p_s/max(n_s,1):.2f}"))
    row_abl.append(("free-form advice", f"{p_f/max(n_f,1):.2f}"))
    return ExperimentResult(rows_adapt, curve_c, curve_b, rows_ge, rows_gt,
                            row_abl)


def render(res: ExperimentResult) -> str:
    n = len(SEEDS)
    out = ["# RESULTS — p01 CLIN\n",
           "Generated by `python -m papers.p01_clin.run --write-results` "
           "(fixed seed, MockLLM). Demonstrates the mechanism offline — does NOT "
           "reproduce the paper's numbers.\n",
           "## Adapt — success rate at trial k (the paper's main figure)\n",
           "Cumulative success after trial k, mean over 12 tasks "
           f"× {n} seeds.\n",
           "| agent | trial 1 | trial 2 | trial 3 | trial 4 |",
           "|---|---|---|---|---|",
           "| CLIN | " + " | ".join(f"{v:.2f}" for v in res.curve_clin) + " |",
           "| no memory | " + " | ".join(f"{v:.2f}" for v in res.curve_base)
           + " |",
           "", f"## Adapt — per task ({n} seeds)\n",
           "| task | type | variant | CLIN cum. | no-memory cum. | CLIN post-1 | base post-1 |",
           "|---|---|---|---|---|---|---|"]
    for r in res.adapt:
        out.append("| " + " | ".join(r) + " |")
    out += ["", "## Gen-Env — meta-memory to a new environment (same task type)\n",
            "| ood task | type | CLIN+meta | no-memory |", "|---|---|---|---|"]
    for r in res.gen_env:
        out.append("| " + " | ".join(r) + " |")
    out += ["", "## Gen-Task — meta-memory to a different task\n",
            "| ood task | type | CLIN+meta | no-memory |", "|---|---|---|---|"]
    for r in res.gen_task:
        out.append("| " + " | ".join(r) + " |")
    out += ["", "## Ablation (Adapt, post-1 success rate)\n",
            "| memory type | post-1 |", "|---|---|"]
    for r in res.ablation:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default=None, choices=["mock", "openai"])
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()
    res = run_experiment(args.seed, args.llm or "mock")
    text = render(res)
    print(text)
    if args.write_results:
        Path(__file__).with_name("RESULTS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
