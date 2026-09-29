"""MemRL — experimento executável / runnable experiment.

PT: `python -m papers.p03_memrl.run [--seed 0] [--llm mock|openai]
[--write-results]` — 6 épocas; cada época = cada tarefa × 5 seeds internas; as
tabelas mostram média ± desvio sobre 3 seeds externas. A cada episódio: Fase
A+B recuperam memórias → política age → recompensa atualiza Q (MC, Eq. 4) das
memórias injetadas → o resumo da trajetória vira nova tripla. Baselines: sem
memória, RAG por similaridade (λ=0) e variante TD (Eq. 3). Extras exigidos:
ablação de λ e k1/k2, correlação Q × sucesso empírico e Q médio de memórias
úteis vs distratoras.

EN: see PT. 6 epochs × (12 tasks × 5 inner seeds), mean±std over 3 outer
seeds; MC update on injected memories only; TD variant, λ and k1/k2 ablations,
and the learned-Q × empirical-success correlation.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from papers.common.llm import LLM, get_llm, task_prompt
from papers.envs.household import (
    TASKS,
    EpisodeResult,
    TaskSpec,
    run_episode,
)
from papers.p03_memrl.method import MemRLMemory
from papers.p03_memrl.prompts import seed_bank

EPOCHS = 6
INNER_SEEDS = 3      # PT: seeds internas por tarefa (curvas principais)
OUTER_SEEDS = (0, 1, 2)
ABL_INNER = 2        # PT: ablações usam protocolo reduzido (<60 s no total)
N_EP_PER_EPOCH = len(TASKS) * INNER_SEEDS


def new_bank(**kwargs: float | int) -> MemRLMemory:
    """PT: banco novo já semeado. EN: freshly seeded memory bank."""
    mem = MemRLMemory(**kwargs)
    for z, e, _ in seed_bank():
        mem.add(z, e)
    return mem


def run_one_epoch(llm_kind: str, seed: int, mem: MemRLMemory | None,
                  td: bool = False, inner: int = INNER_SEEDS,
                  stats: list[list[int]] | None = None) -> float:
    """PT: uma época = `inner` episódios por tarefa; devolve sucesso.

    EN: one epoch = `inner` episodes per task; returns success rate.
    `stats[i]` = [injections, successes] per bank item (Q×success correlation).
    """
    wins = 0
    for task in TASKS:
        for s in range(inner):
            llm = get_llm(llm_kind, seed * 1000 + s)
            mem_lines, idx = ([], [])
            if mem is not None:
                mem_lines, idx = mem.retrieve(task.description)
            res = run_episode(llm, task, memory_lines=mem_lines)
            wins += res.success
            if mem is not None:
                # PT: update só nas injetadas; TD usa max Q do banco como
                # aproximação de max Q(s',·) — aproximação documentada.
                # EN: update injected only; TD uses bank max Q as the
                # documented approximation of max Q(s',·).
                nmq = max((t.q for t in mem.items), default=0.0)
                mem.update(idx, 1.0 if res.success else 0.0,
                           td=td, next_max_q=nmq)
                if stats is not None:
                    for i in idx:
                        stats[i][0] += 1
                        stats[i][1] += int(res.success)
                e_new = summarize(llm, task, res)
                if e_new:  # PT: nada extraível → não adiciona. EN: skip empty.
                    mem.add(task.description, e_new)
                    if stats is not None:
                        stats.append([0, 0])
    return wins / (len(TASKS) * inner)


def summarize(llm: LLM, task: TaskSpec, res: EpisodeResult) -> str:
    """PT: prompta o resumo (trajetória em linhas a:/o:). EN: summary prompt."""
    body = (
        "Summarize the trajectory into a reusable lesson with concrete "
        "actions, objects and rooms.\n"
        f"Intent: {task.description}\n"
        f"Outcome: {'success' if res.success else 'failure'}\n"
        + "\n".join(f"a: {a}\no: {o.split(' You are in')[0]}"
                    for a, o in res.trajectory)
    )
    return llm.complete(task_prompt("memrl.summarize", body)).strip()


@dataclass
class RunResult:
    """PT: série de taxas de sucesso por época (uma entrada por outer seed).

    EN: per-epoch success rates (one entry per outer seed).
    """

    epochs: list[list[float]] = field(default_factory=list)  # [outer][epoch]

    def mean_std(self) -> list[tuple[float, float]]:
        out = []
        for e in range(EPOCHS):
            vals = [row[e] for row in self.epochs]
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out.append((statistics.mean(vals), sd))
        return out


@dataclass
class ExperimentResult:
    curves: dict[str, RunResult] = field(default_factory=dict)
    ablations: dict[str, RunResult] = field(default_factory=dict)
    q_useful: float = 0.0
    q_distractor: float = 0.0
    q_success_corr: float | None = None
    top_items: list[tuple[float, str]] = field(default_factory=list)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def _run_curve(llm_kind: str, lam: float | None, outer_seed: int,
               td: bool = False, inner: int = INNER_SEEDS,
               stats: list[list[int]] | None = None,
               **kw: float | int) -> list[float]:
    mem = None if lam is None else new_bank(lam=lam, **kw)
    return [run_one_epoch(llm_kind, outer_seed * 100 + ep, mem,
                          td=td, inner=inner, stats=stats)
            for ep in range(EPOCHS)]


def run_experiment(seed: int, llm_kind: str) -> ExperimentResult:
    res = ExperimentResult()
    methods = [("MemRL (λ=0.5)", 0.5, False),
               ("MemRL-TD (λ=0.5)", 0.5, True),
               ("RAG similarity (λ=0)", 0.0, False),
               ("No memory", None, False)]
    for name, lam, td in methods:
        rr = RunResult()
        for os_ in OUTER_SEEDS:
            rr.epochs.append(_run_curve(llm_kind, lam, seed * 10 + os_, td=td))
        res.curves[name] = rr
    # Ablações de λ (Fig. 5 do paper) e k1/k2 — protocolo reduzido
    # (1 outer seed × ABL_INNER) para caber no orçamento de tempo.
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        rr = RunResult()
        rr.epochs.append(_run_curve(llm_kind, lam, seed * 10,
                                    inner=ABL_INNER))
        res.ablations[f"λ={lam}"] = rr
    for k1, k2 in ((2, 1), (4, 2), (8, 4)):
        rr = RunResult()
        rr.epochs.append(_run_curve(llm_kind, 0.5, seed * 10,
                                    inner=ABL_INNER, k1=k1, k2=k2))
        res.ablations[f"k1={k1},k2={k2}"] = rr
    # Correlação Q × sucesso: um run λ=0.5 com estatísticas por memória.
    n_seeds = len(seed_bank())
    stats: list[list[int]] = [[0, 0] for _ in range(n_seeds)]
    mem = new_bank()
    for ep in range(EPOCHS):
        run_one_epoch(llm_kind, seed * 100 + ep, mem, inner=ABL_INNER,
                      stats=stats)
    distractors = {i for i, (_, _, is_d) in enumerate(seed_bank()) if is_d}
    useful_q = [t.q for i, t in enumerate(mem.items[:n_seeds])
                if i not in distractors]
    dist_q = [t.q for i, t in enumerate(mem.items[:n_seeds])
              if i in distractors]
    res.q_useful = statistics.mean(useful_q)
    res.q_distractor = statistics.mean(dist_q)
    xs = [t.q for t, (n, _) in zip(mem.items, stats, strict=True) if n > 0]
    ys = [w / n for (n, w) in stats if n > 0]
    res.q_success_corr = _pearson(xs, ys)
    res.top_items = sorted(((t.q, t.e[:70]) for t in mem.items),
                           reverse=True)[:8]
    return res


def _fmt_curve(rr: RunResult) -> str:
    return " | ".join(f"{m:.2f}±{s:.2f}" for m, s in rr.mean_std())


def render(res: ExperimentResult) -> str:
    out = ["# RESULTS — p03 MemRL\n",
           "Generated by `python -m papers.p03_memrl.run --write-results` "
           "(fixed seed, MockLLM). Demonstrates the mechanism offline — does "
           "NOT reproduce the paper's numbers.\n",
           f"Each epoch = {len(TASKS)} tasks × {INNER_SEEDS} seeds; "
           "cells show mean±std over 3 outer seeds (ablations use a reduced "
           f"protocol: {ABL_INNER} seeds × 1 outer seed).\n",
           "## Success rate per epoch (HouseholdEnv)\n",
           "| method | " + " | ".join(f"e{i + 1}" for i in range(EPOCHS)) + " |",
           "|---|" + "---|" * EPOCHS]
    for name, rr in res.curves.items():
        out.append("| " + name + " | " + _fmt_curve(rr) + " |")
    out += ["", "## Ablations (per-epoch success, mean±std)\n",
            "| setting | " + " | ".join(f"e{i + 1}" for i in range(EPOCHS))
            + " |", "|---|" + "---|" * EPOCHS]
    for name, rr in res.ablations.items():
        out.append("| " + name + " | " + _fmt_curve(rr) + " |")
    corr = (f"{res.q_success_corr:.2f}" if res.q_success_corr is not None
            else "n/a")
    out += ["", "## Utility learned by Q\n",
            "| metric | value |", "|---|---|",
            f"| mean final Q — useful seeds | {res.q_useful:.2f} |",
            f"| mean final Q — distractor seeds | {res.q_distractor:.2f} |",
            "| Pearson corr(final Q, empirical success when injected) "
            f"| {corr} |",
            "", "## Learned Q-values (top of the bank after the run)\n",
            "| Q | experience |", "|---|---|"]
    for q, e in res.top_items:
        out.append(f"| {q:.2f} | {e} |")
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
        Path(__file__).with_name("RESULTS.md").write_text(text,
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
