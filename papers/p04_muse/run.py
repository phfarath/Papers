"""MUSE — experimento executável / runnable experiment.

PT: `python -m papers.p04_muse.run [--seed 0] [--llm mock|openai]
[--write-results]`.

Protocolo (seção 4 do paper):
1. Pré-deployment: agente Reflexion (p02) coleta trajetórias nas tarefas ID →
   treina M_sa (embedder + g_η BCE) → grava sft.jsonl e dpo.jsonl.
2. Deployment nas tarefas OOD: 5 episódios de adaptação (atualiza só g_η) +
   5 de teste, vs baseline sem adaptação, ReAct e Reflexion.
Métricas: taxa de sucesso, time-to-completion (falha = 100 passos), acurácia
metacognitiva e AUROC2 do M_sa.

EN: see PT — pre-deployment data collection with Reflexion, then deployment on
OOD tasks (5 adaptation + 5 test episodes) vs no-adaptation, ReAct, Reflexion.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from papers.common.llm import get_llm
from papers.envs.household import TASKS, TaskSpec, run_episode
from papers.p02_reflexion.method import ReflexionAgent
from papers.p04_muse.method import (
    FAIL_TTC,
    MuseAgent,
    SelfAssessment,
    adapt_online,
    auroc,
    build_dpo_dataset,
    build_sft_dataset,
    chunk_pairs,
    write_jsonl,
)

ID_TASKS = [t for t in TASKS if t.split == "id"]
OOD_TASKS = [t for t in TASKS if t.split == "ood"]
TRAIN_SEEDS = (0, 1)
ADAPT_EPS, TEST_EPS = 5, 5


@dataclass
class Metrics:
    """PT: métricas de um agente no deployment. EN: deployment metrics."""

    success: float
    ttc: float
    meta_acc: float | None = None
    auroc2: float = float("nan")


@dataclass
class PredeployResult:
    sa: SelfAssessment
    n_sft: int
    n_dpo: int


@dataclass
class ExperimentResult:
    n_sft: int
    n_dpo: int
    muse_noadapt: Metrics
    muse_adapt: Metrics
    baselines: dict[str, Metrics]


def collect_predeployment(seed: int, llm_kind: str, out_dir: Path
                          ) -> PredeployResult:
    """PT: Reflexion nas tarefas ID → dados + datasets SFT/DPO em JSONL.

    EN: Reflexion on ID tasks → data + SFT/DPO datasets as JSONL.
    """
    sa = SelfAssessment()
    chunks: list[tuple[str, str, float]] = []
    episodes_meta: list[tuple[TaskSpec, bool, list[tuple[str, str]], str]] = []
    for task in ID_TASKS:
        for s in TRAIN_SEEDS:
            sd = seed * 100 + s
            ag = ReflexionAgent(get_llm(llm_kind, sd))
            trials = ag.run_task(task)
            # PT: reflexão vigente antes de cada trial i>0.
            # EN: reflection in force before each trial i>0.
            refl_for = [ag.mem[min(i, len(ag.mem) - 1)] if ag.mem else ""
                        for i in range(len(trials))]
            for tr, refl in zip(trials, refl_for, strict=True):
                episodes_meta.append((task, tr.success, tr.trajectory, refl))
                plan = f"1. find item; 2. take it; 3. finish {task.task_type}."
                for ch in chunk_pairs(tr.trajectory, plan, task.description,
                                      tr.success):
                    chunks.append(ch)
    # PT: várias épocas sobre o replay (BCE) para separar sucesso de falha.
    # EN: several replay epochs (BCE) to separate success from failure.
    for _ in range(20):
        for ch in chunks:
            sa.train_step(*ch)
    sft = build_sft_dataset([(s, ok, tr) for s, ok, tr, _ in episodes_meta])
    dpo = build_dpo_dataset(episodes_meta)
    write_jsonl(sft, out_dir / "sft.jsonl")
    write_jsonl(dpo, out_dir / "dpo.jsonl")
    return PredeployResult(sa, len(sft), len(dpo))


def run_muse(seed: int, llm_kind: str, sa: SelfAssessment,
             adapt: bool) -> Metrics:
    """PT: deployment OOD. EN: OOD deployment."""
    wins = ttc = 0
    sa_scores: list[float] = []
    sa_labels: list[int] = []
    n_eps = 0
    for task in OOD_TASKS:
        ag = MuseAgent(get_llm(llm_kind, seed), SelfAssessment()
                       if not adapt else _clone_sa(sa))
        # fase de adaptação (só atualiza g_η; falhas também geram reflexões)
        for ep in range(ADAPT_EPS):
            ok, traj = ag.run_episode(task)
            if adapt:
                adapt_online(ag.sa, ag.replay, epochs=1,
                             rng=random.Random(ep))
        # fase de teste
        for _ep in range(TEST_EPS):
            ok, traj = ag.run_episode(task)
            wins += ok
            ttc += len(traj) if ok else FAIL_TTC
            n_eps += 1
            if ag.sa.w2 is not None and len(traj) >= 4:
                txt = " ".join(a + " " + o.split(" You are in")[0]
                               for a, o in traj[:4])
                sa_scores.append(ag.sa.predict(txt, task.description))
                sa_labels.append(int(ok))
    acc = None
    if sa_scores:
        acc = sum(int(s > 0.5) == lbl
                  for s, lbl in zip(sa_scores, sa_labels, strict=True)
                  ) / len(sa_scores)
    return Metrics(wins / n_eps, ttc / n_eps, acc, auroc(sa_scores, sa_labels))


def _clone_sa(sa: SelfAssessment) -> SelfAssessment:
    """PT: mesma g_η, replay novo. EN: same g_η weights, fresh replay."""
    c = SelfAssessment(sa.emb, sa.lr, sa.hidden)
    if sa.w1 is not None:
        c.w1 = sa.w1.copy()
        c.b1 = None if sa.b1 is None else sa.b1.copy()
        c.w2 = None if sa.w2 is None else sa.w2.copy()
        c.b2 = sa.b2
    return c


def run_baselines(seed: int, llm_kind: str) -> dict[str, Metrics]:
    """PT: ReAct (sem memória) e Reflexion nos OOD. EN: baselines on OOD."""
    out: dict[str, Metrics] = {}
    for name in ("ReAct", "Reflexion"):
        wins = ttc = n = 0
        for task in OOD_TASKS:
            for ep in range(ADAPT_EPS + TEST_EPS):
                sd = seed * 100 + ep
                if name == "ReAct":
                    res = run_episode(get_llm(llm_kind, sd), task)
                    ok, steps = res.success, res.steps
                else:
                    ag = ReflexionAgent(get_llm(llm_kind, sd))
                    trials = ag.run_task(task)
                    ok = trials[-1].success
                    steps = trials[-1].steps
                if ep >= ADAPT_EPS:
                    wins += ok
                    ttc += steps if ok else FAIL_TTC
                    n += 1
        out[name] = Metrics(wins / n, ttc / n)
    return out


def run_experiment(seed: int, llm_kind: str, out_dir: Path) -> ExperimentResult:
    pre = collect_predeployment(seed, llm_kind, out_dir)
    return ExperimentResult(
        pre.n_sft, pre.n_dpo,
        muse_noadapt=run_muse(seed, llm_kind, _clone_sa(pre.sa), adapt=False),
        muse_adapt=run_muse(seed, llm_kind, pre.sa, adapt=True),
        baselines=run_baselines(seed, llm_kind),
    )


def _fmt2(x: float) -> str:
    """PT: 2 casas decimais; nan → n/a. EN: 2 decimals; nan → n/a."""
    return "n/a" if x != x else f"{x:.2f}"


def render(res: ExperimentResult) -> str:
    def row(name: str, m: Metrics) -> str:
        acc = f"{m.meta_acc:.2f}" if m.meta_acc is not None else "—"
        return f"| {name} | {m.success:.2f} | {m.ttc:.2f} | {acc} | {_fmt2(m.auroc2)} |"

    out = ["# RESULTS — p04 MUSE (LLM implementation)\n",
           "Generated by `python -m papers.p04_muse.run --write-results` "
           "(fixed seed, MockLLM). Demonstrates the mechanism offline — does NOT "
           "reproduce the paper's numbers.\n",
           f"SFT dataset: {res.n_sft} episodes; DPO pairs: {res.n_dpo} "
           "(saved as sft.jsonl / dpo.jsonl; weight training does not run "
           "offline — there are no weights with MockLLM).\n",
           "| agent | success | time-to-completion (fail=100) | metacog. acc. | AUROC2 |",
           "|---|---|---|---|---|",
           row("ReAct (no memory)", res.baselines["ReAct"]),
           row("Reflexion", res.baselines["Reflexion"]),
           row("MUSE (no adaptation)", res.muse_noadapt),
           row("MUSE (5 adapt + 5 test)", res.muse_adapt),
           "",
           "\\* AUROC2 shows `n/a` when the run's labels are single-class "
           "(e.g. the no-adaptation agent either always fails or always "
           "succeeds, so no positive/negative pair exists to rank)."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default=None, choices=["mock", "openai"])
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()
    out_dir = Path(__file__).parent
    res = run_experiment(args.seed, args.llm or "mock", out_dir)
    text = render(res)
    print(text)
    if args.write_results:
        (out_dir / "RESULTS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
