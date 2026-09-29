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
from pathlib import Path

from papers.common.llm import get_llm
from papers.envs.household import TASKS, run_episode
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


def collect_predeployment(seed: int, llm_kind: str, out_dir: Path) -> tuple:
    """PT: Reflexion nas tarefas ID → dados + datasets SFT/DPO em JSONL.

    EN: Reflexion on ID tasks → data + SFT/DPO datasets as JSONL.
    """
    sa = SelfAssessment()
    chunks: list[tuple[str, str, float]] = []
    episodes_meta: list[tuple] = []
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
    return sa, len(sft), len(dpo)


def run_muse(seed: int, llm_kind: str, sa: SelfAssessment,
             adapt: bool) -> dict:
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
            if ag.sa.w is not None and len(traj) >= 4:
                txt = " ".join(a + " " + o.split(" You are in")[0]
                               for a, o in traj[:4])
                sa_scores.append(ag.sa.predict(txt, task.description))
                sa_labels.append(int(ok))
    acc = None
    if sa_scores:
        acc = sum(int(s > 0.5) == lbl
                  for s, lbl in zip(sa_scores, sa_labels, strict=True)
                  ) / len(sa_scores)
    return {"success": wins / n_eps, "ttc": ttc / n_eps,
            "meta_acc": acc, "auroc2": auroc(sa_scores, sa_labels)}


def _clone_sa(sa: SelfAssessment) -> SelfAssessment:
    """PT: mesma g_η, replay novo. EN: same g_η weights, fresh replay."""
    c = SelfAssessment(sa.emb, sa.lr)
    c.w = None if sa.w is None else sa.w.copy()
    c.b = sa.b
    return c


def run_baselines(seed: int, llm_kind: str) -> dict:
    """PT: ReAct (sem memória) e Reflexion nos OOD. EN: baselines on OOD."""
    out = {}
    for name in ("ReAct", "Reflexion"):
        wins = ttc = n = 0
        for task in OOD_TASKS:
            for ep in range(ADAPT_EPS + TEST_EPS):
                sd = seed * 100 + ep
                if name == "ReAct":
                    res = run_episode(get_llm(llm_kind, sd), task)
                    ok, steps = res["success"], res["steps"]
                else:
                    ag = ReflexionAgent(get_llm(llm_kind, sd))
                    trials = ag.run_task(task)
                    ok = trials[-1].success
                    steps = trials[-1].steps
                if ep >= ADAPT_EPS:
                    wins += ok
                    ttc += steps if ok else FAIL_TTC
                    n += 1
        out[name] = {"success": wins / n, "ttc": ttc / n}
    return out


def run_experiment(seed: int, llm_kind: str, out_dir: Path) -> dict:
    sa, n_sft, n_dpo = collect_predeployment(seed, llm_kind, out_dir)
    res = {"n_sft": n_sft, "n_dpo": n_dpo}
    res["muse_noadapt"] = run_muse(seed, llm_kind, _clone_sa(sa), adapt=False)
    res["muse_adapt"] = run_muse(seed, llm_kind, sa, adapt=True)
    res.update(run_baselines(seed, llm_kind))
    return res


def render(res: dict) -> str:
    rows = [
        ("ReAct (no memory)", res["ReAct"]["success"], res["ReAct"]["ttc"], "—", "—"),
        ("Reflexion", res["Reflexion"]["success"], res["Reflexion"]["ttc"], "—", "—"),
        ("MUSE (no adaptation)", res["muse_noadapt"]["success"],
         res["muse_noadapt"]["ttc"],
         (f"{res['muse_noadapt']['meta_acc']:.2f}"
          if res["muse_noadapt"]["meta_acc"] is not None else "—"),
         f"{res['muse_noadapt']['auroc2']:.2f}"),
        ("MUSE (5 adapt + 5 test)", res["muse_adapt"]["success"],
         res["muse_adapt"]["ttc"],
         (f"{res['muse_adapt']['meta_acc']:.2f}"
          if res['muse_adapt']['meta_acc'] is not None else "—"),
         f"{res['muse_adapt']['auroc2']:.2f}"),
    ]
    out = ["# RESULTS — p04 MUSE (LLM implementation)\n",
           "Generated by `python -m papers.p04_muse.run --write-results` "
           "(fixed seed, MockLLM). Demonstrates the mechanism offline — does NOT "
           "reproduce the paper's numbers.\n",
           f"SFT dataset: {res['n_sft']} episodes; DPO pairs: {res['n_dpo']} "
           "(saved as sft.jsonl / dpo.jsonl; weight training does not run "
           "offline — there are no weights with MockLLM).\n",
           "| agent | success | time-to-completion (fail=100) | metacog. acc. | AUROC2 |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
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
