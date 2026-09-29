"""PT: experimento do p14_memory_r1 — treina a política do manager com PPO
simplificado e GRPO (vantagem relativa ao grupo) sobre recompensa de exact
match; curva de aprendizado medida no BANCO DEPLOYADO (argmax) a cada época;
treino em personas de treino e avaliação em personas held-out; mean±std de
3 seeds. Estabilizadores: normalização de vantagem, bônus de entropia, lr
menor.

EN: p14_memory_r1 experiment — trains the manager policy with simplified
PPO and GRPO (group-relative advantage) on exact-match reward; learning
curve measured on the DEPLOYED (argmax) bank every epoch; train on train
personas, evaluate on held-out personas; mean±std over 3 seeds.
Stabilizers: advantage normalization, entropy bonus, lower lr.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np

from papers.common.conv_data import ConvSet, generate
from papers.common.conv_eval import (
    QTYPES,
    EvalResult,
    baseline_rows,
    render_acc_table,
)
from papers.common.llm import LLM, get_llm
from papers.common.reader import answer_with_evidence, judge_answer
from papers.common.utils import token_set
from papers.p14_memory_r1.method import (
    OPS,
    HeuristicManager,
    MemoryR1Manager,
    UntrainedManager,
    distill,
    exact_match,
    extract_facts,
    fact_features,
)

LR = 0.2
EPOCHS = 6
ENTROPY_BONUS = 0.01  # PT: bônus de entropia para evitar colapso prematuro.
                      # EN: entropy bonus against premature collapse.
N_SEEDS = 3


def _persona_qs(data: ConvSet, names: set[str]) -> list:
    """PT: perguntas cujo nome pertence ao conjunto. EN: questions whose
    name belongs to the set."""
    return [q for q in data.questions
            if any(n in q.question.split() or f"{n}'s" in q.question
                   for n in names)]


def _qa_reward(mgr, llm: LLM, questions) -> float:
    ok = 0
    for q in questions:
        ev = distill(mgr.retrieve(q.question), q.question, k=5)
        pred = answer_with_evidence(llm, q.question, ev)
        ok += max(judge_answer(llm, q.question, q.gold, pred),
                  exact_match(q.gold, pred))
    return ok / len(questions)


def _turns(data: ConvSet, names: set[str]) -> list:
    return [t for n in names for t in data.by_persona[n]]


def _facts(turns) -> list[str]:
    out: list[str] = []
    for t in turns:
        out += extract_facts(t.text)
    return out


def _op_rewards(mgr: MemoryR1Manager, fact: str, llm: LLM,
                questions) -> np.ndarray:
    """PT: para cada op, aplica numa cópia do manager e mede a recompensa QA
    local (perguntas relevantes ao fato) — o "grupo" GRPO e a fonte da
    vantagem do PPO simplificado. EN: apply each op on a manager copy and
    measure local QA reward — the GRPO group / PPO advantage source."""
    qs = [q for q in questions
          if token_set(fact) & (token_set(q.question) | token_set(q.gold))]
    subset = (qs or questions)[:5]
    r = np.zeros(len(OPS))
    for a in range(len(OPS)):
        if OPS[a] == "noop":
            snap = mgr
        else:
            snap = copy.deepcopy(mgr)
            snap._apply(a, fact)
        ok = 0
        for q in subset:
            ev = distill(snap.retrieve(q.question), q.question, k=5)
            pred = answer_with_evidence(llm, q.question, ev)
            ok += max(judge_answer(llm, q.question, q.gold, pred),
                      exact_match(q.gold, pred))
        r[a] = ok / len(subset)
    return r


def _deployed(mgr: MemoryR1Manager, facts: list[str],
              seed: int) -> MemoryR1Manager:
    """PT: reconstrói o banco com a política treinada em argmax.
    EN: rebuilds the bank with the trained policy in argmax mode."""
    d = MemoryR1Manager(seed=seed)
    d.policy.w = mgr.policy.w.copy()
    d.greedy = True
    for f in facts:
        d.ingest(f)
    return d


def train(mgr: MemoryR1Manager, train_facts: list[str], train_qs,
          all_facts: list[str], held_qs, llm: LLM, algo: str,
          seed: int) -> list[float]:
    """PT: laço de treino. GRPO: adv_a = (r_a - média)/std do grupo de 4 ops.
    PPO simplificado: adv clipada vs baseline móvel. Ambos com normalização
    de vantagem e bônus de entropia. A curva é medida no banco DEPLOYADO
    (argmax) sobre perguntas held-out.
    EN: training loop. GRPO: group-mean/std advantage. Simplified PPO:
    clipped advantage vs moving baseline. Both with advantage normalization
    and an entropy bonus. The curve is the DEPLOYED (argmax) bank on
    held-out questions."""
    curve: list[float] = []
    baseline = 0.0
    for _ep in range(EPOCHS):
        rng_t = random.Random(seed + _ep)
        for f in rng_t.sample(train_facts, min(40, len(train_facts))):
            sims = [mgr.memories[d] for d, _ in mgr._index.search(f, 10)
                    if d in mgr.memories]
            x = fact_features(f, sims)
            r = _op_rewards(mgr, f, llm, train_qs)
            if algo == "grpo":
                advs = r - float(r.mean())
            else:
                advs = np.clip(r - baseline, -1.0, 1.0)
                baseline = 0.9 * baseline + 0.1 * float(r.mean())
            sd = float(advs.std())
            if sd > 1e-6:  # PT: normalização de vantagem. EN: normalize.
                advs = advs / sd
            for a in range(len(OPS)):
                if advs[a] != 0.0:
                    mgr.policy.update(x, a, float(advs[a]), LR)
            # PT: bônus de entropia — puxa os logits para zero (exploração).
            # EN: entropy bonus — pulls logits toward zero (exploration).
            mgr.policy.w *= (1.0 - ENTROPY_BONUS)
            mgr._apply(int(np.argmax(r)), f)
        dep = _deployed(mgr, all_facts, seed)
        curve.append(_qa_reward(dep, llm, held_qs))
    return curve


def _qa_per_type(mgr, data: ConvSet, names: set[str], llm: LLM) -> EvalResult:
    by: dict[str, list[int]] = {qt: [] for qt in QTYPES}
    for q in _persona_qs(data, names):
        ev = distill(mgr.retrieve(q.question), q.question, k=5)
        pred = answer_with_evidence(llm, q.question, ev)
        by.setdefault(q.qtype, []).append(
            int(judge_answer(llm, q.question, q.gold, pred)))
    allv = [v for vs in by.values() for v in vs]
    return EvalResult("", sum(allv) / len(allv),
                      {qt: (sum(vs) / len(vs) if vs else 0.0)
                       for qt, vs in by.items()})


def experiment(seed: int, llm: LLM) -> tuple[list[EvalResult],
                                            dict[str, list[float]]]:
    data = generate(seed)
    names = data.personas()
    # PT: split treino/holdout por persona. EN: train/holdout persona split.
    train_names, held_names = set(names[:7]), set(names[7:])
    train_facts = _facts(_turns(data, train_names))
    train_qs = _persona_qs(data, train_names)
    held_qs = _persona_qs(data, held_names)
    all_facts = _facts(_turns(data, set(names)))

    curves: dict[str, list[float]] = {}
    rows: list[EvalResult] = []
    for label, algo in [("Memory-R1 (PPO)", "ppo"),
                        ("Memory-R1 (GRPO)", "grpo")]:
        vals: list[float] = []
        for s in range(N_SEEDS):
            mgr = MemoryR1Manager(seed=seed + s)
            c = train(mgr, train_facts, train_qs, all_facts, held_qs,
                      llm, algo, seed + s)
            vals.append(c[-1])
            if s == 0:
                curves[label] = c
        r = _qa_per_type(_deployed(mgr, all_facts, seed + N_SEEDS - 1),
                         data, held_names, llm)
        r.name = label
        r.overall = float(np.mean(vals))
        r.extra = f"{np.mean(vals):.2f}±{np.std(vals):.2f}"
        rows.append(r)
    for label, cls in [("Memory-R1 (heuristic)", HeuristicManager),
                       ("Memory-R1 (untrained)", UntrainedManager)]:
        mgr = cls(seed)
        for f in all_facts:
            mgr.ingest(f)
        r = _qa_per_type(mgr, data, held_names, llm)
        r.name = label
        rows.append(r)
    # PT: baselines avaliados só nas personas held-out (justo). Para
    # simplicidade avaliam nos turnos held-out. EN: baselines evaluated on
    # held-out personas only — evaluate on held-out turns.
    held_data = ConvSet(sessions=[], questions=_persona_qs(data, held_names))
    held_data.by_persona = {n: data.by_persona[n] for n in held_names}
    held_data.sessions = [list(held_data.by_persona[n]) for n in held_names]
    for b in baseline_rows(held_data, llm):
        rows.append(b)
    return rows, curves


def render(rows: list[EvalResult], curves: dict[str, list[float]]) -> str:
    lines = ["# RESULTS — p14 Memory-R1", "",
             "Generated by `python -m papers.p14_memory_r1.run --write-results` "
             "(fixed seed, MockLLM). Demonstrates the mechanism offline — "
             "does NOT reproduce the paper's numbers.", "",
             "## Learning curves — DEPLOYED (argmax) bank on held-out "
             "questions (seed 0)", "",
             "| epoch | " + " | ".join(curves) + " |", "|---|" + "---|" *
             len(curves)]
    n = max(len(c) for c in curves.values())
    for ep in range(n):
        lines.append(f"| {ep} | " + " | ".join(
            f"{c[ep]:.2f}" if ep < len(c) else "-" for c in curves.values())
            + " |")
    lines += ["", "## Held-out conversational QA (mean±std over 3 seeds "
              "for trained rows)", ""]
    lines += render_acc_table(rows, extra_col="mean±std")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default="mock")
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()
    llm = get_llm(args.llm, seed=args.seed)
    rows, curves = experiment(args.seed, llm)
    out = render(rows, curves)
    print(out)
    if args.write_results:
        Path(__file__).with_name("RESULTS.md").write_text(out,
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
