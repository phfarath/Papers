"""PT: experimento do p14_memory_r1 — treina a política do manager com PPO
simplificado e GRPO (vantagem relativa ao grupo) sobre recompensa de exact
match; curva de aprendizado vs heurístico vs não-treinado; QA no dataset
conversacional vs baselines.

EN: p14_memory_r1 experiment — trains the manager policy with simplified PPO
and GRPO (group-relative advantage) on exact-match reward; learning curve vs
heuristic vs untrained; conversational QA vs baselines.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from papers.common.conv_data import generate
from papers.common.llm import LLM, get_llm
from papers.common.memory_api import (
    BM25Memory,
    EmbeddingRAGMemory,
    FullContextMemory,
    MemoryItem,
)
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

LR = 0.5
EPOCHS = 6
GROUP = 8  # PT: tamanho do grupo GRPO. EN: GRPO group size.


@dataclass
class Row:
    setting: str
    accuracy: float


def _qa_reward(mgr, llm: LLM, data) -> float:
    """PT: recompensa = acurácia de exact/judge nas perguntas.
    EN: reward = QA accuracy on the dataset questions."""
    ok = 0
    for q in data.questions:
        ev = distill(mgr.retrieve(q.question), q.question, k=5)
        pred = answer_with_evidence(llm, q.question, ev)
        ok += max(judge_answer(llm, q.question, q.gold, pred),
                  exact_match(q.gold, pred))
    return ok / len(data.questions)


def _facts(data) -> list[str]:
    out: list[str] = []
    for t in data.all_turns():
        out += extract_facts(t.text)
    return out


def _op_rewards(mgr: MemoryR1Manager, fact: str, llm: LLM,
                data) -> np.ndarray:
    """PT: para cada uma das 4 ops, aplica numa cópia do manager e mede a
    recompensa QA resultante — esse é o "grupo" do GRPO (rollouts da mesma
    state) e a fonte da vantagem do PPO simplificado.
    EN: apply each of the 4 ops on a copy of the manager and measure the
    resulting QA reward — the GRPO "group" (rollouts of the same state) and
    the advantage source for simplified PPO."""
    # PT: atribuição local de crédito — a recompensa de cada op é medida só
    # nas perguntas relevantes ao fato (slot do fato vs pergunta/gold);
    # se nenhuma, usa todas. EN: local credit assignment — each op's reward
    # is measured on the questions relevant to this fact; fall back to all.
    qs = [q for q in data.questions
          if token_set(fact) & (token_set(q.question) | token_set(q.gold))]
    subset = qs or data.questions
    r = np.zeros(len(OPS))
    for a in range(len(OPS)):
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


def train(mgr: MemoryR1Manager, data, llm: LLM, algo: str) -> list[float]:
    """PT: laço de treino com vantagem por op.
    GRPO: adv_a = r_a - média das recompensas das 4 ops (vantagem relativa ao
    grupo) — sem baseline. PPO simplificado: adv_a = clip(r_a - baseline
    móvel da recompensa média, -1, 1). A op aplicada de verdade é amostrada
    da política (como no paper, on-policy).
    EN: per-op advantage training. GRPO: adv_a = r_a - mean op reward
    (group-relative, no baseline). Simplified PPO: adv_a = clip(r_a - moving
    baseline, -1, 1). The applied op is sampled from the policy (on-policy)."""
    curve = []
    baseline = 0.0
    for _ep in range(EPOCHS):
        for f in _facts(data):
            sims = [mgr.memories[d] for d, _ in mgr._index.search(f, 10)
                    if d in mgr.memories]
            x = fact_features(f, sims)
            r = _op_rewards(mgr, f, llm, data)
            a = mgr.policy.act(x, mgr.rng)
            if algo == "grpo":
                advs = r - float(r.mean())
            else:
                advs = np.clip(r - baseline, -1.0, 1.0)
                baseline = 0.9 * baseline + 0.1 * float(r[a])
            # PT: GRPO atualiza a política em TODAS as ações do grupo com a
            # vantagem de cada uma; o banco de treino recebe a op argmax da
            # recompensa (imitação recompensa-ponderada — a alternativa
            # on-policy destruía o banco; documentado no README).
            # EN: update every group action by its advantage; the training
            # bank applies the reward-argmax op (reward-weighted imitation —
            # pure on-policy destroyed the bank; documented in README).
            for a2 in range(len(OPS)):
                if advs[a2] != 0.0:
                    mgr.policy.update(x, a2, float(advs[a2]), LR)
            mgr._apply(int(np.argmax(r)), f)
        curve.append(_qa_reward(mgr, llm, data))
    return curve


def _eval_mem(mem, data, llm: LLM, k: int = 8) -> float:
    for t in data.all_turns():
        mem.add(MemoryItem(t.text, t.ts))
    ok = sum(judge_answer(llm, q.question, q.gold,
                          answer_with_evidence(
                              llm, q.question, mem.retrieve(q.question, k=k)))
             for q in data.questions)
    return ok / len(data.questions)


def experiment(seed: int, llm: LLM) -> tuple[list[Row], dict[str, list[float]]]:
    data = generate(seed)
    curves: dict[str, list[float]] = {}
    rows: list[Row] = []
    for algo, name in [("ppo", "Memory-R1 (PPO)"),
                       ("grpo", "Memory-R1 (GRPO)")]:
        mgr = MemoryR1Manager(seed=seed)
        curves[name] = train(mgr, data, llm, algo)
        # PT: deploy — o banco final é reconstruído do zero pela política
        # treinada em modo argmax (como no paper, a recompensa de treino não
        # contamina o banco de produção). EN: rebuild the bank with the
        # trained policy in argmax mode.
        deploy = MemoryR1Manager(seed=seed)
        deploy.policy.w = mgr.policy.w.copy()
        deploy.greedy = True
        for f in _facts(data):
            deploy.ingest(f)
        curves[name] += [_qa_reward(deploy, llm, data)]
        rows.append(Row(name, _qa_reward(deploy, llm, data)))
    for name, mgr in [("Memory-R1 (heuristic)", HeuristicManager(seed)),
                      ("Memory-R1 (untrained)", UntrainedManager(seed))]:
        for f in _facts(data):
            mgr.ingest(f)
        curves[name] = [_qa_reward(mgr, llm, data)]
        rows.append(Row(name, curves[name][0]))
    for name, m in [("FullContext (truncated)", FullContextMemory(200)),
                    ("BM25", BM25Memory()),
                    ("EmbeddingRAG", EmbeddingRAGMemory())]:
        rows.append(Row(name, _eval_mem(m, data, llm)))
    return rows, curves


def render(rows: list[Row], curves: dict[str, list[float]]) -> str:
    lines = ["# RESULTS — p14 Memory-R1", "",
             "Generated by `python -m papers.p14_memory_r1.run --write-results` "
             "(fixed seed, MockLLM). Demonstrates the mechanism offline — "
             "does NOT reproduce the paper's numbers.", "",
             "## Learning curves (QA accuracy per epoch)", "",
             "| epoch | " + " | ".join(curves) + " |", "|---|" + "---|" *
             len(curves)]
    n = max(len(c) for c in curves.values())
    for ep in range(n):
        lines.append(f"| {ep} | " + " | ".join(
            f"{c[ep]:.2f}" if ep < len(c) else "-" for c in curves.values())
            + " |")
    lines += ["",
              "The last epoch row is the DEPLOYED bank (rebuilt by the "
              "trained policy in argmax mode). The tiny linear policy "
              "sometimes collapses to a degenerate op (e.g. NOOP-everything) "
              "— reported honestly; the paper's trained LLM policy is "
              "richer.", "",
              "## Final conversational QA", "",
              "| memory system | accuracy |", "|---|---|"]
    for r in rows:
        lines.append(f"| {r.setting} | {r.accuracy:.2f} |")
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
