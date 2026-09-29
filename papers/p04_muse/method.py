"""MUSE (Valiente & Pilly) — implementação LLM (seção 4 do paper).

PT: Pré-deployment: roda o agente Reflexion (p02) nas tarefas in-distribution
para coletar dados e treina o Self-Assessment Model M_sa = embedder + MLP
g_η (sigmoid, BCE — Eq. 7) sobre concat(emb(τ_chunk, P^e), emb(I)), com
trajetórias segmentadas em chunks NÃO sobrepostos de 4 pares ação-observação
(Algoritmo 2). Deployment: a cada passo o World Model/Actor gera 5 rollouts
hipotéticos (temperatura 0.5), M_sa pontua cada um e o agente executa a 1ª ação
do melhor rollout (self-regulation, §4.1.3); reflexões em falhas; adaptação
online só de g_η com replay buffer.

SFT do actor e DPO do reflection LLM: construímos os datasets exatamente como o
paper define (SFT = episódios de sucesso; DPO: par positivo = falha em e_i e
sucesso em e_{i+1} após a reflexão; negativo = o inverso), salvos em JSONL —
com MockLLM não há pesos para treinar, então só geramos os dados.

EN: see PT. Builds the SFT/DPO datasets exactly as defined in the paper and
saves them as JSONL — with a MockLLM there are no weights to train.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM, task_prompt
from papers.envs.household import HouseholdEnv, TaskSpec, score_actions
from papers.p04_muse.prompts import first_action

CHUNK = 4           # PT: pares ação-observação por chunk. EN: pairs per chunk.
N_ROLLOUTS = 5      # PT: 5 rollouts hipotéticos. EN: 5 hypothetical rollouts.
FAIL_TTC = 100      # PT: penalidade de ttc em falha. EN: failure ttc penalty.


# ---------------------------------------------------------------------------
# Self-Assessment Model M_sa = embedder + MLP g_η (sigmoid, BCE)
# ---------------------------------------------------------------------------

class SelfAssessment:
    """PT: g_η logístico sobre concat(emb(τ_chunk+P^e), emb(I)).

    EN: logistic g_η over concat(emb(chunk+plan), emb(task)) — Eq. 6/7.
    """

    def __init__(self, embedder: Embedder | None = None, lr: float = 0.5) -> None:
        self.emb = embedder or HashingEmbedder()
        self.lr = lr
        self.w: np.ndarray | None = None
        self.b = 0.0

    def _x(self, traj_text: str, task: str) -> np.ndarray:
        e = self.emb.embed([traj_text, task])
        return np.concatenate([e[0], e[1]])

    def predict(self, traj_text: str, task: str) -> float:
        if self.w is None:
            return 0.5
        return float(1 / (1 + np.exp(-(self._x(traj_text, task) @ self.w + self.b))))

    def train_step(self, traj_text: str, task: str, y: float) -> float:
        x = self._x(traj_text, task)
        if self.w is None:
            self.w = np.zeros(x.shape[0], dtype=np.float64)
        p = 1 / (1 + np.exp(-(x @ self.w + self.b)))
        # PT: gradiente da BCE com sigmoid: (p − y)·x. EN: BCE+sigmoid gradient.
        self.w -= self.lr * (p - y) * x
        self.b -= self.lr * (p - y)
        return float(p)


def chunk_pairs(traj: list[tuple[str, str]], plan: str, task: str,
                success: bool) -> list[tuple[str, str, float]]:
    """PT: segmenta a trajetória em chunks de CHUNK pares (não sobrepostos),
    rotulados pelo desfecho do episódio — como no Algoritmo 2.

    EN: segments the trajectory into non-overlapping CHUNK-pair chunks labeled
    by the episode outcome — per Algorithm 2.
    """
    out = []
    for i in range(0, len(traj) - CHUNK + 1, CHUNK):
        txt = plan + " " + " ".join(f"{a} {o.split(' You are in')[0]}"
                                    for a, o in traj[i:i + CHUNK])
        out.append((txt, task, 1.0 if success else 0.0))
    return out


# ---------------------------------------------------------------------------
# Agente MUSE (deployment)
# ---------------------------------------------------------------------------

@dataclass
class MuseAgent:
    llm: LLM
    sa: SelfAssessment
    reflections: list[str] = field(default_factory=list)
    use_msa: bool = True      # ablação: False ≈ ReAct+reflexão sem M_sa
    replay: list[tuple[str, str, float]] = field(default_factory=list)

    def plan(self, task: str) -> str:
        return self.llm.complete(task_prompt(
            "muse.plan", f"Write a short plan.\nTask: {task}")).strip()

    def _rollout(self, spec: TaskSpec, obs: str, actions: list[str]) -> str:
        body = (
            "Imagine a plausible next-steps trajectory.\n"
            f"Task: {spec.description}\nObservation: {obs}\n"
            + ("Reflections:\n" + "\n".join(f"- {m}" for m in self.reflections)
               + "\n" if self.reflections else "")
            + "Valid actions:\n" + "\n".join(f"- {a}" for a in actions)
        )
        return self.llm.complete(task_prompt("muse.rollout", body),
                                 temperature=0.5)

    def act(self, spec: TaskSpec, obs: str, actions: list[str],
            plan: str, recent: list[str]) -> str:
        if not self.use_msa:
            return score_actions(spec.description, obs, actions,
                                 self.reflections, recent, random.Random(0))
        # PT: 5 rollouts → M_sa pontua → executa a 1ª ação do melhor.
        # EN: 5 rollouts → M_sa scores → execute best rollout's 1st action.
        best_a, best_p = None, -1.0
        for _ in range(N_ROLLOUTS):
            ro = self._rollout(spec, obs, actions)
            a = first_action(ro)
            if a not in actions:
                continue
            p = self.sa.predict(ro + " " + plan, spec.description)
            if p > best_p:
                best_p, best_a = p, a
        if best_a is None:
            best_a = score_actions(spec.description, obs, actions,
                                   self.reflections, recent, random.Random(0))
        return best_a

    def run_episode(self, spec: TaskSpec) -> tuple[bool, list[tuple[str, str]]]:
        env = HouseholdEnv(spec)
        obs = env.reset()
        plan = self.plan(spec.description)
        traj: list[tuple[str, str]] = []
        recent: list[str] = []
        while True:
            va = env.valid_actions()
            act = self.act(spec, obs, va, plan, recent)
            res = env.step(act)
            traj.append((act, res.observation))
            recent.append(act)
            obs = res.observation
            if res.done:
                break
        success = env.done
        # PT: reflexão em falhas (§4.1.3). EN: reflect on failures.
        if not success:
            from papers.p02_reflexion.prompts import reflection_body
            body = reflection_body(spec.description, traj, "failure",
                                   self.reflections)
            self.reflections = (self.reflections + [
                self.llm.complete(task_prompt("reflexion.reflect", body))])[-3:]
        # PT: dados novos entram no replay buffer p/ adaptação online de g_η.
        # EN: new data goes to the replay buffer for online g_η adaptation.
        self.replay += chunk_pairs(traj, plan, spec.description, success)
        return success, traj


def adapt_online(sa: SelfAssessment, replay: list, epochs: int = 3,
                 rng: random.Random | None = None) -> None:
    """PT: adaptação online de g_η com replay buffer (Algoritmo 4).

    EN: online adaptation of g_η only, over the replay buffer (Algorithm 4).
    """
    rng = rng or random.Random(0)
    for _ in range(epochs):
        for txt, task, y in replay:
            sa.train_step(txt, task, y)


# ---------------------------------------------------------------------------
# Datasets SFT / DPO (definições do paper; treino de pesos não roda offline)
# ---------------------------------------------------------------------------

def build_sft_dataset(episodes: list[tuple[TaskSpec, bool, list]]) -> list[dict]:
    """PT: SFT = episódios de sucesso. EN: SFT = successful episodes only."""
    return [{"task": s.task_id, "trajectory": [a for a, _ in traj]}
            for s, ok, traj in episodes if ok]


def build_dpo_dataset(episodes: list[tuple[TaskSpec, bool, list, str]]) -> list[dict]:
    """PT: par DPO (chosen, rejected) por tarefa: reflexão positiva = falha em
    e_i e sucesso em e_{i+1}; negativa = o inverso.

    EN: DPO pair per task: positive reflection = fail e_i then success e_{i+1};
    negative = the inverse.
    """
    by_task: dict[str, list] = {}
    for spec, ok, _traj, refl in episodes:
        by_task.setdefault(spec.task_id, []).append((ok, refl))
    out = []
    for task_id, eps in by_task.items():
        pos = neg = None
        for i in range(len(eps) - 1):
            if not eps[i][0] and eps[i + 1][0]:
                pos = eps[i][1]
            if eps[i][0] and not eps[i + 1][0]:
                neg = eps[i][1]
        if pos and neg:
            out.append({"task": task_id, "chosen": pos, "rejected": neg})
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Métricas / metrics
# ---------------------------------------------------------------------------

def auroc(scores: list[float], labels: list[int]) -> float:
    """PT: AUROC simples por pares comparáveis. EN: pairwise AUROC."""
    pos = [s for s, lbl in zip(scores, labels, strict=True) if lbl]
    neg = [s for s, lbl in zip(scores, labels, strict=True) if not lbl]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
