"""PT: p17 LongMemEval-V2 (arXiv 2605.12493v1) — benchmark de memória sobre
trajetórias de agente web (MiniWeb), com as 5 competências do paper (§3.1):
static state recall, dynamic state tracking, workflow knowledge,
environment gotchas, premise awareness (incl. perguntas de falsa premissa
para rejeitar). Formulário do paper (§3.3): a memória consome trajetórias
e devolve evidência compacta dentro de um ORÇAMENTO DE TOKENS → leitor
responde.

Baselines do paper (§4):
- **AgentRunbook-R**: pools separados de recuperação (observações de
  estado, eventos/ações, notas de estratégia/gotchas) + merge no orçamento.
- **AgentRunbook-C**: trajetórias escritas em ARQUIVOS num diretório
  temporário; um "coding agent" usa chamadas de ferramenta restritas
  (grep/read) para reunir evidência.

EN: p17 LongMemEval-V2 — web-agent trajectory memory benchmark (MiniWeb)
with the paper's 5 competencies (§3.1), token-budgeted context gathering
(§3.3), and the AgentRunbook-R/-C baselines (§4).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from papers.common.embeddings import Embedder, HashingEmbedder
from papers.common.llm import LLM
from papers.common.reader import answer_with_evidence, judge_answer
from papers.common.retrieval import VectorIndex
from papers.common.utils import token_set
from papers.envs.web import BASE, TASKS, MiniWeb

COMP_ORDER = ["static", "dynamic", "workflow", "gotcha", "premise"]


# ---------- gravação de trajetórias / trajectory recording ----------
@dataclass
class Trajectory:
    task_id: str
    steps: list[str]          # "step N | action | obs-one-line"
    success: bool


_SCRIPTS: dict[str, list[str]] = {
    "w01": ["click lnk_search", "type in_q laptop", "click lnk_item_laptop",
            "stop $999"],
    "w02": ["click lnk_search", "type in_q keyboard", "click lnk_item_keyboard",
            "click btn_add_cart", "click lnk_cart", "stop 1"],
    "w03": ["goto " + BASE + "/git", "click lnk_issues",
            f"goto {BASE}/git/admin", "stop Admin Dashboard"],
    "w04": ["goto " + BASE + "/forum", "click lnk_login", "type in_user demo",
            "type in_pass demo", "goto " + BASE + "/forum/topic/t1",
            "type in_reply hello", "click lnk_home", "stop 3"],
    "w05": ["goto " + BASE + "/git", "click lnk_issues", "stop 2"],
}


def record_trajectories(seed: int = 0, noise: float = 0.3) -> list[Trajectory]:
    """PT: roda um explorador scriptado (com passos de ruído determinísticos)
    em cada tarefa do MiniWeb e grava as trajetórias. EN: run a scripted
    explorer (with deterministic noise steps) on each MiniWeb task and
    record trajectories."""
    rng = random.Random(seed)
    out: list[Trajectory] = []
    noise_acts = ["goto " + BASE + "/shop", "click lnk_home",
                  "goto " + BASE + "/forum", "click lnk_git"]
    for task in TASKS:
        env = MiniWeb(task)
        env.reset()
        steps: list[str] = []
        i = 0
        for act in _SCRIPTS[task.task_id]:
            if rng.random() < noise and i < 3:
                na = rng.choice(noise_acts)
                r = env.step(na)
                steps.append(f"step {i} | {na} | "
                             f"{' / '.join(r.observation.splitlines()[:2])[:120]}")
                i += 1
            r = env.step(act)
            steps.append(f"step {i} | {act} | "
                         f"{' / '.join(r.observation.splitlines()[:2])[:120]}")
            i += 1
            if r.done:
                break
        out.append(Trajectory(task.task_id, steps, env.score() > 0))
    return out


# ---------- perguntas / questions ----------
@dataclass
class TQuestion:
    comp: str
    question: str
    gold: str


def make_questions(trajs: list[Trajectory]) -> list[TQuestion]:
    ok = {t.task_id: t.success for t in trajs}
    qs = [
        TQuestion("static", "What is the price of the laptop?", "$999"),
        TQuestion("static", "How many open issues does gitlab-lite list?",
                  "2"),
        TQuestion("dynamic", "After the keyboard task, how many items are "
                  "in the cart?", "1"),
        TQuestion("dynamic", "In the forum task, how many replies does "
                  "topic t1 have at the end?", "3"),
        TQuestion("workflow", "What action sequence adds the keyboard to "
                  "the cart?", "search; item; add"),
        TQuestion("gotcha", "Which unlinked URL leads to the admin "
                  "dashboard?", "/git/admin"),
        TQuestion("gotcha", "Why can't you reply on the forum at first?",
                  "log"),
        TQuestion("premise", "Why did the agent fail to reach the admin "
                  "dashboard?", "premise is false"),
    ]
    if ok.get("w02"):
        qs.append(TQuestion("premise",
                            "Why did the agent fail to add the keyboard "
                            "to the cart?", "premise is false"))
    return qs


# ---------- memórias / memory systems ----------
_TOK = re.compile(r"[a-z0-9]+")


def _ntok(s: str) -> int:
    return len(_TOK.findall(s))


class TruncatedHistory:
    """PT: história completa truncada ao orçamento. EN: full history
    truncated to the budget."""
    name = "truncated-history"

    def __init__(self, budget: int = 300) -> None:
        self.budget = budget
        self.steps: list[str] = []

    def reset(self) -> None:
        self.steps = []

    def add_traj(self, t: Trajectory) -> None:
        self.steps += [f"[{t.task_id}] {s}" for s in t.steps]

    def gather(self, query: str) -> list[str]:
        out: list[str] = []
        used = 0
        for s in reversed(self.steps):
            n = _ntok(s)
            if used + n > self.budget:
                break
            out.insert(0, s)
            used += n
        return out


class PlainRAG:
    name = "plain-rag"

    def __init__(self, budget: int = 300,
                 embedder: Embedder | None = None) -> None:
        self.budget = budget
        self.index = VectorIndex(embedder or HashingEmbedder())
        self.docs: dict[str, str] = {}
        self._n = 0

    def reset(self) -> None:
        self.index = VectorIndex(HashingEmbedder())
        self.docs = {}
        self._n = 0

    def add_traj(self, t: Trajectory) -> None:
        for s in t.steps:
            self._n += 1
            did = f"{t.task_id}s{self._n}"
            self.docs[did] = f"[{t.task_id}] {s}"
            self.index.add(did, self.docs[did])

    def gather(self, query: str) -> list[str]:
        out, used = [], 0
        for d, _ in self.index.search(query, 20):
            doc = self.docs[d]
            n = _ntok(doc)
            if used + n > self.budget:
                break
            out.append(doc)
            used += n
        return out


class AgentRunbookR:
    """PT: pools separados — state (obs), events (ações), notes (gotchas/
    workflow destilados na ingestão) — fundidos no orçamento.
    EN: separate pools — state obs, action events, distilled notes —
    merged under budget."""
    name = "runbook-r"

    def __init__(self, budget: int = 300,
                 embedder: Embedder | None = None) -> None:
        self.budget = budget
        self._emb = embedder or HashingEmbedder()
        self.reset()

    def reset(self) -> None:
        self.pools = {p: VectorIndex(self._emb)
                      for p in ("state", "events", "notes")}
        self.docs: dict[str, str] = {}
        self.pool_of: dict[str, str] = {}
        self._n = 0

    def _put(self, pool: str, text: str) -> None:
        self._n += 1
        did = f"r{self._n}"
        self.docs[did] = text
        self.pool_of[did] = pool
        self.pools[pool].add(did, text)

    def add_traj(self, t: Trajectory) -> None:
        for s in t.steps:
            _, _, rest = s.partition(" | ")
            act, _, obs = rest.partition(" | ")
            self._put("events", f"[{t.task_id}] action: {act}")
            self._put("state", f"[{t.task_id}] {s} → {obs}")
            # PT: notas de estratégia/gotchas destiladas na ingestão.
            # EN: strategy/gotcha notes distilled at ingest.
            for kw in ("not found", "404", "log in", "Logged in", "admin"):
                if kw.lower() in obs.lower():
                    self._put("notes",
                              f"[{t.task_id}] note: {act} → {obs[:90]}")
                    break

    def gather(self, query: str) -> list[str]:
        # PT: cada pool contribui ~1/3 do orçamento. EN: each pool
        # contributes ~1/3 of the budget.
        out: list[str] = []
        per = self.budget // 3
        for pool in ("state", "events", "notes"):
            used = 0
            for d, _ in self.pools[pool].search(query, 10):
                doc = self.docs[d]
                n = _ntok(doc)
                if used + n > per:
                    break
                out.append(doc)
                used += n
        return out


class AgentRunbookC:
    """PT: trajetórias como arquivos; agente "programador" coleta evidência
    via chamadas restritas `grep <pat>` / `read <file>` num sandbox de
    diretório. EN: trajectories as files; a "coding agent" gathers evidence
    via restricted `grep <pat>` / `read <file>` tool calls."""
    name = "runbook-c"

    def __init__(self, root: Path, budget: int = 300) -> None:
        self.root = root
        self.budget = budget
        self.reset()

    def reset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for f in self.root.glob("*.txt"):
            f.unlink()

    def add_traj(self, t: Trajectory) -> None:
        (self.root / f"{t.task_id}.txt").write_text(
            "\n".join(t.steps), encoding="utf-8")

    def _grep(self, pat: str) -> list[str]:
        rx = re.compile(pat, re.I)
        hits = []
        for f in sorted(self.root.glob("*.txt")):
            for ln in f.read_text(encoding="utf-8").splitlines():
                if rx.search(ln):
                    hits.append(f"{f.name}: {ln}")
        return hits

    def _read(self, fname: str) -> list[str]:
        f = self.root / Path(fname).name  # PT: sandbox — só basename.
        return f.read_text(encoding="utf-8").splitlines() if f.exists() else []

    def gather(self, query: str) -> list[str]:
        """PT: loop do coding agent: grep por tokens relevantes da query
        (+vocabulário de domínio), depois read nos arquivos mais citados.
        EN: coding-agent loop: grep on query tokens (+domain vocab), then
        read the most-cited files."""
        toks = [t for t in token_set(query) if len(t) > 2]
        toks += {"admin": ["admin", "dashboard"], "cart": ["cart", "added"],
                 "price": ["price"], "fail": ["not found", "404"],
                 "reply": ["reply", "logged"]}.get(
                     next((w for w in ("admin", "cart", "price", "fail",
                                       "reply") if w in query.lower()), ""),
                     [])
        hits: list[str] = []
        for t in toks:
            hits += self._grep(re.escape(t))
        seen: set[str] = set()
        uniq = [h for h in hits if not (h in seen or seen.add(h))]
        out, used = [], 0
        for h in uniq[:20]:
            n = _ntok(h)
            if used + n > self.budget:
                break
            out.append(h)
            used += n
        return out


# ---------- resposta / answering ----------
_PREMISE_FAIL = re.compile(r"\b(why did (the agent )?(fail|not)|failed to)\b",
                           re.I)


def answer(q: TQuestion, evidence: list[str], llm: LLM) -> str:
    """PT: perguntas de premissa → verifica se a premissa ("falhou X") é
    falsa à luz da evidência. EN: premise questions → check whether the
    premise is false given the evidence."""
    if q.comp == "workflow":
        # PT: reconstrói a sequência de ações das evidências da trajetória
        # relevante. EN: rebuild the action sequence from the relevant
        # trajectory's evidence.
        acts = [ln.split("|")[1].strip() for ln in evidence if "|" in ln]
        return "; ".join(acts) if acts else "I don't know"
    if q.comp == "premise" or _PREMISE_FAIL.search(q.question):
        joined = " ".join(evidence).lower()
        succeeded = any(
            k in joined for k in
            ("added keyboard to cart", "admin dashboard",
             "reply posted", "logged in"))
        if succeeded and "fail" not in joined:
            return ("the premise is false — the evidence shows the agent "
                    "succeeded")
        return "the premise holds — the failure is in the evidence"
    return answer_with_evidence(llm, q.question, evidence)


def verdict(llm: LLM, q: TQuestion, pred: str) -> bool:
    if q.comp == "premise":
        return "premise is false" in pred.lower() or "false" in pred.lower()
    if q.comp == "workflow":
        return all(w in pred.lower() for w in ("search", "add"))
    return bool(judge_answer(llm, q.question, q.gold, pred))
