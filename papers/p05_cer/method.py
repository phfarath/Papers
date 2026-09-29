"""CER (Contextual Experience Replay, 2025) — dynamics + skills distillados.

PT: Implementa o fluxo do paper (§3, Fig. 1-2): trajetórias → módulo de
destilação SEPARADO para dynamics (resumo de página + URL + usos possíveis) e
skills (resumo abstrato com placeholders + passo a passo com exemplos de
ação) → buffer com merge/dedup → retrieval separado (top-k dynamics e top-k
skills, k_d=k_s=5 do §5.1) → mapeamento programático f para o contexto do
agente ReAct. Settings: offline (trajetórias fornecidas), online (próprias
trajetórias, sucesso auto-julgado pelo reward do env) e híbrido.

EN: implements the paper's pipeline (§3, Fig. 1-2): trajectories → SEPARATE
distillation modules for dynamics (page summary + URL + usages) and skills
(abstract placeholder summary + step-by-step guide with action examples) →
buffer with merge/dedup → separate retrieval (top-k dynamics and top-k
skills, k_d=k_s=5 from §5.1) → programmatic mapping f into the ReAct agent's
context. Settings: offline, online (self-judged success via env reward) and
hybrid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from papers.common.llm import LLM, task_prompt
from papers.envs.web import MiniWeb, WebTask
from papers.p05_cer import prompts  # noqa: F401  (registra handlers)


@dataclass
class ExperienceBuffer:
    """PT: buffer de experiências — dynamics e skills separados, com merge por
    URL / nome de skill (dedup do §3.2: "existing experiences … to avoid
    repetitive distillation").

    EN: experience buffer — dynamics and skills kept separately, merged by
    URL / skill name (the paper's dedup rule).
    """

    dynamics: dict[str, str] = field(default_factory=dict)   # url -> entrada
    skills: dict[str, str] = field(default_factory=dict)     # nome -> bloco

    def merge(self, dyn_text: str, skl_text: str) -> None:
        for line in dyn_text.splitlines():
            m = re.match(r"- (\S+) — (.*)", line.strip())
            if not m:
                continue
            url, body = m.group(1), m.group(2)
            if "summarized before" in body.lower():
                continue  # PT: já existe — não duplicar. EN: already known.
            self.dynamics.setdefault(url, f"{url} — {body}")
        for blk in re.findall(r"<skill>(.*?)</skill>", skl_text, flags=re.S):
            name = blk.strip().splitlines()[0].strip()
            steps = "\n".join(blk.strip().splitlines()[1:])
            if "summarized before" in steps.lower() or not steps.strip():
                continue
            self.skills.setdefault(name, blk.strip())

    # ---------- retrieval separado (§3.3, k=5) ----------
    def retrieve(self, llm: LLM, goal: str) -> tuple[list[str], list[str]]:
        """PT: dois módulos de retrieval — top-5 dynamics e top-5 skills.

        EN: two retrieval modules — top-5 dynamics and top-5 skills.
        """
        dyn = ""
        if self.dynamics:
            body = ("Select the most useful environment dynamics for the goal.\n"
                    f"Goal: {goal}\nAvailable dynamics:\n"
                    + "\n".join(f"- {d}" for d in self.dynamics.values()))
            dyn = llm.complete(task_prompt("cer.retr.dyn", body))
        skl = ""
        if self.skills:
            body = ("Select the skills that can help most in achieving the "
                    "goal.\n"
                    f"Goal: {goal}\nAvailable skills:\n"
                    + "\n".join(f"- {n}" for n in self.skills))
            skl = llm.complete(task_prompt("cer.retr.skl", body))
        dyn_list = [ln[2:] for ln in dyn.splitlines() if ln.startswith("- ")]
        # PT: o retrieval de skills devolve nomes; mapeia para o bloco cheio.
        # EN: skill retrieval returns names; map back to the full block.
        skl_list = [self.skills[ln[2:].strip()] for ln in skl.splitlines()
                    if ln.startswith("- ") and ln[2:].strip() in self.skills]
        return dyn_list, skl_list


def _traj_text(trajectory: list[tuple[str, str]]) -> str:
    """PT: renderiza a trajetória como linhas a:/o:. EN: a:/o: lines."""
    return "\n".join(f"a: {a}\no: {o}" for a, o in trajectory)


def distill(llm: LLM, buffer: ExperienceBuffer,
            trajectory: list[tuple[str, str]]) -> None:
    """PT: destila uma trajetória no buffer (§3.2). Recebe o buffer existente
    no prompt para evitar distilações repetidas.

    EN: distills one trajectory into the buffer (§3.2). The existing buffer
    goes into the prompt to avoid repetitive distillation.
    """
    traj = _traj_text(trajectory)
    existing_d = "\n".join(f"- {d}" for d in buffer.dynamics.values())
    body = ("Summarize the environment dynamics of each page visited: page "
            "summary, URL and possible usages.\n"
            f"Existing dynamics:\n{existing_d or '(none)'}\n"
            f"Trajectory:\n{traj}")
    dyn_text = llm.complete(task_prompt("cer.dynamics", body))
    existing_s = "\n".join(f"- {n}" for n in buffer.skills)
    body = ("Summarize reusable skills as abstract workflows with "
            "placeholders ({product}, {text}, {topic}) plus step-by-step "
            "action examples.\n"
            f"Existing skills:\n{existing_s or '(none)'}\n"
            f"Trajectory:\n{traj}")
    skl_text = llm.complete(task_prompt("cer.skills", body))
    buffer.merge(dyn_text, skl_text)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    answer: str | None
    trajectory: list[tuple[str, str]]


def run_episode(llm: LLM, task: WebTask,
                buffer: ExperienceBuffer | None = None) -> EpisodeResult:
    """PT: episódio ReAct no MiniWeb com as experiências recuperadas injetadas
    no contexto (mapeamento programático f para linguagem natural, §3.3).
    Uma dynamics pode sugerir goto de URLs fora da lista (pegadinha do admin).

    EN: ReAct episode on MiniWeb with retrieved experience injected into the
    context (programmatic mapping f to natural language, §3.3). A dynamics
    entry may suggest goto URLs outside the listed ones (the admin gotcha).
    """
    env = MiniWeb(task)
    obs = env.reset()
    dyn: list[str] = []
    skl: list[str] = []
    if buffer is not None:
        dyn, skl = buffer.retrieve(llm, task.description)
    traj: list[tuple[str, str]] = []
    recent: list[str] = []
    while True:
        va = env.valid_actions()
        body = (
            "You are a web agent (ReAct). Pick the single best next action "
            "from the valid actions list; reply with ONLY the action. You may "
            "also 'goto <url>' for any URL cited in the Dynamics memory.\n"
            f"Task: {task.description}\n"
            f"Observation:\n{obs}\n"
            + ("Dynamics:\n" + "\n".join(f"- {d}" for d in dyn) + "\n"
               if dyn else "")
            + ("Skills:\n" + "\n".join(
                "- " + s.replace("\n", " | ") for s in skl) + "\n"
               if skl else "")
            + ("Recent actions:\n" + "\n".join(f"- {r}" for r in recent[-8:])
               + "\n" if recent else "")
            + "Valid actions:\n" + "\n".join(f"- {a}" for a in va)
        )
        act = llm.complete(task_prompt("web.act", body)).strip()
        mem_urls = {u for d in dyn for u in re.findall(r"http://\S+", d)}
        if act not in va:
            # PT: tolera "goto <url>" citada na memória ou na página — como um
            # agente real digitaria a URL. EN: tolerate goto to a URL cited in
            # memory or visible on the page.
            if act.startswith("goto ") and act[5:] in mem_urls:
                pass
            elif act.startswith("stop"):
                act = "stop " + (act[5:] if len(act) > 5 else "unknown")
            elif (act.startswith("type ") and len(act.split()) >= 3
                  and f"type {act.split()[1]} <text>" in va):
                # PT: "type <el> <texto livre>" casa o template válido.
                # EN: free-text type matching the valid template.
                pass
            else:
                # PT: ação fora da lista → repete a última válida não-stop.
                # EN: off-list action → fall back to a generic click.
                act = next((a for a in va if a.startswith("click")), va[0])
        res = env.step(act)
        traj.append((act, res.observation))
        recent.append(act)
        obs = res.observation
        if res.done:
            break
    return EpisodeResult(env.score() > 0, env.steps, env.answer, traj)


def run_online(llm: LLM, tasks: list[WebTask], rounds: int,
               buffer: ExperienceBuffer | None = None) -> tuple[list[float], ExperienceBuffer]:
    """PT: setting ONLINE (Fig. 1C): começa sem buffer; após cada episódio a
    trajetória própria é destilada e mergeada; o sucesso é auto-julgado pelo
    reward do env (trajectórias boas e ruins entram — §3.2 destila de ambas).

    EN: ONLINE setting: starts empty; after each episode the agent's own
    trajectory is distilled and merged; success is self-judged via env reward.
    """
    buf = buffer if buffer is not None else ExperienceBuffer()
    rates = []
    for _ in range(rounds):
        wins = 0
        for t in tasks:
            res = run_episode(llm, t, buf)
            wins += res.success
            distill(llm, buf, res.trajectory)
        rates.append(wins / len(tasks))
    return rates, buf
