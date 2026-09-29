"""Agente leitor compartilhado (QA sobre evidencias recuperadas).

PT: `answer_with_evidence` monta um prompt real com a pergunta e as evidencias.
O handler mock "qa.answer" age como um LLM razoavel faria: escolhe a evidencia
com maior overlap com a pergunta (preferindo a mais recente quando ha datas), e
extrai o "slot" pedido de forma generica ("X is Y", "X: Y"). Se nada tem overlap
suficiente, responde "I don't know". Assim a acuracia mede a QUALIDADE DA
RECUPERACAO, e nao um vazamento de gabarito.

EN: `answer_with_evidence` builds a real prompt with question + evidence. The
"qa.answer" mock handler acts like a reasonable LLM: picks the evidence with
most overlap with the question (preferring the most recent when dated), then
extracts the asked slot generically ("X is Y", "X: Y"). If nothing overlaps
enough it answers "I don't know". Accuracy thus measures RETRIEVAL QUALITY,
not gold leakage.
"""

from __future__ import annotations

import random
import re
from datetime import datetime

from papers.common.llm import LLM, mock_handler, task_prompt
from papers.common.utils import token_set


def answer_with_evidence(
    llm: LLM, question: str, evidence: list[str], *, now: datetime | None = None
) -> str:
    """PT: prompta o LLM para responder usando SOMENTE as evidencias.

    EN: prompts the LLM to answer using ONLY the evidence.
    """
    ev_text = "\n".join(f"[{i}] {e}" for i, e in enumerate(evidence)) or "(none)"
    when = f"Current date: {now.date()}\n" if now else ""
    body = (
        f"{when}Answer the question using ONLY the evidence below. "
        "If the evidence is insufficient, reply exactly 'I don't know'.\n"
        f"Evidence:\n{ev_text}\nQuestion: {question}\nAnswer:"
    )
    return llm.complete(task_prompt("qa.answer", body)).strip()


def _dates_in(text: str) -> list[datetime]:
    """PT: encontra datas ISO no texto. EN: find ISO dates in text."""
    out = []
    for m in re.findall(r"\d{4}-\d{2}-\d{2}", text):
        try:
            out.append(datetime.fromisoformat(m))
        except ValueError:
            pass
    return out


@mock_handler("qa.answer")
def _qa_answer(prompt: str, rng: random.Random) -> str:
    # PT: REGRA DE HONESTIDADE — so lemos o prompt. EN: HONESTY RULE — prompt only.
    lines = prompt.splitlines()
    try:
        q_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Question:"))
    except StopIteration:
        return "I don't know"
    question = lines[q_idx][len("Question:") :].strip()
    evidences: list[tuple[int, str]] = []
    for line in lines:
        m = re.match(r"\[(\d+)\]\s*(.*)", line)
        if m:
            evidences.append((int(m.group(1)), m.group(2)))
    if not evidences:
        return "I don't know"

    q_toks = token_set(question)
    best: list[tuple[float, datetime, str]] = []
    for _idx, ev in evidences:
        ov = len(q_toks & token_set(ev))
        dates = _dates_in(ev)
        best.append((ov, max(dates) if dates else datetime.min, ev))
    best_ov = max(b[0] for b in best)
    if best_ov < 1:  # PT: nenhuma evidência toca na pergunta. EN: no overlap at all.
        return "I don't know"
    # PT: desempate pela evidencia mais recente. EN: tie-break by most recent.
    top = sorted((b for b in best if b[0] == best_ov), key=lambda b: b[1])
    ev = top[-1][2]

    # PT: extrai slot generico "X is Y" / "X: Y" mencionando termos da pergunta.
    # EN: generic slot extraction "X is Y" / "X: Y" mentioning question terms.
    for pat in (r"([A-Za-z ']+?) is ([^.;]+)", r"([A-Za-z ']+?): ([^.;]+)"):
        for m in re.finditer(pat, ev):
            if token_set(m.group(1)) & q_toks:
                return m.group(2).strip()
    # PT: fallback: devolve a frase mais relevante da evidencia.
    # EN: fallback: return the most overlapping sentence of the evidence.
    sents = re.split(r"(?<=[.!?])\s+", ev)
    sents.sort(key=lambda s: len(q_toks & token_set(s)), reverse=True)
    return sents[0].strip() if sents else "I don't know"


def _normalize(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


@mock_handler("qa.judge")
def _qa_judge(prompt: str, rng: random.Random) -> str:
    """PT: juiz mock — match normalizado/substring lido do prompt.

    EN: mock judge — normalized substring match read from the prompt.
    """
    g = p = ""
    for line in prompt.splitlines():
        low = line.lower()
        if low.startswith("gold:"):
            g = line[5:].strip()
        elif low.startswith("pred:"):
            p = line[5:].strip()
    gn, pn = _normalize(g), _normalize(p)
    ok = bool(gn) and bool(pn) and (gn in pn or pn in gn or gn == pn)
    return "yes" if ok else "no"


def judge_answer(llm: LLM, question: str, gold: str, pred: str) -> bool:
    """PT: juiz de resposta via prompt (task "qa.judge") — funciona igual no
    mock e num LLM real.

    EN: answer judge via a prompt (task "qa.judge") — works the same on the
    mock and on a real LLM.
    """
    body = (
        "You are a strict judge. Decide if PRED matches GOLD for the QUESTION. "
        "Reply only 'yes' or 'no'.\n"
        f"QUESTION: {question}\nGOLD: {gold}\nPRED: {pred}"
    )
    out = llm.complete(task_prompt("qa.judge", body)).strip().lower()
    return out.startswith("yes")
