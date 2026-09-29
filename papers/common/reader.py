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

from papers.common.facts import SLOT_SYNONYMS, extract_facts
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


_FUNCTION_WORDS = {"s", "what", "where", "which", "who", "do", "does",
                   "did", "is", "are", "the", "a", "an", "of", "to", "in",
                   "on", "at", "and", "or", "my", "your", "their", "his",
                   "her", "it", "its", "i", "you", "we", "they", "this",
                   "that", "there", "here", "so", "any", "some"}

_Q_STOP = {"What", "Where", "Which", "Who", "Do", "Does", "Did", "Is", "Are",
           "The", "After", "Before", "I", "My", "In", "On", "At", "Their",
           "His", "Her", "Now", "Then", "And", "Sad", "Actually", "How"}

def _question_slots(question: str) -> list[str]:
    """PT: slots pedidos pela pergunta via lexico de sinônimos — o slot
    PRINCIPAL é o primeiro sinônimo que aparece depois do interrogativo
    ("what/which/where"); os demais são contexto. EN: asked slots via the
    synonym lexicon — the PRIMARY slot is the first synonym after the
    interrogative word; the rest are context."""
    q = question.lower()
    im = re.search(r"\b(what|which|where|who)\b", q)
    start = im.end() if im else 0
    found: list[tuple[int, str]] = []
    for s, syns in SLOT_SYNONYMS.items():
        if s == "name":
            continue
        for syn in syns:
            m = re.search(rf"\b{re.escape(syn)}\w*\b", q[start:])
            if m:
                found.append((start + m.start(), s))
                break
    found.sort()
    return list(dict.fromkeys(s for _, s in found))


def _question_name(question: str) -> str | None:
    """PT: nome da persona citado na pergunta (token capitalizado).
    EN: persona name cited in the question (capitalized token)."""
    for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", question):
        if w not in _Q_STOP:
            return w
    return None


def _evidence_facts(ev: str) -> list[tuple[str, str, str, str]]:
    """PT: fatos de uma evidência: (nome, slot, valor, texto-da-evidência).
    Formato "{Nome}'s {slot} is {valor}" ou texto natural via extrator.
    EN: facts of an evidence: (name, slot, value, evidence text)."""
    out: list[tuple[str, str, str, str]] = []
    for m in re.finditer(r"(\w+)'s (\w+) is ([^.;(]+)", ev):
        out.append((m.group(1), m.group(2), m.group(3).strip(), ev))
    if not out:
        for f in extract_facts(re.sub(r"\(valid:[^)]*\)", "", ev)):
            if f.startswith("remove:"):
                continue
            m = re.match(r"(.+)'s (\w+) is (.*)", f)
            if m:
                out.append((m.group(1), m.group(2), m.group(3), ev))
    return out


def _evidence_retracts(ev: str, name: str | None, slot: str) -> bool:
    """PT: evidência retrata o slot dessa pessoa? EN: does the evidence
    retract that person's slot?"""
    for f in extract_facts(re.sub(r"\(valid:[^)]*\)", "", ev)):
        m = re.match(r"remove: (\w+) (\w+)", f)
        if m and m.group(2) == slot and (
                name is None or m.group(1).lower() == name.lower()
                or m.group(1) == "user"):  # retratação sem nome vira global
            return True
    return False


def _ev_date(ev: str) -> datetime:
    d = _dates_in(ev)
    return max(d) if d else datetime.min


def _ev_open(ev: str) -> bool:
    """PT: intervalo Zep ainda aberto ("→ present" ou sem fim). EN: Zep
    interval still open ("→ present" or no end)."""
    m = re.search(r"\(valid: [^→]*→ ([^)]*)\)", ev)
    return bool(m and m.group(1).strip() == "present") or m is None


@mock_handler("qa.answer")
def _qa_answer(prompt: str, rng: random.Random) -> str:
    # PT: REGRA DE HONESTIDADE — so lemos o prompt. EN: HONESTY RULE — prompt only.
    lines = prompt.splitlines()
    try:
        q_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Question:"))
    except StopIteration:
        return "I don't know"
    question = lines[q_idx][len("Question:") :].strip()
    evidences: list[str] = []
    for line in lines:
        m = re.match(r"\[(\d+)\]\s*(.*)", line)
        if m:
            evidences.append(m.group(2))
    if not evidences:
        return "I don't know"

    q_toks = token_set(question)
    name = _question_name(question)
    slots = _question_slots(question)
    # PT: pistas temporais no texto cru (token_set descarta "did"/"had").
    # EN: temporal cues on raw text (token_set drops "did"/"had").
    ql = question.lower()
    wants_now = bool(re.search(
        r"\b(now|current|currently|still|today|lately|present)\b", ql))
    wants_past = bool(re.search(
        r"\b(before|previous|previously|used to|originally|earlier|did|had|"
        r"was|were)\b", ql))

    # PT: fatos candidatos das evidências; filtra por nome e slot pedidos.
    # EN: candidate facts from evidence; filter by asked name and slot.
    cand: list[tuple[str, str, str, str]] = []
    if slots:  # PT: só filtra por nome quando há slot pedido. EN: only
        # filter by name when a slot is asked.
        for ev in evidences:
            for f in _evidence_facts(ev):
                # PT: "user" = fato não-atribuído da mesma conversa → aceita.
                # EN: "user" = unattributed same-conversation fact → accept.
                if name and f[0].lower() != name.lower() and f[0] != "user":
                    continue
                if f[1] not in slots:
                    continue
                cand.append(f)
    # PT: sem nome/slot explícitos, aceita fatos com overlap lexical com a
    # pergunta (compatibilidade retroativa). EN: without explicit name/slot,
    # fall back to lexical-overlap facts (backwards compat).
    if not cand:
        # PT: fallback só com overlap de conteúdo real (≥2 tokens de
        # conteúdo) — evita que "what is X's shoe size" pegue qualquer fato.
        # EN: fallback only on real content overlap (≥2 content tokens).
        content_q = q_toks - _FUNCTION_WORDS
        for ev in evidences:
            fs = _evidence_facts(ev)
            if not fs and ev.strip() and ev.strip() != "(none)":
                # PT: evidência crua sem fato extraível → pseudo-fato.
                # EN: raw evidence with no extractable fact → pseudo-fact.
                fs = [("user", "", ev, ev)]
            for f in fs:
                ov = content_q & (token_set(f[3]) - _FUNCTION_WORDS)
                subj = any(
                    bool(token_set(m.group(1)) & content_q)
                    for m in re.finditer(r"([A-Za-z ']+?) is ([^.;]+)", f[3])
                ) if not f[1] else False
                if len(ov) >= 2 or (slots and f[1] in slots) or subj:
                    cand.append(f)

    slot = slots[0] if slots else (cand[0][1] if cand else None)
    retracted = any(_evidence_retracts(ev, name, slot or "") for ev in evidences)
    retr_date = max((_ev_date(ev) for ev in evidences
                     if slot and _evidence_retracts(ev, name, slot)),
                    default=datetime.min)

    if not cand:
        # PT: abstenção — slot retratado → "no X"; nada sobre o slot →
        # "I don't know". EN: abstention — retracted slot → "no X"; nothing
        # about the slot → "I don't know".
        if slot and retracted:
            return f"no {slot}"
        return "I don't know"

    # PT: perguntas de LISTAGEM ("list every X") → agrega os valores
    # distintos das evidências (fatos extraídos ou substantivos após
    # verbos de lugar). EN: LIST questions → aggregate distinct values
    # across evidence (extracted facts or place nouns after verbs).
    if re.search(r"\b(list every|list all|which \w+s|all the \w+s)\b",
                 ql):
        vals: list[str] = []
        for f in cand:
            for m in re.finditer(
                    r"(?:visit|visiting|went to|traveled to|moved to|"
                    r"live in|living in|settled in|stopover in) "
                    r"([A-Z][a-zA-Z]+)", f[3]):
                if m.group(1) not in vals:
                    vals.append(m.group(1))
            if f[1] and f[2] not in vals:
                vals.append(f[2])
        if vals:
            return ", ".join(sorted(vals))

    # PT: escolha temporal — "now" prefere válido+mais recente; "before"
    # prefere o mais antigo/invalidado. EN: temporal choice — "now" prefers
    # valid+latest; "before" prefers the older/invalidated value.
    def valid(f: tuple[str, str, str, str]) -> bool:
        return _ev_open(f[3]) and not _evidence_retracts(f[3], f[0], f[1])

    if wants_past:
        # PT: o valor anterior = o intervalo fechado mais recente
        # (predecessor imediato); sem intervalos, o mais antigo. EN: the
        # previous value = the most recently closed interval (immediate
        # predecessor); without intervals, the oldest.
        closed = [f for f in cand if not _ev_open(f[3])]
        if closed:
            pick = max(closed, key=lambda f: _ev_date(f[3]))
        else:
            pick = min(cand, key=lambda f: _ev_date(f[3]))
    else:
        valids = [f for f in cand if valid(f)]
        pool = valids or cand
        if wants_now:
            # PT: com datas iguais, a última evidência é a mais recente
            # (contexto cru vem em ordem cronológica). EN: on equal dates
            # the later evidence is newest (raw context is chronological).
            pool.sort(key=lambda f: _ev_date(f[3]))
            pick = pool[-1]
        else:
            pool.sort(key=lambda f: _ev_date(f[3]), reverse=True)
            pick = pool[0]
        if not pick[1]:
            # PT: pseudo-fato — extrai "X is Y" cujo sujeito toca a
            # pergunta, senão devolve a evidência crua. EN: pseudo-fact —
            # extract "X is Y" whose subject touches the question, else
            # return the raw evidence.
            for m in re.finditer(r"([A-Za-z ']+?) is ([^.;]+)", pick[3]):
                if token_set(m.group(1)) & content_q:
                    return m.group(2).strip()
            return pick[3]
    if slot and retracted and not wants_past:
        # PT: o slot só sobrevive se houver fato válido POSTERIOR à
        # retratação. EN: the slot survives only if a valid fact POSTDATES
        # the retraction.
        still = [f for f in cand if f[1] == slot and valid(f)
                 and _ev_date(f[3]) > retr_date]
        if not still:
            return f"no {slot}"
    return pick[2]


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
    # PT: equivalência de abstenção — gold "not provided" ≈ pred "don't know".
    # EN: abstention equivalence — gold "not provided" ≈ pred "don't know".
    abst_g = gn in {"not provided", "unknown", "none", "not known", "na"}
    # PT: "I don't know" normaliza para "i don t know" — match por tokens.
    # EN: "I don't know" normalizes to "i don t know" — token match.
    ptoks = set(pn.split())
    abst_p = (("don" in ptoks and "know" in ptoks)
              or "not provided" in pn or "unknown" in pn)
    ok = bool(gn) and bool(pn) and (
        gn in pn or pn in gn or gn == pn or (abst_g and abst_p))
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
