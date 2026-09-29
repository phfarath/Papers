"""PT: extrator de fatos compartilhado — o "prior de LLM" usado pelos mock
handlers (p12 Zep, p13 Mem0, p14 Memory-R1) e pelo leitor (qa.answer).
Honestidade: lê apenas o texto que recebe.

EN: shared fact extractor — the "LLM prior" used by the mock handlers
(p12 Zep, p13 Mem0, p14 Memory-R1) and by the reader (qa.answer). Honesty:
reads only the text it is given.

Fatos normalizados no formato "{Nome}'s {slot} is {valor}" e retratações
como "remove: {Nome} {slot}". Facts are normalized as
"{Name}'s {slot} is {value}"; retractions as "remove: {Name} {slot}".
"""

from __future__ import annotations

import re

_NAME = re.compile(
    r"(?:my name is|this is|it's|here is) ([A-Z][a-zA-Z]+)", flags=re.I)
_NAME_HERE = re.compile(r"\b([A-Z][a-zA-Z]+) (?:here|again)\b")

SLOTS = ("city", "job", "pet", "breakfast", "hobby")

# PT: (slot, [regex com 1 grupo de captura = valor]). Cobre phrasings
# variados do dataset. EN: (slot, [1-capture-group patterns]) covering the
# dataset's varied phrasings.
SLOT_PATTERNS: list[tuple[str, list[str]]] = [
    ("city", [
        r"live in ([A-Z][a-zA-Z]+)", r"moved to ([A-Z][a-zA-Z]+)",
        r"living in ([A-Z][a-zA-Z]+)", r"resid(?:e|ing) in ([A-Z][a-zA-Z]+)",
        r"settled in ([A-Z][a-zA-Z]+)", r"relocated to ([A-Z][a-zA-Z]+)",
    ]),
    ("job", [
        r"(?:work(?:ing)? as|working now as|now I work as) "
        r"(?:an? )?(\w[\w ]*?)(?: at| in| and|\.|,|$)",
        r"started (?:a new job )?as (?:an? )?(\w[\w ]*?)(?: at| in| and|\.|,|$)",
        r"(?:quit|left) my job(?: as (?:an? )?(\w[\w ]*?)(?: at|\.|,|$))?",
        r"(?:I'm|I am) (?:an? )?([a-z]+) (?:by profession|now)",
    ]),
    ("pet", [
        r"(?:my|a) (cat|dog|parrot|hamster|rabbit)"
        r"(?: ?(?:is |named )?(\w+))?",
    ]),
    ("breakfast", [
        r"favorite breakfast is ([\w ]+?)(?:\.| and|$)",
        r"(?:love|have|eat) ([\w ]+?) for breakfast",
        r"(?:usually|always) have ([\w ]+?) in the morning",
    ]),
    ("hobby", [
        r"(?:picked up|taken up|started) ([\w ]+?)(?: recently|\.| and|$)",
        r"hobby is ([\w ]+?)(?:\.| and|$)",
        r"(?:enjoy|love|got into) ([\w ]+?)(?: lately|\.| and| on weekends|$)",
        r"in my free time I (?:practice|do|enjoy) ([\w ]+?)(?:\.| and|$)",
        r"weekends (?:on|doing) ([\w ]+?)(?:\.| and|$)",
        r"took up ([\w ]+?)(?: lately|\.| and|$)",
    ]),
]

# PT: retratações por slot. EN: per-slot retractions.
RETRACT_PATTERNS: list[tuple[str, str]] = [
    ("pet", r"gave .* away|forget.*pet|no longer have|don't have anymore"
            r"|gave (?:it|him|her) away"),
    ("hobby", r"(?:gave up|stopped) (?:my hobby|playing|practicing)"),
]

# PT: sinônimos slot → termos que indicam o slot (usado pelo leitor).
# EN: slot → cue terms (used by the reader).
SLOT_SYNONYMS: dict[str, set[str]] = {
    "city": {"live", "lives", "living", "lived", "moved", "move", "reside",
             "resides", "city", "town", "settled", "relocated", "where"},
    "job": {"work", "works", "working", "worked", "job", "profession",
            "occupation", "employed", "career"},
    "pet": {"pet", "dog", "cat", "parrot", "animal", "hamster", "rabbit"},
    "breakfast": {"breakfast", "morning", "eat", "eats"},
    "hobby": {"hobby", "enjoy", "weekend", "weekends", "free", "time",
              "picked", "taken"},
    "name": {"name", "called"},
}


def extract_name(text: str, default: str = "user") -> str:
    m = _NAME.search(text) or _NAME_HERE.search(text)
    return m.group(1) if m else default


def extract_facts(text: str, default_name: str = "user") -> list[str]:
    """PT: devolve fatos normalizados "{Nome}'s {slot} is {valor}" e
    "remove: {Nome} {slot}". EN: normalized facts and removals."""
    name = extract_name(text, default_name)
    out: list[str] = []
    for slot, pats in SLOT_PATTERNS:
        for pat in pats:
            for m in re.finditer(pat, text, flags=re.I):
                if slot == "pet":
                    val = " ".join(g for g in m.groups() if g)
                else:
                    val = next((g for g in m.groups() if g), "")
                val = val.strip().rstrip(".")
                if not val:
                    continue
                if slot == "job" and re.match(r"quit|left", pat[:12]):
                    # PT: "quit my job" sem valor = não afirmar um cargo.
                    # EN: bare "quit my job" asserts no position.
                    if not m.group(1):
                        continue
                # PT: guarda — hobby não engole fatos de job ("started a
                # new job as X"). EN: guard — hobby must not swallow job
                # facts ("started a new job as X").
                if slot == "hobby" and (
                        re.search(r"\b(job|career|profession|work|breakfast)"
                                  r"\b", m.group(0), flags=re.I)
                        or re.match(r"as\b", val)):
                    continue
                out.append(f"{name}'s {slot} is {val}")
                break
    for slot, pat in RETRACT_PATTERNS:
        if re.search(pat, text, flags=re.I):
            out.append(f"remove: {name} {slot}")
    return list(dict.fromkeys(out))
