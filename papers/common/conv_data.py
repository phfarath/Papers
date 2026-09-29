"""PT: gerador determinístico de um dataset conversacional multi-sessão
(para p10–p14 e os benchmarks de batch 4). ≥10 personas por seed, cada uma
com histórico multi-sessão, ≥2 atualizações, 1 retratação, sessões de
enchimento e ~10 perguntas (5 tipos) → ≥100 perguntas.

EN: deterministic generator of a multi-session conversational dataset
(for p10–p14 and batch-4 benchmarks). ≥10 personas per seed, each with a
multi-session history, ≥2 updates, 1 retraction, filler sessions, and ~10
questions (5 types) → ≥100 questions.

O nome da persona aparece nos turnos factuais ("Pedro here, ...") para que
memória e leitor possam atribuir fatos. The persona name appears in factual
turns so memory and reader can attribute facts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Turn:
    speaker: str           # "user" | "agent"
    text: str
    ts: datetime


@dataclass
class Question:
    qtype: str             # single-hop|multi-hop|temporal|update|abstention
    question: str
    gold: str


@dataclass
class ConvSet:
    sessions: list[list[Turn]]
    questions: list[Question] = field(default_factory=list)
    # PT: turnos por persona (para evidência-ouro de calibração e holdout).
    # EN: turns per persona (calibration gold evidence + train/holdout).
    by_persona: dict[str, list[Turn]] = field(default_factory=dict)

    def all_turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s]

    def personas(self) -> list[str]:
        return list(self.by_persona)


_NAMES = ["Pedro", "Ana", "Luis", "Marta", "Joao", "Sofia", "Rui", "Ines",
          "Tiago", "Lara", "Miguel", "Vera"]
_CITIES = ["Lisbon", "Porto", "Berlin", "Madrid", "Paris", "Rome", "Vienna",
           "Dublin"]
_JOBS = ["barista", "nurse", "pilot", "teacher", "engineer", "chef",
         "artist", "plumber"]
_PETS = [("cat", "Nilo"), ("dog", "Rex"), ("parrot", "Kiwi"),
         ("rabbit", "Luna"), ("hamster", "Pip")]
_HOBBIES = ["surfing", "chess", "baking bread", "birdwatching", "pottery",
            "hiking", "origami", "swimming"]
_FAV = ["boiled eggs", "green tea", "sourdough", "black coffee",
        "pancakes", "fruit salad", "oatmeal", "croissants"]

# PT: phrasings variados — todos cobertos por common.facts.SLOT_PATTERNS.
# EN: varied phrasings — all covered by common.facts.SLOT_PATTERNS.
_P_CITY = [
    "I live in {v}.", "I've been living in {v} for a while.",
    "Right now I live in {v}.", "I settled in {v}.",
]
_P_CITY_UPD = [
    "Good news — I moved to {v} last week.",
    "I just moved to {v}.", "I relocated to {v} recently.",
    "These days I'm living in {v}.",
]
_P_JOB = [
    "I work as a {v}.", "I'm a {v} by profession.",
    "These days I'm working as a {v}.", "I started as a {v}.",
]
_P_JOB_UPD = [
    "Also, I quit my job and now I work as a {v}.",
    "New chapter: I left my old job and started a new job as a {v}.",
    "I changed careers — now I'm working as a {v}.",
]
_P_PET = [
    "I have a {sp} named {n}.", "My {sp} {n} keeps me company.",
    "By the way, I got a {sp} named {n}.",
]
_P_FAV = [
    "My favorite breakfast is {v}.", "I love {v} for breakfast.",
    "I usually have {v} in the morning.",
]
_P_HOBBY = [
    "I picked up {v} recently.", "Lately I got into {v}.",
    "My hobby is {v}.", "I took up {v} lately.",
    "I spend my weekends on {v}.",
]
_P_RETRACT = [
    "Actually, please forget what I told you about my pet — "
    "I gave it away.",
    "Sad update: I gave my pet away, so no pet anymore.",
    "I no longer have the pet — had to give it away.",
]
_FILLER = [
    "How is the weather today?", "Tell me a fun fact.", "What time is it?",
    "Thanks, that helps.", "Can you explain that differently?",
    "Nice.", "I see.", "Interesting, go on.", "Ok great.", "Hmm, alright.",
    "What do you think about the news?", "Just checking in.",
]
_UNKNOWN = ["favorite movie", "shoe size", "middle name", "blood type",
            "first car", "luckiest number"]


def _name_tag(name: str, rng: random.Random) -> str:
    # PT: sempre inclui o nome (atribuição do fato). EN: always includes
    # the name (fact attribution).
    return rng.choice([f"{name} here. ", f"It's {name} again. ",
                       f"This is {name}. ", f"Hey, {name} here. "])


def generate(seed: int = 0, n_personas: int = 10) -> ConvSet:
    """PT: gera o dataset: ~4 sessões por persona (fatos → atualizações →
    retratação → enchimento), ~10 perguntas por persona.
    EN: generate: ~4 sessions per persona (facts → updates → retraction →
    filler), ~10 questions per persona."""
    rng = random.Random(seed)
    base = datetime(2024, 1, 1, 9, 0)
    names = rng.sample(_NAMES, n_personas)
    sessions: list[list[Turn]] = []
    questions: list[Question] = []
    by_persona: dict[str, list[Turn]] = {}
    day = 0

    for name in names:
        c1, c2 = rng.sample(_CITIES, 2)
        j1, j2 = rng.sample(_JOBS, 2)
        sp, pn = rng.choice(_PETS)
        hobby = rng.choice(_HOBBIES)
        fav = rng.choice(_FAV)
        unknown = rng.choice(_UNKNOWN)

        def U(txt: str, h: int) -> Turn:
            # PT: timestamps monotônicos entre sessões (h absoluto crescente).
            # EN: monotonic timestamps across sessions (absolute rising h).
            return Turn("user", txt, base + timedelta(hours=h))

        def filler(n: int, h0: int) -> list[Turn]:
            return [U(rng.choice(_FILLER), h0 + i) for i in range(n)]

        def T(template: str, _n: str = name, **kw) -> str:
            # PT: todo turno factual carrega o nome da persona (atribuição).
            # EN: every factual turn carries the persona name (attribution).
            return _name_tag(_n, rng) + template.format(**kw)

        h = day * 100
        s1 = [
            U(f"Hi! My name is {name}.", h),
            U(T(rng.choice(_P_CITY), v=c1), h + 1),
            U(T(rng.choice(_P_JOB), v=j1), h + 2),
            U(T(rng.choice(_P_PET), sp=sp, n=pn), h + 3),
            U(T(rng.choice(_P_FAV), v=fav), h + 4),
            *filler(3, h + 5),
        ]
        s2 = [
            U(T(rng.choice(_P_CITY_UPD), v=c2), h + 20),
            U(T(rng.choice(_P_JOB_UPD), v=j2), h + 21),
            *filler(3, h + 22),
        ]
        s3 = [
            U(T(rng.choice(_P_HOBBY), v=hobby), h + 40),
            U(T(rng.choice(_P_RETRACT)), h + 41),
            *filler(3, h + 42),
        ]
        s4 = [
            U("Just checking in — everything's going well here.", h + 60),
            *filler(4, h + 61),
        ]
        persona_turns = s1 + s2 + s3 + s4
        by_persona[name] = persona_turns
        sessions += [s1, s2, s3, s4]
        day += 1

        questions += [
            Question("single-hop",
                     f"What is {name}'s favorite breakfast?", fav),
            Question("single-hop", f"What is {name}'s hobby?", hobby),
            Question("single-hop", f"What pet did {name} have?", pn),
            Question("update", f"Where does {name} live now?", c2),
            Question("update", f"What is {name}'s current job?", j2),
            Question("temporal",
                     f"Which city did {name} live in before moving?", c1),
            Question("temporal", f"What was {name}'s previous job?", j1),
            Question("multi-hop",
                     f"After moving to their current city, what hobby did "
                     f"{name} pick up?", hobby),
            Question("abstention", f"Does {name} still have a pet?",
                     "no pet"),
            Question("abstention", f"What is {name}'s {unknown}?",
                     "not provided"),
        ]
    return ConvSet(sessions, questions, by_persona)
