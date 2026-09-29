"""PT: gerador determinístico de um pequeno dataset conversacional
multi-sessão (para p10–p14 e os benchmarks de batch 4).

EN: deterministic generator of a small multi-session conversational dataset
(for p10–p14 and the batch-4 benchmarks).

Conteúdo / Content: fatos de persona, atualizações que substituem fatos
antigos (com timestamps), retratações/deleções, conversa de enchimento e
perguntas com gold + tipo (single-hop, multi-hop, temporal, knowledge-update,
abstention).
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

    def all_turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s]


_CITIES = ["Lisbon", "Porto", "Berlin", "Madrid"]
_JOBS = ["barista", "nurse", "pilot", "teacher"]
_PETS = ["a cat named Nilo", "a dog named Rex", "a parrot named Kiwi"]
_HOBBIES = ["surfing", "chess", "baking bread", "birdwatching"]
_FAV = ["boiled eggs", "green tea", "sourdough", "black coffee"]
_FILLER = [
    "How is the weather today?", "Tell me a fun fact.", "What time is it?",
    "Thanks, that helps.", "Can you explain that differently?",
    "Nice.", "I see.", "Interesting, go on.", "Ok great.", "Hmm, alright.",
]


def generate(seed: int = 0) -> ConvSet:
    """PT: 4 sessões: fatos → atualizações (cidade/emprego mudam) → uma
    retratação (o usuário diz para esquecer o pet). EN: 4 sessions: facts →
    updates (city/job change) → one retraction (user says forget the pet)."""
    rng = random.Random(seed)
    base = datetime(2024, 1, 1, 9, 0)
    c1, c2 = rng.sample(_CITIES, 2)
    j1, j2 = rng.sample(_JOBS, 2)
    pet, hobby, fav = rng.choice(_PETS), rng.choice(_HOBBIES), rng.choice(_FAV)

    def U(txt: str, h: int) -> Turn:
        return Turn("user", txt, base + timedelta(hours=h))

    def A(txt: str, h: int) -> Turn:
        return Turn("agent", txt, base + timedelta(hours=h))

    def filler(n: int, h0: int) -> list[Turn]:
        return [U(rng.choice(_FILLER), h0 + i) for i in range(n)]

    s1 = [
        U("Hi! My name is Pedro.", 0),
        A("Nice to meet you, Pedro!", 0),
        U(f"I live in {c1}.", 1),
        U(f"I work as a {j1}.", 2),
        U(f"I have {pet}.", 3),
        U(f"My favorite breakfast is {fav}.", 4),
        *filler(3, 5),
    ]
    s2 = [
        U(f"Good news — I moved to {c2} last week.", 20),
        U(f"Also, I quit my job and now I work as a {j2}.", 21),
        *filler(3, 22),
    ]
    s3 = [
        U(f"By the way, I picked up {hobby} recently.", 40),
        U("Actually, please forget what I told you about my pet — "
          "I gave it away.", 41),
        *filler(3, 42),
    ]
    s4 = [
        U("Just checking in — everything's going well here.", 60),
        *filler(4, 61),
    ]
    qs = [
        Question("single-hop", "What is my favorite breakfast?", fav),
        Question("single-hop", "What is my hobby?", hobby),
        Question("update", "Where do I live now?", c2),
        Question("update", "What is my current job?", j2),
        Question("temporal", "Which city did I live in before moving?", c1),
        Question("temporal", "What was my previous job?", j1),
        Question("multi-hop",
                 "The city I live in now — what hobby did I pick up "
                 "after moving there?", hobby),
        Question("abstention", "Do I still have a pet?", "no pet"),
        Question("abstention", "What is my favorite movie?", "not provided"),
    ]
    return ConvSet([s1, s2, s3, s4], qs)
