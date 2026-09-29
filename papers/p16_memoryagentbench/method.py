"""PT: p16 MemoryAgentBench (arXiv 2507.05257) — benchmark de memória via
interações multi-turno incrementais: o histórico é ingerido em CHUNKS e a
pergunta só chega no fim. Quatro competências (§3.1):

- **AR** (Accurate Retrieval): recuperar um fato exato do stream.
- **TTL** (Test-Time Learning): aprender um mapeamento de rótulos dado por
  exemplos no próprio stream e aplicá-lo.
- **LRU** (Long-Range Understanding): agregação sobre o stream inteiro —
  medida por cobertura de palavras-chave do ouro.
- **SF** (Selective Forgetting / FactConsolidation): fatos posteriores
  sobrescrevem os anteriores (single-hop + multi-hop).

EN: p16 MemoryAgentBench — memory benchmark via incremental multi-turn
interactions: the history is ingested in CHUNKS and the question arrives at
the end. Four competencies (§3.1): AR (accurate retrieval), TTL (test-time
learning of an in-stream label mapping), LRU (long-range aggregation scored
by keyword coverage), SF (later facts overwrite earlier ones).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from papers.common.conv_data import _CITIES, _JOBS, _PETS, Turn
from papers.common.llm import LLM
from papers.common.memory_api import MemoryItem
from papers.common.reader import answer_with_evidence, judge_answer


@dataclass
class BenchItem:
    """PT: um caso = stream de turnos + pergunta + ouro + competência.
    EN: one case = turn stream + question + gold + competency."""
    comp: str                 # AR | TTL | LRU | SF
    stream: list[Turn]
    question: str
    gold: str


@dataclass
class BenchSet:
    items: list[BenchItem] = field(default_factory=list)


_FOODS = ["pizza", "sushi", "tacos", "pasta", "salad", "ramen"]
_LABELS = ["ALP", "BET", "GAM", "DEL"]
_TTL_CATS = [("barks and wags its tail", "ALP", "a dog"),
             ("has feathers and can fly", "BET", "a bird"),
             ("swims in the ocean", "GAM", "a fish"),
             ("crawls and has scales", "DEL", "a snake")]


def _ts(i: int) -> datetime:
    return datetime(2024, 3, 1, 9, 0) + timedelta(hours=i)


def _stream_ar(rng: random.Random, name: str) -> BenchItem:
    food = rng.choice(_FOODS)
    turns = [Turn("user", f"{name} here. My comfort food is {food}.", _ts(0))]
    filler = ["How are you?", "Tell me a joke.", "Nice weather today.",
              "Thanks!", "Just checking in.", "Interesting."]
    turns += [Turn("user", rng.choice(filler), _ts(i + 1))
              for i in range(rng.randint(15, 20))]
    return BenchItem("AR", turns,
                     f"What is {name}'s comfort food?", food)


def _stream_ttl(rng: random.Random) -> BenchItem:
    """PT: mapeamento de rótulos aprendido em-contexto: o stream declara o
    mapeamento e dá exemplos; a pergunta pede a classe de um novo item.
    EN: in-context label mapping — the stream declares the mapping with
    examples; the question asks the label of a new item."""
    perm = rng.sample(_TTL_CATS, 4)
    turns = [Turn("user", "Annotation guide: assign each description one "
                  "of the labels ALP/BET/GAM/DEL.", _ts(0))]
    for desc, lab, noun in perm:
        turns.append(Turn("user", f"Example: '{desc}' → label {lab} "
                          f"(it describes {noun}).", _ts(len(turns))))
    new_desc, lab, noun = rng.choice(_TTL_CATS)
    turns += [Turn("user", "Please continue.", _ts(len(turns)))]
    return BenchItem("TTL", turns,
                     f"Using the mapping above, which label applies to "
                     f"'{new_desc}' (it describes {noun})?", lab)


def _stream_lru(rng: random.Random, name: str) -> BenchItem:
    """PT: agregação — lista de coisas mencionadas ao longo do stream.
    EN: aggregation — a list of things mentioned across the stream."""
    cities = rng.sample(_CITIES, 5)
    turns = [Turn("user", f"{name} here. Planning a trip.", _ts(0))]
    for i, c in enumerate(cities):
        turns.append(Turn("user", f"On day {i + 1} I'd like to visit {c}.",
                          _ts(2 * i + 1)))
        turns.append(Turn("user", rng.choice(
            ["Sounds fun.", "What about hotels?", "Noted."], ), _ts(2 * i + 2)))
    gold = ", ".join(sorted(cities))
    return BenchItem("LRU", turns,
                     f"Which cities did {name} say they'd like to visit "
                     f"in the trip plan?", gold)


def _stream_sf(rng: random.Random, name: str) -> list[BenchItem]:
    c1, c2 = rng.sample(_CITIES, 2)
    j1, j2 = rng.sample(_JOBS, 2)
    sp, pn = rng.choice(_PETS)
    turns = [
        Turn("user", f"{name} here. I live in {c1} and work as a {j1}.",
             _ts(0)),
        Turn("user", f"Also, I have a {sp} named {pn}.", _ts(1)),
        Turn("user", rng.choice(["How are you?", "Nice day."]), _ts(2)),
        # PT: fatos posteriores sobrescrevem — SF single-hop.
        # EN: later facts overwrite — SF single-hop.
        Turn("user", f"Update: I moved to {c2} and now I work as a {j2}.",
             _ts(3)),
    ]
    items = [BenchItem("SF", list(turns),
                       f"Where does {name} live now?", c2),
             BenchItem("SF", list(turns),
                       f"What is {name}'s current job?", j2)]
    # SF multi-hop: cidade atual + animal
    items.append(BenchItem(
        "SF", list(turns),
        f"In {name}'s current city, what pet do they have?", pn))
    return items


def generate(seed: int = 0, n_personas: int = 8) -> BenchSet:
    """PT: gera o bench sintético: AR + TTL + LRU + SF por persona.
    EN: generate the synthetic bench: AR + TTL + LRU + SF per persona."""
    rng = random.Random(seed)
    names = ["Pedro", "Ana", "Luis", "Marta", "Joao", "Sofia", "Rui",
             "Ines", "Tiago", "Lara"][:n_personas]
    items: list[BenchItem] = []
    for name in names:
        items.append(_stream_ar(rng, name))
        items.append(_stream_ttl(rng))
        items.append(_stream_lru(rng, name))
        items += _stream_sf(rng, name)
    return BenchSet(items)


def coverage_score(gold: str, pred: str) -> float:
    """PT: cobertura de palavras-chave do ouro na predição (LRU).
    EN: gold keyword coverage in the prediction (LRU)."""
    g = [w for w in re.split(r"[^A-Za-z0-9]+", gold.lower()) if w]
    p = set(re.split(r"[^A-Za-z0-9]+", pred.lower()))
    return sum(1 for w in g if w in p) / len(g) if g else 0.0


def score(llm: LLM, item: BenchItem, pred: str) -> float:
    if item.comp == "LRU":
        return coverage_score(item.gold, pred)
    return float(judge_answer(llm, item.question, item.gold, pred))


def answer(mem, item: BenchItem, llm: LLM, k: int = 10) -> str:
    """PT: responde usando só o que a memória devolve (a pergunta chega no
    fim — protocolo incremental). EN: answer using only what memory returns
    (question arrives at the end — incremental protocol)."""
    ev = mem.retrieve(item.question, k=k, now=None)
    return answer_with_evidence(llm, item.question, ev)


def ingest(mem, stream: list[Turn]) -> None:
    for t in stream:
        mem.add(MemoryItem(t.text, t.ts, {}))
