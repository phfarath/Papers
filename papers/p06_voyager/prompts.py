"""PT: prompts + mock handlers do p06_voyager (Voyager, arXiv 2305.16291).

EN: prompts + mock handlers for p06_voyager (Voyager, arXiv 2305.16291).

Handler tasks (honesty rule — recebem só o texto do prompt / prompt text only):
- ``voyager.curriculum``: propõe a próxima tarefa do currículo automático
  (§3.2 / curriculum module).
- ``voyager.write_code``: gera o código da skill (iterative prompting,
  §3.3) — usa skills recuperadas, feedback do env e erros anteriores.
- ``voyager.critic``: self-verification — julga se a tarefa foi cumprida.
- ``voyager.suggest``: escreve a descrição da skill verificada para indexação
  na skill library (§3.3, embedding by description).
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.envs.craft import ITEMS, MINE_REQ, RECIPES, SMELTS

# PT: cadeia de marcos da árvore tecnológica (ordem do currículo implícito).
# EN: tech-tree milestone chain (implicit curriculum order).
CHAIN: list[str] = [
    "log", "planks", "stick", "crafting_table", "wooden_pickaxe",
    "cobblestone", "stone_pickaxe", "iron_ore", "furnace",
    "iron_ingot", "iron_pickaxe", "diamond",
]

_ACTION = {"mine": MINE_REQ, "craft": RECIPES, "smelt": SMELTS}


def _verb(item: str) -> str:
    for v, table in _ACTION.items():
        if item in table:
            return v
    return "mine"


def _parse_section(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z][a-z ]+:|\Z)", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


def _inv_of(prompt: str) -> dict[str, int]:
    inv: dict[str, int] = {}
    for k, v in re.findall(r"'?(\w+)'?:\s*(\d+)",
                           _parse_section(prompt, "Inventory")):
        inv[k] = int(v)
    return inv


@mock_handler("voyager.curriculum")
def _curriculum(prompt: str, rng: random.Random) -> str:
    """PT: currículo automático — dado inventário + tarefas completadas/
    falhas, propõe o próximo marco não alcançado da árvore (§3.2).

    EN: automatic curriculum — given inventory + completed/failed tasks,
    proposes the next unreached tech-tree milestone.
    """
    inv = _inv_of(prompt)
    done = set(re.findall(r"\b(\w+)\b",
                          _parse_section(prompt, "Completed tasks")))
    for item in CHAIN:
        if item not in inv and item not in done:
            return f"{_verb(item)} {item}"
    # PT: árvore completa → propõe o alvo do episódio (ex.: torch no zero-shot).
    # EN: chain complete → propose the episode target (e.g. torch zero-shot).
    target = _parse_section(prompt, "Episode target")
    if target and inv.get(target, 0) == 0:
        return f"{_verb(target)} {target}"
    return "explore the cave"


def _needs(item: str) -> dict[str, int]:
    """PT: demanda total de cada item para obter `item` (pré-requisitos
    recursivos). EN: total demand per item to obtain `item` (recursive
    prerequisites)."""
    needs: dict[str, int] = {}

    def demand(it: str, n: int) -> None:
        needs[it] = needs.get(it, 0) + n
        if it in MINE_REQ:
            req = MINE_REQ[it]
            if req:
                demand(req, 1)
        elif it in RECIPES:
            if "*" in RECIPES[it]:
                demand("crafting_table", 1)
            for src, per in RECIPES[it].items():
                if src != "*":
                    demand(src, per * n)
        elif it in SMELTS:
            demand("furnace", 1)
            demand(SMELTS[it][0], SMELTS[it][1] * n)

    demand(item, 1)
    return needs


def _lines_for(needs: dict[str, int]) -> list[str]:
    # PT: emite na ordem da árvore — insumos antes de quem os consome.
    # EN: emit in tech-tree order — inputs before their consumers.
    order = CHAIN + [i for i in needs if i not in CHAIN]
    return [f"bot.{_verb(it)}('{it}', {needs[it]})" for it in order
            if it in needs]


LINE_BUG_P = 0.22  # PT: prob. de bug por linha emitida. EN: per-line bug prob.


def _mutate(line: str, rng: random.Random) -> str:
    """PT: muta uma linha do código — erro de execução OU falha silenciosa
    (item errado: roda sem exceção mas a tarefa não é cumprida; é isso que o
    critic pega). EN: mutate one code line — execution error OR silent
    wrong-item failure (runs fine but the task fails; the critic catches it)."""
    m = re.match(r"bot\.(mine|craft|smelt)\('(\w+)', (\d+)\)", line)
    if not m:
        return line
    verb, it, n = m.groups()
    mode = rng.random()
    if mode < 0.4:
        # PT: erro de execução — mina/crafta algo inválido. EN: execution
        # error — mine/craft something invalid.
        return f"bot.{verb}('unobtainium', {n})"
    # PT: falha silenciosa — produz o item errado (mas válido). EN: silent
    # failure — produces a wrong (but valid) item.
    other = rng.choice(sorted(ITEMS - {it}))
    return f"bot.{_verb(other)}('{other}', {n})"


@mock_handler("voyager.write_code")
def _write_code(prompt: str, rng: random.Random) -> str:
    """PT: gera código (§3.3 iterative prompting). Se uma skill recuperada já
    resolve a tarefa, reutiliza seu código (reuse da skill library). Na 1ª
    tentativa sem skill, omite pré-requisitos → erro de execução; com o
    feedback do erro no prompt (rounds 2+), gera a cadeia completa.

    EN: generates code (§3.3 iterative prompting). A retrieved matching skill
    is reused verbatim. Without it, attempt 1 skips prerequisites → execution
    error; with the error feedback in the prompt (rounds 2+), the full chain
    is generated.
    """
    task = _parse_section(prompt, "Task")
    item = task.split()[-1] if task.split() else "log"
    skills_txt = _parse_section(prompt, "Relevant skills")
    # PT: reutiliza o código de uma skill que já obtém o item. EN: reuse the
    # code of a retrieved skill that already obtains the item.
    blocks = re.findall(r"- (\w+) — .*?\n```(.*?)```", skills_txt,
                        flags=re.S)
    needs = _needs(item)
    # PT: reuso por PREFIXO — uma skill recuperada cobre o próprio item como
    # sub-meta; executa até a linha que o produz (§3.3).
    # EN: PREFIX reuse — a retrieved skill covers the item itself as a
    # sub-goal; run up to the line producing it.
    for _name, blk in blocks:
        blines = blk.strip().splitlines()
        for i, ln in enumerate(blines):
            if re.match(rf"bot\.(?:mine|craft|smelt)\('{item}'", ln.strip()):
                return "\n".join(blines[:i + 1])
    # PT: COMPOSIÇÃO — uma skill recuperada cobre um PRÉ-REQUISITO: seu
    # código verificado é emendado intacto e só as linhas novas (delta) são
    # geradas do zero, com bug por linha. Sem biblioteca, TODA a cadeia é
    # gerada — cadeias longas falham muito mais (Fig. do paper).
    # EN: COMPOSITION — a retrieved skill covers a PREREQUISITE: its verified
    # code is spliced in intact and only the fresh delta lines get per-line
    # bugs. Without the library, the WHOLE chain is generated — long chains
    # fail far more often.
    def _produced(code: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for it, n in re.findall(
                r"bot\.(?:mine|craft|smelt)\('(\w+)', (\d+)\)", code):
            out[it] = out.get(it, 0) + int(n)
        return out

    reuse_code = ""
    produced: dict[str, int] = {}
    for name, blk in blocks:
        if name in needs and name != item:
            p = _produced(blk)
            if len(p) > len(produced):
                reuse_code, produced = blk.strip(), p
    delta = {it: n - produced.get(it, 0)
             for it, n in needs.items() if n - produced.get(it, 0) > 0}
    lines = [_mutate(ln, rng) if rng.random() < LINE_BUG_P else ln
             for ln in _lines_for(delta)]
    return (reuse_code + "\n" if reuse_code else "") + "\n".join(lines)


@mock_handler("voyager.critic")
def _critic(prompt: str, rng: random.Random) -> str:
    """PT: self-verification (§3.3): 'success' se o item alvo está no
    inventário, senão 'fail: <motivo>'. EN: self-verification — 'success' if
    the target item is in the inventory, else 'fail: <reason>'."""
    task = _parse_section(prompt, "Task")
    item = task.split()[-1] if task.split() else ""
    inv = _inv_of(prompt)
    if inv.get(item, 0) > 0:
        return "success"
    err = _parse_section(prompt, "Last execution error") or "item missing"
    return f"fail: {err}"


@mock_handler("voyager.suggest")
def _suggest(prompt: str, rng: random.Random) -> str:
    """PT: descrição curta da skill para indexação por embedding (§3.3).
    EN: short skill description for embedding-based indexing."""
    task = _parse_section(prompt, "Task")
    item = task.split()[-1] if task.split() else "item"
    return f"obtain {item}: reusable routine to get {item} in CraftWorld"
