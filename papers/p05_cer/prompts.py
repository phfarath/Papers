"""CER — prompts e handlers mock / prompts and mock handlers.

PT: handlers do MockLLM para o CER (Contextual Experience Replay):
- "cer.dynamics": destila DYNAMICS — resumo de páginas + URL + usos possíveis.
- "cer.skills": destila SKILLS — resumo abstrato com placeholders ({item}) +
  guia passo a passo com exemplos de ação. Recebe o buffer existente para
  evitar duplicatas ("Summarized before", como no prompt A.1 do paper).
- "cer.retr.dyn" / "cer.retr.skl": retrieval separado (top-k por relevância).
- "web.act": política ReAct do MiniWeb — escolhe a próxima ação dentre as
  válidas; quando a informação pedida já está visível, emite "stop <answer>".
  A memória de dynamics pode sugerir URLs que não estão na lista de gotos
  (pegadinha: /git/admin não é linkado nem listado).

REGRA DE HONESTIDADE: handlers só leem o prompt (+rng).

EN: MockLLM handlers for CER (see PT). HONESTY RULE: handlers only read the
prompt (+rng).
"""

from __future__ import annotations

import random
import re

from papers.common.llm import mock_handler
from papers.common.utils import token_set

# ---------------------------------------------------------------------------
# Destilação / distillation
# ---------------------------------------------------------------------------

# PT: resumo-estático de cada página do MiniWeb — o "conhecimento de mundo" do
# mock sobre o que cada página faz. EN: static per-page summary — the mock's
# world knowledge of what each MiniWeb page does.
_PAGE_USES = {
    "shop": ("Shop home page", "starting point; links to the search page"),
    "shop/search": ("Product search page",
                    "type a product name in the search box to list items "
                    "and their prices"),
    "shop/item": ("Item detail page",
                  "shows the item price; has the add-to-cart button"),
    "shop/cart": ("Shopping cart page", "shows how many items were added"),
    "forum": ("Forum home", "lists topics; links to the login page"),
    "forum/login": ("Login page", "fill username+password to enable posting"),
    "forum/topic": ("Topic page", "shows replies; reply box needs login"),
    "git": ("GitLab-lite home", "lists projects; links to the issues page"),
    "git/issues": ("Issues page", "counts open issues"),
    "git/admin": ("Admin dashboard (UNLINKED url)",
                  "reports the internal metrics dashboard title; only "
                  "reachable by typing the URL directly"),
}

_KEY = {
    "shop/search": ("goto", "type"), "shop/item": ("click",), "shop/cart": (),
    "forum/login": ("type",), "forum/topic": ("type", "click"),
}


@mock_handler("cer.dynamics")
def _dyn_distill(prompt: str, rng: random.Random) -> str:
    """PT: lê a trajetória (linhas "url: "/"text: ") e resume cada página vista
    + URL + usos. Páginas já no buffer viram "Summarized before".

    EN: reads the trajectory ("url: "/"text: " lines) and summarizes each seen
    page + URL + usages. Pages already in the buffer → "Summarized before".
    """
    existing = set()
    m = re.search(r"existing dynamics:\n((?:- .*\n?)*)", prompt.lower())
    if m:
        existing = set(re.findall(r"http://\S+", m.group(1)))
    out: list[str] = []
    seen: set[str] = set()
    for um in re.finditer(r"url: (\S+)", prompt.lower()):
        url = um.group(1)
        if url in seen:
            continue
        seen.add(url)
        key = next((k for k in _PAGE_USES if url.endswith("/" + k)), None)
        if key is None:
            continue
        if url in existing:
            out.append(f"- {url} — Summarized before")
            continue
        title, uses = _PAGE_USES[key]
        out.append(f"- {url} — {title}; usage: {uses}")
    return "\n".join(out) or "(no pages visited)"


# PT: padrões de skill abstratos — o mock generaliza a sequência de ações em
# workflows com placeholders, como o prompt A.1 do paper pede.
# EN: abstract skill patterns — the mock generalizes the action sequence into
# workflows with placeholders, as the paper's A.1 prompt requires.
_SKILL_PATTERNS = [
    ("buy item {product}", "shop",
     ["goto http://mini.web/shop/search",
      "type in_q {product}", "click lnk_item_{product}",
      "click btn_add_cart", "goto http://mini.web/shop/cart"]),
    ("find price of {product}", "shop/item",
     ["goto http://mini.web/shop/search", "type in_q {product}",
      "click lnk_item_{product}"]),
    ("log in and post reply {text} on topic {topic}", "forum",
     ["goto http://mini.web/forum/login", "type in_user demo",
      "type in_pass demo", "goto http://mini.web/forum/topic/{topic}",
      "type in_reply {text}"]),
    ("count open issues", "git/issues",
     ["goto http://mini.web/git/issues"]),
    ("open the hidden admin dashboard", "git/admin",
     ["goto http://mini.web/git/admin"]),
]

_TRIGGERS = {
    "git/admin": ("type in_reply" not in "", "git/admin"),
}


def _skills_from_trajectory(prompt: str) -> list[str]:
    """PT: extrai skills da trajetória pelas URLs/ações observadas.
    EN: extracts skills from the trajectory by observed URLs/actions."""
    urls = set(re.findall(r"url: (\S+)", prompt.lower()))
    acts = set(re.findall(r"a: (.+)", prompt.lower()))
    out: list[str] = []
    if any(u.endswith("/git/admin") for u in urls):
        out.append(
            "<skill>\nopen the hidden admin dashboard\n"
            "steps:\n1. type the unlinked URL directly: "
            "goto http://mini.web/git/admin\n2. read the page title\n</skill>")
    if any(u.endswith("/shop/cart") for u in urls) or \
            "click btn_add_cart" in " ".join(acts):
        out.append(
            "<skill>\nbuy item {product}\n"
            "steps:\n1. go to the search page: goto http://mini.web/shop/search\n"
            "2. search the product: type in_q {product}\n"
            "3. open the item: click lnk_item_{product}\n"
            "4. add it: click btn_add_cart\n"
            "5. check the cart: goto http://mini.web/shop/cart\n</skill>")
    if any("/shop/item/" in u for u in urls):
        out.append(
            "<skill>\nfind price of {product}\n"
            "steps:\n1. goto http://mini.web/shop/search\n"
            "2. type in_q {product}\n"
            "3. open the item: click lnk_item_{product}\n"
            "4. read the Price field\n</skill>")
    if any(u.endswith("/forum/topic/t1") for u in urls) and \
            any("in_reply" in a for a in acts):
        out.append(
            "<skill>\nlog in and post reply {text} on topic {topic}\n"
            "steps:\n1. goto http://mini.web/forum/login\n"
            "2. type in_user demo\n3. type in_pass demo\n"
            "4. goto http://mini.web/forum/topic/{topic}\n"
            "5. post the reply: type in_reply {text}\n</skill>")
    if any(u.endswith("/git/issues") for u in urls):
        out.append(
            "<skill>\ncount open issues\n"
            "steps:\n1. goto http://mini.web/git/issues\n"
            "2. read the 'N open' counter\n</skill>")
    return out


@mock_handler("cer.skills")
def _skl_distill(prompt: str, rng: random.Random) -> str:
    """PT: destila skills abstratas da trajetória; nomes já no buffer viram
    "Summarized before" (dedup do paper, prompt A.1 regra 2).

    EN: distills abstract skills; names already in the buffer → "Summarized
    before" (the paper's dedup rule, prompt A.1 rule 2).
    """
    existing = set(re.findall(r"^- ([a-z{} ]+)$",
                            prompt.lower().split("existing skills:")[-1],
                            flags=re.M))
    out = []
    for sk in _skills_from_trajectory(prompt):
        name = sk.split("\n")[1]
        out.append(f"<skill>\n{name}\nsteps:\nSummarized before\n</skill>"
                   if name in existing else sk)
    return "\n".join(out) or "(no skills extracted)"


# ---------------------------------------------------------------------------
# Retrieval separado / separate retrieval
# ---------------------------------------------------------------------------

def _rank(prompt: str, section: str, k: int = 5) -> str:
    """PT: top-k entradas por overlap com o 'Goal:'. EN: top-k by goal overlap."""
    goal = ""
    cands: list[str] = []
    cur = None
    for line in prompt.splitlines():
        low = line.lower()
        if low.startswith("goal:"):
            goal = line[5:].strip()
        elif low.startswith(section):
            cur = "on"
        elif line.startswith("- ") and cur == "on":
            cands.append(line[2:].strip())
        elif not line.startswith("- ") and cur == "on" and line.strip():
            cur = None
    gt = token_set(goal)
    scored = sorted(cands,
                    key=lambda c: len(gt & token_set(re.sub(r"[{}]", "", c))),
                    reverse=True)
    return "\n".join(f"- {c}" for c in scored[:k])


@mock_handler("cer.retr.dyn")
def _retr_dyn(prompt: str, rng: random.Random) -> str:
    return _rank(prompt, "available dynamics", 5)


@mock_handler("cer.retr.skl")
def _retr_skl(prompt: str, rng: random.Random) -> str:
    return _rank(prompt, "available skills", 5)


# ---------------------------------------------------------------------------
# Política MiniWeb (task "web.act")
# ---------------------------------------------------------------------------

# PT: extrai a resposta do texto da página quando a informação já está visível.
# EN: extracts the answer from the page text once the info is visible.
def _extract_answer(task: str, obs: str, recent: list[str]) -> str | None:
    low_t, low_o = task.lower(), obs
    if "price" in low_t:
        m = re.search(r"\$\d+", low_o)
        return m.group(0) if m else None
    if "cart size" in low_t or "cart" in low_t:
        # PT: só responde o tamanho depois de adicionar algo. EN: only answer
        # the cart size after an add-to-cart action.
        if not any("btn_add_cart" in r for r in recent):
            return None
        m = re.search(r"Cart \((\d+) items", low_o)
        return m.group(1) if m else None
    if "title" in low_t:
        m = re.search(r"\n([A-Z][A-Za-z ]+) —", low_o)
        return m.group(1).strip() if m else None
    if "replies" in low_t:
        # PT: só conta replies depois de postar. EN: only count replies after
        # actually posting one.
        if not any("in_reply" in r for r in recent):
            return None
        m = re.search(r"Replies \((\d+)\)", low_o)
        return m.group(1) if m else None
    if "issues" in low_t:
        m = re.search(r"(\d+) open", low_o)
        return m.group(1) if m else None
    return None


@mock_handler("web.act")
def _web_act(prompt: str, rng: random.Random) -> str:
    """PT: política ReAct do MiniWeb. Seções do prompt: Task, Observation,
    Dynamics (memória), Skills (memória), Recent actions, Valid actions.

    EN: MiniWeb ReAct policy. Prompt sections: Task, Observation, Dynamics,
    Skills, Recent actions, Valid actions.
    """
    section = None
    task = obs = ""
    actions: list[str] = []
    dynamics: list[str] = []
    skills: list[str] = []
    recent: list[str] = []
    for line in prompt.splitlines():
        low = line.strip().lower()
        if low.startswith("task:"):
            task, section = line[5:].strip(), None
        elif low.startswith("observation:"):
            obs, section = "", "obs"
        elif low.startswith("valid actions"):
            section = "act"
        elif low.startswith("dynamics"):
            section = "dyn"
        elif low.startswith("skills"):
            section = "skl"
        elif low.startswith("recent actions"):
            section = "rec"
        elif line.startswith("- "):
            if section == "act":
                actions.append(line[2:].strip())
            elif section == "dyn":
                dynamics.append(line[2:].strip())
            elif section == "skl":
                skills.append(line[2:].strip())
            elif section == "rec":
                recent.append(line[2:].strip())
        elif section == "obs":
            obs += line + "\n"
    if not actions:
        return "stop nothing"
    tt = token_set(task)

    # PT: 1) se a informação pedida já está visível, parar com a resposta.
    # EN: 1) if the asked info is already visible, stop with the answer.
    ans = _extract_answer(task, obs, recent)
    if ans is not None:
        return f"stop {ans}"

    # PT: 2) dynamics sugere URL desconhecida? Se um dynamics cita uma URL com
    # tokens da tarefa (ex.: admin), navegar direto — a pegadinha do /git/admin.
    # EN: 2) a dynamics line cites a URL matching the task (e.g. admin)? Go
    # there directly — the /git/admin gotcha.
    for d in dynamics:
        for url in re.findall(r"http://\S+", d):
            toks = token_set(url.replace("http://", " ").replace("/", " ")
                             .replace(".", " "))
            if tt & (toks - {"mini", "web", "http", "shop", "search", "forum",
                             "git", "issues", "cart", "login", "topic"}):
                return f"goto {url}"

    # PT: 3) ação que aparece numa skill recuperada → seguir o workflow NA
    # ORDEM (a ação que casa o passo mais cedo do guia vence).
    # EN: 3) an action appearing in a retrieved skill → follow the workflow IN
    # ORDER (the action matching the earliest step wins).
    skl_text = " ".join(skills).lower()
    # PT: passos concretos extraídos do guia (goto <url> | click <id> |
    # type <el> <arg>) em ordem; placeholders {x} casam qualquer valor.
    # EN: concrete steps extracted from the guide in order; {x} placeholders
    # match any value.
    steps = [(m.start(), m.group(0))
             for m in re.finditer(
                 r"(?:goto|click|type|stop) [\w{}:/.\-]+(?: \{?\w+\}?)?",
                 skl_text)]

    def _matches(action: str, step: str) -> bool:
        # PT: casa ação válida com passo do guia; {x} casa qualquer valor e
        # "type el <text>" casa "type el {arg}". EN: match a valid action to
        # a guide step; {x} wildcards any value.
        if action.startswith("type "):
            return step.startswith(action.replace("<text>", "").strip() + " ")
        pat = re.escape(step)
        pat = re.sub(r"\\\{[^}]+\\\}", r"\\S+", pat)
        return re.fullmatch(pat, action) is not None

    # PT: posição do último passo já executado (por recent) — a skill avança
    # a partir dele. EN: position of the last already-executed step (via
    # recent actions) — the skill resumes after it.
    done_pos = -1
    for pos, st in steps:
        if any(_matches(r, st) or
               (r.startswith("type ") and st.startswith(
                   r.split()[0] + " " + r.split()[1]))
               for r in recent):
            done_pos = pos
    cands: list[tuple[int, str]] = []
    for a in actions:
        if a.startswith("stop"):
            continue
        for pos, st in steps:
            if pos > done_pos and _matches(a, st):
                cands.append((pos, a))
                break
    if cands:
        cands.sort()
        return _fill_text(task, cands[0][1])

    # PT: 4) heurística genérica: overlap ação×tarefa/observação - repetição.
    # EN: 4) generic heuristic: action×task/obs overlap - repetition.
    best_a, best_s = "stop unknown", -1e9
    for a in actions:
        at = token_set(a)
        s = 2.0 * len(at & tt) + 0.3 * len(at & token_set(obs))
        if a.startswith("stop"):
            s -= 5.0
        s -= 1.5 * sum(1 for r in recent if r == a)
        s += rng.gauss(0, 0.3)
        if s > best_s:
            best_s, best_a = s, a
    return _fill_text(task, best_a)


def _fill_text(task: str, action: str) -> str:
    """PT: preenche <text> com o argumento da tarefa (produto, usuário, senha,
    texto do reply). EN: fills <text> with the task's argument."""
    if not action.startswith("type "):
        return action
    el = action.split()[1]
    low = task.lower()
    if el in ("in_user", "in_pass"):
        return f"type {el} demo"
    if el == "in_reply":
        m = re.search(r"reply '([^']+)'", low)
        return f"type {el} {m.group(1) if m else 'hi'}"
    if el == "in_q":
        m = (re.search(r"of the (\w+)", low)
             or re.search(r"the (\w+)[?!. ]*$", low.strip())
             or re.search(r"the (\w+)", low))
        return f"type {el} {m.group(1) if m else 'all'}"
    return f"type {el} x"
