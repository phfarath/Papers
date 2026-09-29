"""MiniWeb — WebArena-lite deterministico (CER, LongMemEval-V2).

PT: 3 sites simulados (shop, forum, gitlab-lite) com URLs, links, busca,
formulários e "pegadinhas" reais: itens só aparecem pela busca com filtro,
página de admin em URL não linkada, botão de postar só após login, e estado que
muda (carrinho, issues). Checagem de tarefa é programática, por `check`.

EN: 3 simulated sites (shop, forum, gitlab-lite) with URLs, links, search,
forms and real gotchas: items only via filtered search, an admin page at an
unlinked URL, a post button only after login, and mutable state (cart, issues).
Task checking is programmatic, via `check`.
"""

from __future__ import annotations

from dataclasses import dataclass

from papers.envs import StepResult

BASE = "http://mini.web"


@dataclass
class WebTask:
    """PT: tarefa do MiniWeb. EN: a MiniWeb task."""

    task_id: str
    description: str
    gold: str


TASKS: list[WebTask] = [
    WebTask("w01", "What is the price of the laptop?", "$999"),
    WebTask("w02", "Add the keyboard to the cart, then report the cart size.", "1"),
    WebTask("w03", "Find the secret admin dashboard and report its title.",
            "Admin Dashboard"),
    WebTask("w04", "Log in (user demo / pass demo) and post the reply 'hello' "
                   "on topic t1, then report the number of replies.", "3"),
    WebTask("w05", "How many open issues are there in gitlab-lite?", "2"),
]


class MiniWeb:
    """PT: env web deterministico. EN: deterministic web env."""

    max_steps = 40

    def __init__(self, task: WebTask) -> None:
        self.task = task
        self.task_description = task.description
        self.reset()

    # ---------- estado / state ----------
    def reset(self) -> str:
        self.url = f"{BASE}/shop"
        self.logged_in = False
        self.cart: list[str] = []
        self.issues = {"i1": "open", "i2": "open", "i3": "closed"}
        self.replies = {"t1": ["nice post", "agreed"]}
        self.search_results: list[str] = []
        self.steps = 0
        self.done = False
        self.answer: str | None = None
        self._typed: dict[str, str] = {}
        return self._observe()

    # ---------- páginas / pages ----------
    def _page(self) -> tuple[str, dict[str, str], dict[str, str]]:
        """PT: (texto, links id->url, inputs id->label). EN: (text, links, inputs)."""
        u = self.url
        if u == f"{BASE}/shop":
            return ("Shop home. Use search to find products.",
                    {"lnk_search": f"{BASE}/shop/search"}, {"in_q": "search box"})
        if u.startswith(f"{BASE}/shop/search"):
            items = ("laptop $999", "keyboard $49", "mouse $19")
            res = ", ".join(items) if self.search_results else "(none — type a query first)"
            text = "Search results: " + res
            links = {f"lnk_item_{i.split()[0]}": f"{BASE}/shop/item/{i.split()[0]}"
                     for i in items if i.split()[0] in " ".join(self.search_results)}
            return (text, links, {"in_q": "search box"})
        if u.startswith(f"{BASE}/shop/item/"):
            name = u.rsplit("/", 1)[-1]
            prices = {"laptop": "$999", "keyboard": "$49", "mouse": "$19"}
            return (f"Item page: {name}. Price: {prices.get(name, 'n/a')}.",
                    {"btn_add_cart": f"__add:{name}", "lnk_cart": f"{BASE}/shop/cart"}, {})
        if u == f"{BASE}/shop/cart":
            return (f"Cart ({len(self.cart)} items): {', '.join(self.cart)}",
                    {"lnk_home": f"{BASE}/shop"}, {})
        if u == f"{BASE}/forum":
            links = {"lnk_t1": f"{BASE}/forum/topic/t1", "lnk_login": f"{BASE}/forum/login"}
            return ("Forum home. Topics: t1 'Best keyboards'.", links, {})
        if u == f"{BASE}/forum/login":
            if self.logged_in:
                return ("You are logged in as demo.", {"lnk_home": f"{BASE}/forum"}, {})
            return ("Login page.", {}, {"in_user": "username", "in_pass": "password"})
        if u == f"{BASE}/forum/topic/t1":
            links = {"lnk_home": f"{BASE}/forum"}
            text = (f"Topic t1 'Best keyboards'. Replies ({len(self.replies['t1'])}): "
                    + "; ".join(self.replies["t1"]))
            if self.logged_in:
                return (text + " [post box available]", links, {"in_reply": "reply box"})
            return (text + " (log in to reply)", links, {})
        if u == f"{BASE}/git":
            return ("GitLab-lite. Projects: papers.",
                    {"lnk_issues": f"{BASE}/git/issues"}, {})
        if u == f"{BASE}/git/issues":
            opens = [k for k, s in self.issues.items() if s == "open"]
            return (f"Issues: {len(opens)} open ({', '.join(opens)}).",
                    {"lnk_git": f"{BASE}/git"}, {})
        if u == f"{BASE}/git/admin":
            return ("Admin Dashboard — internal metrics.", {"lnk_git": f"{BASE}/git"}, {})
        return ("404 Not Found.", {"lnk_home": f"{BASE}/shop"}, {})

    def _observe(self) -> str:
        text, links, inputs = self._page()
        el = ", ".join(list(links) + list(inputs)) or "none"
        return f"URL: {self.url}\n{text}\nElements: {el}"

    def valid_actions(self) -> list[str]:
        _, links, inputs = self._page()
        acts = [f"click {i}" for i in links]
        acts += [f"type {i} <text>" for i in inputs]
        known = [f"{BASE}/shop", f"{BASE}/shop/search", f"{BASE}/forum",
                 f"{BASE}/git", f"{BASE}/git/issues", f"{BASE}/shop/cart"]
        acts += [f"goto {u}" for u in known]
        acts.append("stop <answer>")
        return sorted(acts)

    # ---------- passo / step ----------
    def step(self, action: str) -> StepResult:
        if self.done:
            return StepResult("Episode is over.", 0.0, True)
        self.steps += 1
        a = action.strip()
        msg = ""
        if a.startswith("goto "):
            self.url = a[5:].strip()
        elif a.startswith("click "):
            msg = self._click(a[6:].strip())
        elif a.startswith("type "):
            msg = self._type(a[5:].strip())
        elif a.startswith("stop"):
            self.answer = a[4:].strip()
            self.done = True
        else:
            msg = f"Unknown action {a!r}."
        obs = (msg + "\n" if msg else "") + self._observe()
        ok = self.done and self.answer is not None and \
            self.task.gold.lower() in self.answer.lower()
        return StepResult(obs, 1.0 if ok else 0.0,
                          self.done or self.steps >= self.max_steps)

    def _click(self, el: str) -> str:
        text, links, _ = self._page()
        if el in links:
            target = links[el]
            if target.startswith("__add:"):
                self.cart.append(target[6:])
                return f"Added {target[6:]} to cart."
            self.url = target
            return ""
        # PT: submit de login é um click implícito após preencher o form.
        # EN: login submit is an implicit click after filling the form.
        if el == "btn_login" and self.url == f"{BASE}/forum/login":
            self.logged_in = True
            return "Logged in as demo."
        if el == "btn_post" and self.logged_in:
            self.replies["t1"].append(self._typed.get("in_reply", ""))
            return "Reply posted."
        return f"Element {el} not found on this page."

    _typed: dict[str, str]

    def _type(self, rest: str) -> str:
        el, _, text = rest.partition(" ")
        _, _, inputs = self._page()
        if el not in inputs:
            return f"Input {el} not found on this page."
        self._typed[el] = text
        if el == "in_q":
            q = text.lower()
            self.search_results = [
                i for i in ("laptop", "keyboard", "mouse") if q in i or q == "all"
            ]
            return f"Search for '{text}'."
        if el == "in_user":
            if text != "demo":
                return "Unknown user."
            return "Username ok."
        if el == "in_pass":
            if text == "demo" and self._typed.get("in_user") == "demo":
                self.logged_in = True
                return "Logged in as demo."
            return "Wrong password."
        if el == "in_reply":
            self.replies["t1"].append(text)
            return "Reply posted."
        return "Typed."

    def score(self) -> float:
        return 1.0 if self.done and self.answer and \
            self.task.gold.lower() in self.answer.lower() else 0.0
