"""PT: método do p07_muse_autoskill — ciclo de vida de skills como diretórios
+ contexto como DAG comprimido (arXiv 2605.27366 — paper DIFERENTE do MUSE de
p04, Valiente & Pilly).

EN: p07_muse_autoskill method — skills-as-directories lifecycle + context as a
compressed DAG (a DIFFERENT paper from p04's MUSE).

Skill dir layout (§4.1):
    <name>/SKILL.md      frontmatter: name/description/inputs/outputs
    <name>/scripts/skill.py
    <name>/tests/test_skill.py
    <name>/.memory.md    notas de aprendizado / learning notes

Lifecycle (§4.3): create → test-gated register → use → memory → evaluate →
refine (from error trace) → merge near-duplicates → prune low-utility.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from papers.common.llm import LLM, task_prompt
from papers.common.utils import token_set

KEEP_FIRST = "KEEP_FIRST"   # PT: nó que nunca sai do início. EN: pinned head.
KEEP_LAST = "KEEP_LAST"     # PT: nó que nunca sai do fim. EN: pinned tail.


def _tok_len(text: str) -> int:
    return len(text.split())


@dataclass
class SkillDir:
    """PT: skill materializada em disco. EN: skill materialized on disk."""
    name: str
    path: Path

    @property
    def script(self) -> Path:
        return self.path / "scripts" / "skill.py"

    @property
    def test_file(self) -> Path:
        return self.path / "tests" / "test_skill.py"

    @property
    def memory_file(self) -> Path:
        return self.path / ".memory.md"

    def description(self) -> str:
        m = re.search(r"description: (.*)",
                      (self.path / "SKILL.md").read_text())
        return m.group(1) if m else self.name

    def run(self, arg: str) -> tuple[bool, str]:
        """PT: executa o script em subprocesso (sandbox simples — processo
        isolado, timeout). EN: run the script in a subprocess (simple sandbox
        — isolated process, timeout)."""
        try:
            res = subprocess.run(
                [sys.executable, str(self.script), arg],
                capture_output=True, text=True, timeout=10, check=False)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if res.returncode != 0:
            return False, res.stderr.strip() or f"exit {res.returncode}"
        return True, res.stdout.strip()

    def gate_test(self) -> bool:
        """PT: test-gated register (§4.3) — a skill só é registrada se o seu
        próprio teste passar. EN: register only if the skill's own test
        passes."""
        if not self.test_file.exists():
            return False
        res = subprocess.run([sys.executable, str(self.test_file)],
                             capture_output=True, text=True, timeout=10,
                             check=False)
        return res.returncode == 0


class SkillStore:
    """PT: loja de skills com ciclo de vida completo (§4.3).
    EN: skill store with the full lifecycle."""

    def __init__(self, root: Path, lifecycle: bool = True) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lifecycle = lifecycle
        self.skills: dict[str, SkillDir] = {}
        self.uses: dict[str, int] = {}
        self.fails: dict[str, int] = {}
        self.merged = 0
        self.pruned = 0

    # ---------- create + register ----------
    def create(self, llm: LLM, task: str) -> SkillDir | None:
        """PT: escreve SKILL.md + script + teste; se lifecycle, só registra
        após o teste passar (test-gated). EN: write skill files; with
        lifecycle, register only after its test passes."""
        out = llm.complete(task_prompt("muse.write_skill", f"Task: {task}"))
        m = re.search(r"name: (\w+)", out)
        name = m.group(1) if m else "helper"
        sd = SkillDir(name, self.root / name)
        (sd.path / "scripts").mkdir(parents=True, exist_ok=True)
        (sd.path / "tests").mkdir(parents=True, exist_ok=True)
        fm = out.split("<script>")[0].strip() + "\n"
        (sd.path / "SKILL.md").write_text(fm)
        script = re.search(r"<script>\n(.*?)\n</script>", out, flags=re.S)
        (sd.script).write_text(script.group(1) if script else "print('')")
        tcode = re.search(r"<test>\n(.*?)\n</test>", out, flags=re.S)
        test_body = (tcode.group(1) if tcode else "def test(s): return s")
        a = re.search(r"<arg>(.*?)</arg>", out, flags=re.S)
        sample = a.group(1) if a else "abc"
        # PT: o teste executa o script real com um input-fixture do criador.
        # EN: the test executes the real script on the creator's fixture arg.
        (sd.test_file).write_text(
            "import subprocess, sys\n"
            + test_body + "\n"
            "res = subprocess.run([sys.executable, "
            f"'{sd.script}', '{sample}'], capture_output=True, text=True)\n"
            "assert res.returncode == 0, res.stderr\n"
            f"assert res.stdout.strip() == str(test('{sample}'))\n")
        if self.lifecycle and not sd.gate_test():
            return None
        if name in self.skills:
            return self.skills[name]  # PT: já registrada. EN: already known.
        self.skills[name] = sd
        self.uses[name] = 0
        return sd

    # ---------- catálogo (progressive disclosure, §4.2) ----------
    def catalog(self) -> str:
        """PT: só nome+descrição — SKILL.md completo é lido só após a escolha.
        EN: names+descriptions only — full SKILL.md read after selection."""
        return "\n".join(f"- {s.name} — {s.description()}"
                         for s in self.skills.values())

    def full_doc(self, name: str) -> str:
        return (self.skills[name].path / "SKILL.md").read_text()

    # ---------- use + memory ----------
    def use(self, llm: LLM, name: str, task: str, arg: str) -> tuple[bool, str]:
        sd = self.skills[name]
        ok, out = sd.run(arg)
        self.uses[name] = self.uses.get(name, 0) + 1
        if not ok:
            self.fails[name] = self.fails.get(name, 0) + 1
        if self.lifecycle:
            note = llm.complete(task_prompt(
                "muse.memory",
                f"Task: {task}\nOutcome: {'success' if ok else 'failure'}: "
                f"{out[:80]}"))
            with sd.memory_file.open("a") as f:
                f.write(note + "\n")
        return ok, out

    # ---------- evaluate + refine ----------
    def refine(self, llm: LLM, name: str, task: str, error: str) -> bool:
        """PT: reescreve o script a partir do error trace; retesta pelo gate.
        EN: rewrite the script from the error trace; re-gate by test."""
        if not self.lifecycle or name not in self.skills:
            return False
        sd = self.skills[name]
        out = llm.complete(task_prompt(
            "muse.refine",
            f"Task: {task}\nError trace:\n{error}\n"
            f"Current script:\n{sd.script.read_text()}"))
        m = re.search(r"<script>\n(.*?)\n</script>", out, flags=re.S)
        if not m:
            return False
        sd.script.write_text(m.group(1))
        return sd.gate_test()

    # ---------- merge + prune ----------
    def merge_near_duplicates(self) -> int:
        """PT: funde skills com descrições quase iguais (overlap alto de
        tokens). EN: merge skills with near-identical descriptions."""
        names = list(self.skills)
        toks = {n: token_set(self.skills[n].description())
                for n in names}
        drop: set[str] = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a in drop or b in drop:
                    continue
                uni = toks[a] | toks[b]
                if uni and len(toks[a] & toks[b]) / len(uni) > 0.7:
                    drop.add(b)
                    # PT: transfere uso/memória para o sobrevivente.
                    # EN: transfer usage/memory to the survivor.
                    self.uses[a] = self.uses.get(a, 0) + self.uses.get(b, 0)
        for b in drop:
            del self.skills[b]
            self.merged += 1
        return len(drop)

    def prune_low_utility(self, min_uses: int = 1) -> int:
        """PT: poda skills nunca usadas ou que só falharam.
        EN: prune never-used or always-failing skills."""
        if not self.lifecycle:
            return 0
        dead = [n for n in self.skills
                if self.uses.get(n, 0) < min_uses
                or self.fails.get(n, 0) > self.uses.get(n, 0)]
        for n in dead:
            del self.skills[n]
            self.pruned += 1
        return len(dead)


# ---------------------------------------------------------------------------
# Contexto como DAG (§4.4)
# ---------------------------------------------------------------------------

@dataclass
class ContextDAG:
    """PT: contexto como DAG: nós ordenados com KEEP_FIRST/KEEP_LAST fixos;
    Level-1 comprime nós grandes; Level-2 funde a cadeia do meio — tudo até
    caber no orçamento de tokens. EN: context as a DAG: ordered nodes with
    pinned KEEP_FIRST/KEEP_LAST; Level-1 compresses big nodes; Level-2 merges
    the middle chain — all until under the token budget."""
    nodes: list[tuple[str, str]] = field(default_factory=list)  # (label, text)

    def add(self, label: str, text: str) -> None:
        self.nodes.append((label, text))

    def total_tokens(self) -> int:
        return sum(_tok_len(t) for _, t in self.nodes)

    def compress(self, llm: LLM, budget: int,
                 big: int = 30) -> ContextDAG:
        """PT: Level-1 → Level-2 até `budget`. EN: Level-1 then Level-2."""
        while self.total_tokens() > budget:
            # PT: Level-1: comprime o maior nó não-pinned.
            # EN: Level-1: compress the biggest unpinned node.
            inner = [(i, t) for i, (lab, t) in enumerate(self.nodes)
                     if lab not in (KEEP_FIRST, KEEP_LAST)]
            big_i = max(inner, key=lambda x: _tok_len(x[1]), default=None)
            if big_i is not None and _tok_len(big_i[1]) > big:
                i = big_i[0]
                label, text = self.nodes[i]
                self.nodes[i] = (label, llm.complete(task_prompt(
                    "muse.compress", f"Node:\n{text}")))
            else:
                # PT: Level-2: funde a cadeia do meio num único nó-resumo.
                # EN: Level-2: merge the middle chain into one summary node.
                if len(inner) <= 1:
                    break
                chain = "\n".join(t for _, t in inner)
                merged = llm.complete(task_prompt("muse.merge_chain",
                                                  f"Chain:\n{chain}"))
                first = self.nodes[0]
                last = self.nodes[-1]
                self.nodes = [first, ("merged", merged), last]
        return self
