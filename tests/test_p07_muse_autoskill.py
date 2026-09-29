"""PT: testes do p07_muse_autoskill — skill dirs, lifecycle, DAG compressão.
EN: p07_muse_autoskill tests — skill dirs, lifecycle, DAG compression."""

import tempfile
from pathlib import Path

from papers.common.llm import MockLLM
from papers.p07_muse_autoskill.method import (
    KEEP_FIRST,
    KEEP_LAST,
    ContextDAG,
    SkillStore,
)


def test_skill_dir_layout_and_gate() -> None:
    llm = MockLLM(0)
    with tempfile.TemporaryDirectory() as td:
        store = SkillStore(Path(td))
        sd = store.create(llm, "reverse the string")
        assert sd is not None
        assert (sd.path / "SKILL.md").exists()
        assert sd.script.exists() and sd.test_file.exists()
        assert sd.gate_test()
        ok, out = sd.run("hello")
        assert ok and out == "olleh"


def test_catalog_progressive_disclosure() -> None:
    llm = MockLLM(0)
    with tempfile.TemporaryDirectory() as td:
        store = SkillStore(Path(td))
        store.create(llm, "sum the comma list")
        cat = store.catalog()
        assert "sum" in cat and "<script>" not in cat  # nome+desc only
        assert "description" in store.full_doc("sum")


def test_lifecycle_prune_and_memory() -> None:
    llm = MockLLM(0)
    with tempfile.TemporaryDirectory() as td:
        store = SkillStore(Path(td))
        store.create(llm, "reverse the string")
        store.create(llm, "unused placeholder skill")
        store.use(llm, "reverse", "reverse the string", "ab")
        assert store.skills["reverse"].memory_file.read_text().strip() != ""
        assert store.prune_low_utility() >= 1
        assert "helper" not in store.skills


def test_dag_compression_pinned() -> None:
    llm = MockLLM(0)
    dag = ContextDAG()
    dag.add(KEEP_FIRST, "head")
    for i in range(5):
        dag.add(f"n{i}", "tok " * 30)
    dag.add(KEEP_LAST, "tail")
    dag.compress(llm, budget=40)
    labels = [lab for lab, _ in dag.nodes]
    assert labels[0] == KEEP_FIRST and labels[-1] == KEEP_LAST
    assert dag.total_tokens() <= 40 or len(dag.nodes) <= 3
