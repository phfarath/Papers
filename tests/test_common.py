"""PT: testes de papers/common. EN: tests for papers/common."""

import random

import numpy as np

from papers.common.embeddings import HashingEmbedder, cosine_matrix
from papers.common.llm import MockLLM, count_tokens, mock_handler, task_prompt
from papers.common.memory_api import (
    BM25Memory,
    EmbeddingRAGMemory,
    FullContextMemory,
    MemoryItem,
)
from papers.common.reader import answer_with_evidence, judge_answer
from papers.common.retrieval import BM25Index, VectorIndex, mmr, reciprocal_rank_fusion
from papers.common.utils import parse_json_block, tokens, zscore


def test_mock_llm_routes_and_determinism():
    @mock_handler("t.echo")
    def _h(prompt: str, rng: random.Random) -> str:
        return "pong" + str(rng.randint(0, 9))

    a = MockLLM(seed=1).complete(task_prompt("t.echo", "x"))
    b = MockLLM(seed=1).complete(task_prompt("t.echo", "x"))
    assert a == b and a.startswith("pong")


def test_mock_llm_missing_handler_errors():
    llm = MockLLM(seed=0)
    try:
        llm.complete(task_prompt("no.such.task", "x"))
        raise AssertionError("should raise")
    except KeyError as e:
        assert "no.such.task" in str(e)


def test_parse_json_block():
    assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_block('[1, 2]') == [1, 2]


def test_hashing_embedder_deterministic_and_cosine():
    e = HashingEmbedder()
    v1 = e.embed(["the cat sits"])
    v2 = HashingEmbedder().embed(["the cat sits"])
    assert np.allclose(v1, v2)
    assert np.linalg.norm(v1[0]) > 0
    sim = cosine_matrix(e.embed(["cat"]), e.embed(["cat dog"]))
    assert sim[0, 0] > 0.5


def test_bm25_and_vector_index():
    b = BM25Index()
    b.add("d1", "apples are red")
    b.add("d2", "trains are fast")
    assert b.search("apples", 1)[0][0] == "d1"
    v = VectorIndex()
    v.add("d1", "apples are red")
    v.add("d2", "trains are fast")
    assert v.search("apple", 1)[0][0] == "d1"


def test_rrf_and_mmr():
    r = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert r[0] == "b"
    q = np.array([1.0, 0.0])
    c = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    idx = mmr(q, c, lambda_=0.5, k=2)
    assert idx[0] in (0, 1) and len(idx) == 2


def test_memory_baselines():
    for mem in (BM25Memory(), EmbeddingRAGMemory()):
        mem.add(MemoryItem("Bob likes tea."))
        mem.add(MemoryItem("Trains are fast."))
        assert "tea" in mem.retrieve("what does Bob like", k=1)[0]
    fc = FullContextMemory(max_tokens=5)
    fc.add(MemoryItem("one two three four five six seven eight nine ten eleven"))
    fc.add(MemoryItem("four five six"))
    assert fc.retrieve("q") == ["four five six"]


def test_qa_answer_handler_uses_evidence_only():
    llm = MockLLM(seed=0)
    ev = ["Paris is the capital of France.", "The cat is black."]
    ans = answer_with_evidence(llm, "What is the capital of France?", ev)
    assert "France" in ans or "capital" in ans.lower() or "Paris" in ans
    assert answer_with_evidence(llm, "What color is the cat?", ev) == "black"


def test_qa_answer_unknown_and_recency():
    llm = MockLLM(seed=0)
    assert answer_with_evidence(llm, "capital of France?",
                                ["cats are cute"]) == "I don't know"
    ev = ["2020-01-01 The mayor of Springfield is Alice.",
          "2023-05-05 The mayor of Springfield is Bob."]
    assert "Bob" in answer_with_evidence(
        llm, "Who is the mayor of Springfield?", ev)


def test_judge():
    llm = MockLLM(seed=0)
    assert judge_answer(llm, "q", "Bob", "bob")
    assert judge_answer(llm, "q", "Bob", "it is Bob today")
    assert not judge_answer(llm, "q", "Bob", "Alice")


def test_utils():
    assert "cafe" in tokens("Café quente")
    assert zscore([1, 1, 1]) == [0.0, 0.0, 0.0]
    zs = zscore([0.0, 1.0, 2.0])
    assert abs(zs[1]) < 1e-9
    assert count_tokens("hello world") >= 2
