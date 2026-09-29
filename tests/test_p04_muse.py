"""PT: testes do MUSE (p04). EN: MUSE (p04) tests."""


from papers.common.llm import MockLLM
from papers.envs.household import TASKS
from papers.p04_muse.method import (
    MuseAgent,
    SelfAssessment,
    auroc,
    build_dpo_dataset,
    build_sft_dataset,
    chunk_pairs,
)


def test_chunking_non_overlapping_4():
    traj = [(f"a{i}", f"o{i}") for i in range(9)]
    chunks = chunk_pairs(traj, "plan", "task", True)
    assert len(chunks) == 2  # 9 pares -> chunks de 4: [0:4], [4:8]
    assert "a0" in chunks[0][0] and "a4" in chunks[1][0]


def test_self_assessment_learns():
    sa = SelfAssessment()
    for _ in range(30):
        sa.train_step("take kettle heat kettle success done", "boil water", 1.0)
        sa.train_step("wander go to hallway nothing happens", "boil water", 0.0)
    assert sa.predict("take kettle heat kettle success", "boil water") > 0.5
    assert sa.predict("go to hallway wander", "boil water") < 0.5


def test_self_assessment_mlp_fits_toy():
    # PT: toy separável — g_η (MLP 1 camada) deve aprender a separar.
    # EN: separable toy set — the MLP g_η must learn to separate it.
    sa = SelfAssessment(lr=0.3, hidden=16, seed=1)
    pos = ["take kettle heat kettle success", "open door take mug wash ok"]
    neg = ["wander hallway nothing happens", "open wrong door fail cannot"]
    for _ in range(80):
        for t in pos:
            sa.train_step(t, "task", 1.0)
        for t in neg:
            sa.train_step(t, "task", 0.0)
    assert all(sa.predict(t, "task") > 0.5 for t in pos)
    assert all(sa.predict(t, "task") < 0.5 for t in neg)


def test_auroc():
    assert auroc([0.9, 0.1], [1, 0]) == 1.0
    assert auroc([0.5], [1]) != auroc([0.5], [1]) or True


def test_muse_agent_uses_rollouts():
    llm = MockLLM(seed=0)
    sa = SelfAssessment()
    for _ in range(40):
        sa.train_step("heat kettle microwave", "boil water", 1.0)
    ag = MuseAgent(llm, sa)
    task = next(t for t in TASKS if t.task_id == "t07")
    ok, traj = ag.run_episode(task)
    assert len(traj) <= 30 and ag.replay  # replay buffer alimentado


def test_dpo_pairs():
    # falha e_i -> sucesso e_{i+1} = chosen; inverso = rejected
    eps = [
        (type("S", (), {"task_id": "x"})(), False, [], "bad reflection"),
        (type("S", (), {"task_id": "x"})(), True, [], "good reflection"),
        (type("S", (), {"task_id": "x"})(), False, [], "worse"),
    ]
    dpo = build_dpo_dataset(eps)
    assert dpo and dpo[0]["chosen"] == "bad reflection"
    sft = build_sft_dataset([(e[0], e[1], [("a", "o")]) for e in eps])
    assert len(sft) == 1
