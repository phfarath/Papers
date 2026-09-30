"""p24 — memória situacional: contexto resolve cues ambíguos."""

import numpy as np

from papers.p24_situation_memory.method import (
    SituationMemory,
    context_persistence,
    cosine,
    gated_vs_flat,
)


def test_determinism_same_seed():
    a = SituationMemory(seed=0)
    b = SituationMemory(seed=0)
    assert (a.cues == b.cues).all() and (a.Y == b.Y).all()


def test_gated_beats_flat():
    g, f = gated_vs_flat(SituationMemory(seed=0))
    assert g > 0.85
    assert g > f + 0.2


def test_same_cue_different_contexts():
    mem = SituationMemory(seed=1)
    k = 0
    for c in range(2):
        yh = mem.recall_gated(mem.cues[k], mem.ctxs[c])
        assert cosine(yh, mem.Y[k, c]) > cosine(yh, mem.Y[k, 1 - c])


def test_glial_latch_sustains_context():
    mem = SituationMemory(seed=2, tau_ctx=4.0)
    assert context_persistence(mem, gap=1.0) > 0.8


def test_no_latch_no_context():
    # latch zerado: sem contexto nenhuma unidade dispara -> recall degenerado
    mem = SituationMemory(seed=3)
    cs = []
    for k in range(mem.cues.shape[0]):
        for c in range(2):
            yh = mem.recall_latched(mem.cues[k])
            cs.append(cosine(yh, mem.Y[k, c]))
    assert np.mean(cs) < 0.5
