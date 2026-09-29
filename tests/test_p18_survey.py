"""PT: teste do p18 — toda pasta pNN (exceto o próprio survey) está
classificada com rótulos válidos. EN: p18 test — every pNN folder (except
the survey itself) is classified with valid labels."""
from papers.common.catalog import CATALOG
from papers.p18_survey.taxonomy import (
    CLASSIFICATIONS,
    DYNAMICS,
    FORMS,
    FUNCTIONS,
)


def test_every_paper_classified() -> None:
    folders = {p.folder for p in CATALOG} - {"p18_survey"}
    assert {c.folder for c in CLASSIFICATIONS} == folders


def test_valid_labels() -> None:
    for c in CLASSIFICATIONS:
        assert c.form in FORMS
        fun = {x.strip() for x in c.functions.split("+")}
        dyn = {x.strip() for x in c.dynamics.split("+")}
        assert fun <= FUNCTIONS and fun
        assert dyn <= DYNAMICS and dyn
        assert c.justification_pt and c.justification_en
