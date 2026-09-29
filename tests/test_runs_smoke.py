"""PT: smoke — todo pacote pNN tem run.py importável com RESULTS.md gerado.
EN: smoke — every pNN package has an importable run.py with a generated
RESULTS.md."""
import importlib
from pathlib import Path

import pytest

from papers.common.catalog import CATALOG

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("p", [p.folder for p in CATALOG])
def test_run_module_importable(p: str) -> None:
    mod = importlib.import_module(f"papers.{p}.run")
    assert hasattr(mod, "main") or hasattr(mod, "experiment") or \
        hasattr(mod, "render")


@pytest.mark.parametrize("p", [p.folder for p in CATALOG])
def test_results_generated(p: str) -> None:
    assert (ROOT / "papers" / p / "RESULTS.md").exists(), p
