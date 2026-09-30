"""PT: regenera TODOS os RESULTS.md (p01–p31) com a seed fixa e mede o
tempo total. Uso: `python scripts/run_all.py`.

EN: regenerates ALL RESULTS.md (p01–p31) at the fixed seed and times the
total. Usage: `python scripts/run_all.py`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = [f"p{i:02d}_{s}" for i, s in enumerate([
    "clin", "reflexion", "memrl", "muse", "cer", "voyager",
    "muse_autoskill", "coala", "generative_agents", "memgpt", "a_mem",
    "zep", "mem0", "memory_r1", "longmemeval", "memoryagentbench",
    "longmemeval_v2", "survey", "astrocyte_assoc", "astrocyte_context",
    "astrocyte_transformer", "neuron_glia_nets", "astrocyte_wm",
    "situation_memory", "agmp_continual", "dual_memory_nav",
    "emergent_attention", "astromorphic_repair", "lsm_astrocytes",
    "hybrid_automaton", "futility_passivity"], start=1)]


def main() -> None:
    t0 = time.perf_counter()
    failed: list[str] = []
    for p in PAPERS:
        t = time.perf_counter()
        r = subprocess.run(
            [sys.executable, "-m", f"papers.{p}.run", "--write-results"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        dt = time.perf_counter() - t
        status = "ok" if r.returncode == 0 else "FAIL"
        if r.returncode != 0:
            failed.append(p)
            print(r.stderr[-2000:])
        print(f"{p}: {status} ({dt:.1f}s)")
    total = time.perf_counter() - t0
    print(f"\nTOTAL: {total:.1f}s | failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
