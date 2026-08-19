"""Graded ranking-speed test for base-ranker: agent tree vs pristine base.

The reward-carrying measurement for `base-ranker`: capped attribution's
between-batch feature ranking dominates cap-mode runtime on
ranking-dominated workloads; how much faster is the agent's capped
attribution than the pristine baseline — with the SAME selections where the
baseline's method is exact?

Design:

- Same pristine-baseline A/B discipline as test_base_attribution_speedup
  (grader-owned copy, subprocess per side, single-thread process_time,
  1 warmup + best-of-3): the baseline cannot be slowed down or edited.
- TWO capped ranking-dominated fixtures via _lib/speedup_runner.py --mode ranker:
  - "shallow": influence chains well inside the baseline method's
    convergence range. Used as the fail-closed SELECTION-PARITY GATE:
    selections, logit nodes, and adjacency must match the pristine run
    exactly — a "faster ranking" that picks different features scores 0.
    The gate is a precondition inside this test, never a scored test of
    its own (a scored parity test would hand the untouched state free
    credit).
  - "deep": influence chains past that range, where an exact method may
    legitimately select differently — graded on TIMING ONLY.
- The deep-fixture ratio goes to the metrics sidecar BEFORE the threshold
  assert, so the grader's continuous scoring converts partial speedups to
  partial credit.

Gated on LAT_BENCH_PRISTINE_SRC (set only for base-* problem ids); the
grader additionally scopes this file to base-ranker via IGNORE_FILES.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _lib.grade_env import pristine_src, record_metric as _record_metric

# Floor between the no-change band (measured 1.00x) and the genuine
# algorithmic-win band (measured 1.65x on this fixture — on the naive base
# the backward sweep is a large fixed cost the ranking work sits on top of,
# so ratios here are structurally tighter than the selection contract's
# other bands). Continuous credit above the floor is the grader's job.
_MIN_RANKER_SPEEDUP = 1.25

_PRISTINE_SRC = pristine_src()

pytestmark = pytest.mark.skipif(
    not _PRISTINE_SRC,
    reason="base-* ranker harness (LAT_BENCH_PRISTINE_SRC not set for this problem)",
)


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat src not found on PYTHONPATH")


def _run_side(src: str, out: Path) -> dict:
    # os.environ no longer carries LAT_BENCH_* (stashed by _lib.grade_env
    # before anything imported), so the agent-code subprocess is born clean.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    runner = Path(__file__).parents[2] / "_lib" / "speedup_runner.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--src", src, "--out", str(out), "--mode", "ranker"],
        capture_output=True,
        text=True,
        timeout=15 * 60,
        env=env,
    )
    assert proc.returncode == 0, (
        f"ranker runner failed for src={src}:\n{proc.stdout}\n{proc.stderr}"
    )
    return torch.load(out, weights_only=False)


def test_ranker_speedup(tmp_path):
    """Capped attribution must be substantially faster than the pristine
    baseline on a ranking-dominated workload — while selecting identically
    wherever the baseline's method is exact."""
    pristine_src = _PRISTINE_SRC
    assert pristine_src and Path(pristine_src, "lat", "__init__.py").exists(), (
        f"pristine baseline missing at {pristine_src!r} — cannot grade "
        "(environment setup should have staged it)"
    )

    agent = _run_side(_agent_src(), tmp_path / "agent.pt")
    pristine = _run_side(pristine_src, tmp_path / "pristine.pt")

    # Fail-closed parity gate on the shallow fixture (baseline exact there).
    a, p = agent["shallow"], pristine["shallow"]
    assert a["selected_triples"] == p["selected_triples"], (
        "shallow capped fixture: agent selected different features than the "
        "pristine baseline — the ranking contract (same selections) is broken"
    )
    assert a["logit_nodes"] == p["logit_nodes"], (
        "shallow capped fixture: agent logit nodes differ from pristine"
    )
    scale = p["adjacency"].abs().max().item()
    assert torch.allclose(
        a["adjacency"], p["adjacency"], rtol=1e-4, atol=1e-5 * max(scale, 1.0)
    ), (
        "shallow capped fixture: agent adjacency diverges from pristine "
        f"(max diff {(a['adjacency'] - p['adjacency']).abs().max().item():.3e})"
    )

    # Timing on the deep fixture only.
    ratio = pristine["deep"]["best_s"] / max(agent["deep"]["best_s"], 1e-9)
    _record_metric("ranker_speedup", ratio)
    _record_metric("ranker_speedup_shallow", p["best_s"] / max(a["best_s"], 1e-9))

    assert ratio >= _MIN_RANKER_SPEEDUP, (
        f"capped attribution is not meaningfully faster than the pristine "
        f"baseline on the ranking-dominated fixture: {ratio:.2f}x < "
        f"{_MIN_RANKER_SPEEDUP}"
    )
