"""base-stream backward speed retention (metricMultiplier feed).

The workspace SHIPS the optimized causal-cone backward; the agent must not
de-optimize it to make streaming ratios look better. This module records
``v2_bwd_retention`` = t_pristine_build / t_agent_build (geomean over two
fixtures, best-of-3 per side, correctness via checksum comparison) — the
engine applies it as a fail-closed multiplier (floor/ceiling in the
grading module), never as a scored row: a scored retention row would hand
the untouched state free credit.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _lib.grade_env import pristine_src, record_metric as _record_metric

_PRISTINE = pristine_src()

pytestmark = pytest.mark.skipif(
    not _PRISTINE,
    reason="base-stream retention (LAT_BENCH_PRISTINE_SRC not set)",
)


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat src not found on PYTHONPATH")


def _build(src, fixture):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LAT_BENCH_METRICS_FILE", "LAT_BENCH_PRISTINE_SRC", "PYTHONPATH")
    }
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_probe.py"),
         "--src", src, "--mode", "build-time", "--fixture", fixture],
        capture_output=True, text=True, timeout=15 * 60, env=env,
    )
    assert proc.returncode == 0, f"build-time probe failed ({fixture}):\n{proc.stderr}"
    return {k: float(v) for k, v in (p.split("=") for p in proc.stdout.split())}


def test_v2_backward_speed_retention():
    """Records the retention metric; asserts only correctness (checksum
    parity with pristine) so an honest solution can never fail this row —
    the retention consequence is the multiplier's job.

    Calibration note (post-mortem): sequential-block A/B
    measured 0.915 and 1.147 on BYTE-IDENTICAL engines — cross-subprocess
    drift is ~±15%. The probes therefore run INTERLEAVED (p/a/p/a/p/a,
    best-of-3 per side) to cancel drift, and the multiplier band in the
    grading module sits outside the residual noise. The fixture is ``big``
    alone: large enough that a genuine naive-revert measures ~0.5 (scale
    ~0.1), small enough (12s/build) that six interleaved builds fit the
    suite's 20-minute budget — big2/temporal builds (41-53s) blew it."""
    ratios = []
    for fx in ("big",):
        p_best, a_best, p_chk, a_chk = float("inf"), float("inf"), None, None
        for _ in range(3):
            p = _build(_PRISTINE, fx)
            a = _build(_agent_src(), fx)
            p_best, a_best = min(p_best, p["t_build"]), min(a_best, a["t_build"])
            p_chk, a_chk = p["checksum"], a["checksum"]
        assert abs(p_chk - a_chk) <= 1e-3 * max(abs(p_chk), 1.0), (
            f"{fx}: agent build diverges from pristine (checksum mismatch)"
        )
        ratios.append(p_best / max(a_best, 1e-9))
    _record_metric("v2_bwd_retention", math.exp(sum(map(math.log, ratios)) / len(ratios)))
