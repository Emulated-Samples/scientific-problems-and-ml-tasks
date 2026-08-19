"""base-stream forward-cone discriminators.

Query-cost asymmetry: out-query cost must track the FORWARD cone (late
position / high layer => small cone => cheap), measured relative to what
the GIVEN backward engine demonstrates on the backward side. Same probe
isolation and correctness gating as the contract anchors. Targets set from
the v2 reference implementation's bands (dossier calibration table).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _lib.grade_env import pristine_src, record_metric as _record_metric

_PRISTINE = pristine_src()

pytestmark = pytest.mark.skipif(
    not _PRISTINE,
    reason="base-stream discriminator (LAT_BENCH_PRISTINE_SRC not set)",
)

_MIN_FWD_TEMPORAL = 1.6
_MIN_FWD_LAYER = 3.0


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat src not found on PYTHONPATH")


def _probe(src, mode, fixture, out):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LAT_BENCH_METRICS_FILE", "LAT_BENCH_PRISTINE_SRC", "PYTHONPATH")
    }
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_probe.py"),
         "--src", src, "--mode", mode, "--fixture", fixture, "--out", str(out)],
        capture_output=True, text=True, timeout=15 * 60, env=env,
    )
    assert proc.returncode == 0, f"probe failed ({mode}/{fixture}):\n{proc.stdout}\n{proc.stderr}"
    return {k: float(v) for k, v in (p.split("=") for p in proc.stdout.split())}


def _pristine_graph(fixture, tmp_path):
    p = tmp_path / f"{fixture}.graph.pt"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LAT_BENCH_METRICS_FILE", "LAT_BENCH_PRISTINE_SRC", "PYTHONPATH")
    }
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_probe.py"),
         "--src", _PRISTINE, "--mode", "graph", "--fixture", fixture, "--out", str(p)],
        capture_output=True, text=True, timeout=15 * 60, env=env,
    )
    assert proc.returncode == 0, f"pristine graph probe failed:\n{proc.stderr}"
    return torch.load(p, weights_only=False)


def _asym(fixture, metric, tmp_path):
    ref = _pristine_graph(fixture, tmp_path)
    A = ref["adjacency"]
    scale = max(A.abs().max().item(), 1.0)
    out = tmp_path / f"{fixture}.asym.pt"
    r = _probe(_agent_src(), "fwd-asym", fixture, out)
    s = torch.load(out, weights_only=False)
    for tag in ("a", "b", "a_mid", "b_mid"):
        assert torch.allclose(s[f"sample_{tag}"], A[:, s[f"col_{tag}"]], rtol=1e-4,
                              atol=1e-5 * scale), (
            f"{fixture}: forward out-query ({tag}) returned the wrong column"
        )
    ratio = r["t_a"] / max(r["t_b"], 1e-9)
    _record_metric(metric, ratio)
    return ratio


def test_v2_query_cost_respects_temporal_causality(tmp_path):
    """Late-position sources have tiny forward cones; querying them must be
    much cheaper than early-position sources on the same graph."""
    ratio = _asym("temporal", "v2_fwd_temporal_asymmetry", tmp_path)
    assert ratio >= _MIN_FWD_TEMPORAL, (
        f"query cost ignores temporal causality (ratio {ratio:.2f} < {_MIN_FWD_TEMPORAL})"
    )


def test_v2_query_cost_respects_layer_causality(tmp_path):
    """High-layer sources have tiny forward cones; querying them must be much
    cheaper than low-layer sources."""
    ratio = _asym("layer", "v2_fwd_layer_asymmetry", tmp_path)
    assert ratio >= _MIN_FWD_LAYER, (
        f"query cost ignores layer causality (ratio {ratio:.2f} < {_MIN_FWD_LAYER})"
    )
