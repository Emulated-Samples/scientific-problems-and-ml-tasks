"""base-stream contract anchors: streaming + node queries on the fast base.

Same probe-isolation architecture as base-stream (the parent never imports
agent ``lat``; all agent code runs in scrubbed ``_probe.py`` subprocesses;
every timing fixture carries a correctness gate). v2 differences:

- the workspace SHIPS the optimized causal-cone backward, so no backward
  rows are graded here — a speed-retention multiplier guards de-optimizing
  the given engine (test_v2_retention.py);
- the never-materialize constraint is STRUCTURAL: the ``huge`` fixture's
  adjacency (~12 GB) cannot exist under the probe's hard 8 GB address-space
  cap, and the grader verifies sampled rows via single-target pristine
  backward sweeps — neither side ever builds the matrix;
- anti-drip is carried by first-frame online-ness (a cadence row was
  falsified in calibration — see the note mid-file).

Test names are v2-prefixed — never reuse base-stream's bare names (the
junit skip-scan matches bare names across problems).
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
    reason="base-stream anchors (LAT_BENCH_PRISTINE_SRC not set for this problem)",
)

# Targets calibrated on the v2 reference solution (fast backward + suffix-cone
# forward): see the dossier calibration table. Thresholds are the fail floor;
# continuous credit runs to the target.
_MIN_FRAME_SPEEDUP = 1.5
_MIN_OUT_QUERY_SPEEDUP = 20.0
_HUGE_QUERY_BUDGET_S = 30.0


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat src not found on PYTHONPATH")


def _probe(src: str, mode: str, fixture: str, out: Path | None = None) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LAT_BENCH_METRICS_FILE", "LAT_BENCH_PRISTINE_SRC", "PYTHONPATH")
    }
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [sys.executable, str(Path(__file__).parent / "_probe.py"),
           "--src", src, "--mode", mode, "--fixture", fixture]
    if out is not None:
        cmd += ["--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15 * 60, env=env)
    assert proc.returncode == 0, (
        f"probe failed ({mode}/{fixture}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    return proc.stdout


def _kv(stdout: str) -> dict:
    return {k: float(v) for k, v in (p.split("=") for p in stdout.split())}


@pytest.fixture(scope="module")
def pristine_graphs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pristine")
    out = {}
    for fx in ("eq1", "eq2", "big"):
        p = root / f"{fx}.pt"
        _probe(_PRISTINE, "graph", fx, p)
        out[fx] = torch.load(p, weights_only=False)
    return out


def _close(a, b, scale):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5 * max(scale, 1.0))


def test_v2_stream_reconstructs_reference_graph(pristine_graphs, tmp_path):
    """Concatenated frames reproduce the pristine batch graph exactly on a
    plain causal fixture AND a jacobian-mode ragged one."""
    for fx in ("eq1", "eq2"):
        ref = pristine_graphs[fx]
        A = ref["adjacency"]
        F = ref["active_features"].shape[0]
        n_pos, n_layers = ref["n_pos"], ref["n_layers"]
        log_off = F + n_layers * n_pos + n_pos
        out = tmp_path / f"{fx}.drain.pt"
        _probe(_agent_src(), "drain", fx, out)
        d = torch.load(out, weights_only=False)
        assert d["ascending"], f"{fx}: frames must arrive by ascending source layer"
        assert d["no_dup"], f"{fx}: a source was emitted in more than one frame"
        assert d["n_sources_seen"] == d["expected_sources"] == log_off
        scale = max(A.abs().max().item(), 1.0)
        assert _close(d["features_block"], A[:F, :log_off], scale), f"{fx}: feature edges diverge"
        assert _close(d["logit_block"], A[log_off:, :log_off], scale), f"{fx}: logit edges diverge"


def test_v2_first_frame_is_online(pristine_graphs, tmp_path):
    """First frame in a small fraction of the drain — and the drain correct."""
    ref = pristine_graphs["big"]
    A, F = ref["adjacency"], ref["active_features"].shape[0]
    n_pos, n_layers = ref["n_pos"], ref["n_layers"]
    log_off = F + n_layers * n_pos + n_pos
    out = tmp_path / "big.drain.pt"
    stats = _kv(_probe(_agent_src(), "drain", "big", out))
    d = torch.load(out, weights_only=False)
    scale = max(A.abs().max().item(), 1.0)
    assert d["n_sources_seen"] == log_off and _close(d["features_block"], A[:F, :log_off], scale)
    ratio = stats["t_total"] / max(stats["t_first"], 1e-9)
    _record_metric("v2_frame_speedup", ratio)
    assert ratio >= _MIN_FRAME_SPEEDUP, (
        f"first frame took {stats['t_first']:.2f}s of {stats['t_total']:.2f}s "
        f"(ratio {ratio:.1f} < {_MIN_FRAME_SPEEDUP}) — the stream is not online"
    )


# NOTE: a cadence row (t_total/t50 of sources) was FALSIFIED in calibration
# 2026-07-17 and removed before ever grading a rollout: honest streams here
# are inherently FRONT-loaded (the token frame — the deepest forward cones —
# arrives first, shallow top layers last), so the reference measures ~1.18
# vs ~1.0 for compute-all-then-drip: no usable separation, and the row
# penalized the canonical solution for stream physics. The anti-drip signal
# is carried by first-frame online-ness (drip pays everything before frame
# one), which absorbed the removed row's weight.


def test_v2_node_edges_match_reference(pristine_graphs, tmp_path):
    ref = pristine_graphs["eq1"]
    A = ref["adjacency"]
    scale = max(A.abs().max().item(), 1.0)
    out = tmp_path / "nodes.pt"
    _probe(_agent_src(), "node-edges", "eq1", out)
    r = torch.load(out, weights_only=False)
    for name in ("out_feature", "out_error", "out_token"):
        col = r[name + "_col"]
        assert _close(r[name], A[:, col], scale), f"{name}: outgoing edges diverge"
    row = r["in_feature_row"]
    assert _close(r["in_feature"], A[row, :], scale), "incoming edges diverge"


def test_v2_out_query_is_incremental(pristine_graphs, tmp_path):
    """One out-query costs a small fraction of a full build; fresh IR per rep."""
    ref = pristine_graphs["big"]
    A = ref["adjacency"]
    scale = max(A.abs().max().item(), 1.0)
    pristine_build = _kv(_probe(_PRISTINE, "build-time", "big"))["t_build"]
    out = tmp_path / "q.pt"
    t_query = _kv(_probe(_agent_src(), "out-query", "big", out))["t_query"]
    q = torch.load(out, weights_only=False)
    assert _close(q["result"], A[:, q["col"]], scale), "out-query returned the wrong column"
    ratio = pristine_build / max(t_query, 1e-9)
    _record_metric("v2_out_query_speedup", ratio)
    assert ratio >= _MIN_OUT_QUERY_SPEEDUP, (
        f"one out-query costs {t_query:.3f}s vs {pristine_build:.2f}s build "
        f"(ratio {ratio:.1f} < {_MIN_OUT_QUERY_SPEEDUP})"
    )


def test_v2_never_materializes_the_matrix(tmp_path):
    """STRUCTURAL: on the huge fixture (~12 GB adjacency) under the probe's
    hard 8 GB cap, a partial drain must stay lazy, respect the frame
    contract, and reproduce grader-sampled target rows — verified against
    single-target pristine backward sweeps (the grader never builds the
    matrix either)."""
    agent_out = tmp_path / "huge.drain.pt"
    stats = _kv(_probe(_agent_src(), "huge-drain", "huge", agent_out))
    d = torch.load(agent_out, weights_only=False)
    assert d["ascending"] and d["no_dup"], "frame contract violated on the huge fixture"
    assert d["n_sources_seen"] >= 4096, "partial drain covered too few sources"

    ref_out = tmp_path / "huge.rows.pt"
    _probe(_PRISTINE, "row-ref", "huge", ref_out)
    ref = torch.load(ref_out, weights_only=False)
    assert d["targets"] == ref["targets"]
    for t in ref["targets"]:
        row = ref["rows"][t]
        seen = d["rows"][t]
        assert seen, f"no sampled edges accumulated for target {t}"
        vals = torch.tensor([row[s] for s in sorted(seen)])
        got = torch.tensor([seen[s] for s in sorted(seen)])
        scale = max(float(row.abs().max()), 1.0)
        assert torch.allclose(got, vals, rtol=1e-4, atol=1e-5 * scale), (
            f"sampled row {t} diverges from the pristine backward row"
        )


def test_v2_out_query_survives_the_huge_fixture(tmp_path):
    """One out-query on the huge fixture must complete under the 8 GB cap,
    within a hard time budget, and be pointwise-correct at the grader's
    sampled targets (pristine backward rows give edge (s -> t) as row_t[s])."""
    agent_out = tmp_path / "huge.oq.pt"
    stats = _kv(_probe(_agent_src(), "huge-oq", "huge", agent_out))
    assert stats["t_query"] <= _HUGE_QUERY_BUDGET_S, (
        f"out-query on the huge fixture took {stats['t_query']:.1f}s "
        f"(> {_HUGE_QUERY_BUDGET_S}s) — a query must not pay for the graph"
    )
    q = torch.load(agent_out, weights_only=False)
    ref_out = tmp_path / "huge.rows.pt"
    _probe(_PRISTINE, "row-ref", "huge", ref_out)
    ref = torch.load(ref_out, weights_only=False)
    s = q["node_col"]
    for t in ref["targets"]:
        row = ref["rows"][t]
        scale = max(float(row.abs().max()), 1.0)
        assert torch.allclose(q["col"][t], row[s], rtol=1e-4, atol=1e-5 * scale), (
            f"out-query column diverges from pristine row {t} at source {s}"
        )
