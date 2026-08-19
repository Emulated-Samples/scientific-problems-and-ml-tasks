"""Graded speedup test for base-* problems: agent tree vs GRADER-OWNED pristine base.

The reward-carrying measurement for `base-speedup`: how much faster is the
agent's attribution than the pristine naive baseline on identical mixed
workloads, same box, same interpreter?

Design (the reward-relevant properties):

- The baseline is a pristine copy of the problem's own workspace src, staged
  by the environment OUTSIDE the agent-visible workspace
  (LAT_BENCH_PRISTINE_SRC). The agent cannot slow it down or edit it, which
  closes the slow-the-baseline hack class that in-repo A/B comparisons have.
- Both sides run in fresh subprocesses via _lib/speedup_runner.py: single-thread
  CPU process_time, one warmup call (cold-start amortization) + best-of-3
  measured reps per fixture. The ratio is same-box A/B so machine speed
  cancels.
- MIXED fixtures (features at every position and layer, two shapes covering
  the position-heavy and layer-heavy regimes): ANY legitimate speedup of the
  sweep earns credit here. Mechanism classification is the asymmetry
  discriminators' job (test_base_causal_perf.py), not this file's.
- Fail-closed output check: before any ratio is trusted, the two sides'
  graphs must agree (identical selected-feature triples, adjacency allclose).
  A "speedup" that changes the answer scores zero — including one that only
  misbehaves on these fixtures.
- The measured ratio is written to the metrics sidecar BEFORE the threshold
  assert, so failing runs still report their measurement and the grader's
  continuous scoring can convert it to partial credit.

Gated on LAT_BENCH_PRISTINE_SRC: the environment sets it only for base-*
problem ids, so on every other problem this file skips (neutral p2p).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _lib.grade_env import gold_src, pristine_src, record_metric as _record_metric

# Calibrated floor: pristine-vs-pristine is ~1.0 by construction; the
# threshold only needs to sit above noise. Continuous credit above it is the
# grader's job (CONTINUOUS_PERF target).
_MIN_SPEEDUP = 1.3

_PRISTINE_SRC = pristine_src()

pytestmark = pytest.mark.skipif(
    not _PRISTINE_SRC,
    reason="base-* speedup harness (LAT_BENCH_PRISTINE_SRC not set for this problem)",
)


def _agent_src() -> str:
    """The agent's lat src — the same path the grader put on PYTHONPATH."""
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat src not found on PYTHONPATH")


def _run_side(src: str, out: Path) -> dict:
    # os.environ no longer carries LAT_BENCH_* (stashed by _lib.grade_env
    # before anything imported), so the agent-code subprocess is born clean.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # the runner adds --src itself
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    runner = Path(__file__).parents[2] / "_lib" / "speedup_runner.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--src", src, "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=15 * 60,
        env=env,
    )
    assert proc.returncode == 0, (
        f"speedup runner failed for src={src}:\n{proc.stdout}\n{proc.stderr}"
    )
    return torch.load(out, weights_only=False)


def test_attribution_sweep_speedup(tmp_path):
    """Agent tree must be substantially faster than the pristine baseline on
    identical mixed workloads — with identical outputs."""
    pristine_src = _PRISTINE_SRC
    assert pristine_src and Path(pristine_src, "lat", "__init__.py").exists(), (
        f"pristine baseline missing at {pristine_src!r} — cannot grade speedup "
        "(environment setup should have staged it)"
    )

    agent = _run_side(_agent_src(), tmp_path / "agent.pt")
    pristine = _run_side(pristine_src, tmp_path / "pristine.pt")

    # Output equality first — a wrong-answer speedup must never score.
    ratios = {}
    for name in pristine:
        a, p = agent[name], pristine[name]
        assert a["selected_triples"] == p["selected_triples"], (
            f"fixture {name}: agent selected different features than pristine"
        )
        assert a["logit_nodes"] == p["logit_nodes"], (
            f"fixture {name}: agent logit nodes differ from pristine"
        )
        scale = p["adjacency"].abs().max().item()
        assert torch.allclose(
            a["adjacency"], p["adjacency"], rtol=1e-4, atol=1e-5 * max(scale, 1.0)
        ), (
            f"fixture {name}: agent adjacency diverges from pristine "
            f"(max diff {(a['adjacency'] - p['adjacency']).abs().max().item():.3e})"
        )
        ratios[name] = p["best_s"] / max(a["best_s"], 1e-9)
        _record_metric(f"speedup_{name}", ratios[name])

    # One combined number: geometric mean over the fixtures, so credit needs
    # improvement across the workload mix rather than a single lucky shape.
    combined = 1.0
    for r in ratios.values():
        combined *= r
    combined = combined ** (1.0 / len(ratios))
    _record_metric("attribution_speedup", combined)

    # Host-relative calibration: time the GRADER-OWNED gold reference on this
    # same box against the same pristine baseline and record its combined
    # speedup. The grader normalises the agent's ratio against gold's ratio
    # HERE instead of an x86-calibrated constant, so gold scores ~1.0 on any
    # architecture. best-of-N inside the runner (below) handles per-side noise;
    # taking the geomean over the two fixtures further damps it. Skipped
    # silently when gold was not staged (target stays the constant).
    gold = gold_src()
    if gold and Path(gold, "lat", "__init__.py").exists():
        # best-of-2 gold measurements → keep the FASTER gold (higher gold
        # ratio, i.e. the more demanding target) so noise cannot inflate an
        # agent's score by handing it a slow gold to beat.
        gold_combined = float("inf")
        for i in range(2):
            gold_side = _run_side(gold, tmp_path / f"gold{i}.pt")
            gc = 1.0
            for name in pristine:
                gc *= pristine[name]["best_s"] / max(gold_side[name]["best_s"], 1e-9)
            gc = gc ** (1.0 / len(pristine))
            gold_combined = min(gold_combined, gc)
        _record_metric("attribution_speedup_gold", gold_combined)

    assert combined >= _MIN_SPEEDUP, (
        f"attribution is not meaningfully faster than the pristine baseline: "
        f"combined {combined:.2f}x (per-fixture: "
        + ", ".join(f"{k}={v:.2f}x" for k, v in ratios.items())
        + f") < {_MIN_SPEEDUP}"
    )
