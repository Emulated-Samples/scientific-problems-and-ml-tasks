"""base-backward anchors: the hand-written adjoints must be correct AND
must be hand-derived (no autograd, no numerical-Jacobian shortcut).

This module NEVER imports the workspace ``lat``. Agent code runs only in a
scrubbed ``_probe.py`` subprocess (no LAT_BENCH_* env vars). The probe calls
the agent's backward primitives under ``torch.no_grad()`` AND under a runtime
sensor that replaces the actual autograd function objects (autograd.grad /
Tensor.backward / torch.func / autograd.functional VJP-family) and the
forward-linear ops with trip-wires — so autograd (however aliased, getattr'd,
imported from a sibling module, or wrapped in ``enable_grad``) and the
"sweep basis vectors through the forward" shortcut both raise, scoring that
primitive 0 while leaving the other primitive's anchors intact. Ground truth
is the VJP of the GRADER-OWNED pristine forward (apply_norm_linear /
apply_attention_linear are the fixed forward operators; their transpose is
the exact adjoint), computed in a separate pristine probe.

Gated on LAT_BENCH_PRISTINE_SRC (set only for pristine-baseline problems);
scoped to base-backward via the suites mechanism.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _lib.grade_env import pristine_src

_PRISTINE = pristine_src()

pytestmark = pytest.mark.skipif(
    not _PRISTINE,
    reason="base-backward anchors (LAT_BENCH_PRISTINE_SRC not set)",
)


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat src not found on PYTHONPATH")


def _probe(src: str, mode: str, out: Path):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LAT_BENCH_METRICS_FILE", "LAT_BENCH_PRISTINE_SRC", "PYTHONPATH")
    }
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_probe.py"),
         "--src", src, "--mode", mode, "--out", str(out)],
        capture_output=True, text=True, timeout=10 * 60, env=env,
    )
    assert proc.returncode == 0, f"probe failed ({mode}):\n{proc.stdout}\n{proc.stderr}"
    return torch.load(out, weights_only=False)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("bw")
    agent = _probe(_agent_src(), "agent-backward", root / "agent.pt")
    truth = _probe(_PRISTINE, "forward-vjp", root / "truth.pt")
    return agent, truth


def _check(results, name):
    agent, truth = results
    a = agent[name]
    assert a["ok"], (
        f"{name}: the backward tripped the runtime sensor — it relies on "
        f"autograd or calls the forward operator instead of a hand-derived "
        f"adjoint ({a.get('err')})"
    )
    got, want = a["val"], truth[name]
    scale = max(want.abs().max().item(), 1.0)
    diff = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-8 * scale), (
        f"{name}: adjoint disagrees with the forward's transpose "
        f"(max abs diff {diff:.3e})"
    )


def test_backward_norm_layernorm_adjoint(results):
    """backward_norm is the exact adjoint of apply_norm_linear (LayerNorm)."""
    _check(results, "norm_ln")


def test_backward_norm_rmsnorm_adjoint(results):
    """backward_norm is the exact adjoint (RMSNorm — no mean-centering)."""
    _check(results, "norm_rms")


def test_backward_norm_jacobian_radial_adjoint(results):
    """backward_norm handles the jacobian-mode radial projection adjoint."""
    _check(results, "norm_radial")


def test_backward_attention_mha_adjoint(results):
    """backward_attention is the exact adjoint of apply_attention_linear
    (multi-head, frozen pattern — V/O/pattern transpose + norm adjoint)."""
    _check(results, "attn_mha")


def test_backward_attention_qk_formation_adjoint(results):
    """backward_attention handles the qk-jacobian pattern-formation adjoint
    (softmax-Jacobian through RoPE + per-head QK-RMSNorm) — the hard rung."""
    _check(results, "attn_qk")
