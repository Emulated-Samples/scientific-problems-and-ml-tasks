"""Parity anchors for base-port-parity: the port must match the REFERENCE.

Each test dual-runs one seeded fixture: the reference side through the
GRADER-OWNED pristine ``lat`` (LAT_BENCH_PRISTINE_SRC — the agent cannot
edit it, so "make the reference match the port" earns nothing), the fast
side through the agent's ``lat_fast``. Fixtures isolate one public contract
each; see _anchor_runner.py.

Self-gated: runs only when the staged pristine tree carries lat_fast (the
base-port-parity problem); skips neutrally everywhere else.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _lib.grade_env import pristine_src

_PRISTINE = pristine_src()


def _pristine_has_port() -> bool:
    return bool(_PRISTINE) and (Path(_PRISTINE) / "lat_fast" / "__init__.py").exists()


pytestmark = pytest.mark.skipif(
    not _pristine_has_port(),
    reason="port-parity anchors (pristine tree does not carry the port)",
)


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat_fast" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat_fast src not found on PYTHONPATH")


def _run(anchor: str, engine: str, src: str, out: Path) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    runner = Path(__file__).parent / "_anchor_runner.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--src", src, "--engine", engine,
         "--anchor", anchor, "--out", str(out)],
        capture_output=True, text=True, timeout=5 * 60, env=env,
    )
    assert proc.returncode == 0, (
        f"anchor runner failed ({anchor}/{engine}, src={src}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    return torch.load(out, weights_only=True)


def _assert_match(anchor: str, tmp_path: Path) -> None:
    ref = _run(anchor, "ref", _PRISTINE, tmp_path / "ref.pt")
    fast = _run(anchor, "fast", _agent_src(), tmp_path / "fast.pt")
    for name in ("features", "errors", "tokens"):
        a, b = ref[name], fast[name]
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), (
            f"{anchor}: port diverges from the reference on '{name}' "
            f"(max abs diff {(a - b).abs().max().item():.3e})"
        )


def test_port_matches_reference_on_mixed_causality_models(tmp_path):
    """Attribution rows agree with the reference when the layer stack mixes
    causal and non-causal attention patterns."""
    _assert_match("mixed_causality", tmp_path)


def test_port_matches_reference_error_attribution(tmp_path):
    """Error-node attribution agrees with the reference at every position,
    including the attributed position itself."""
    _assert_match("error_rows", tmp_path)


def test_port_matches_reference_on_feature_batches(tmp_path):
    """Feature-injection rows agree with the reference — a feature's own
    injection must not contaminate its layer's records."""
    _assert_match("feature_records", tmp_path)


def test_port_matches_reference_in_jacobian_mode(tmp_path):
    """Attribution rows agree with the reference when norm snapshots carry
    the jacobian-mode radial operating point."""
    _assert_match("jacobian_mode", tmp_path)
