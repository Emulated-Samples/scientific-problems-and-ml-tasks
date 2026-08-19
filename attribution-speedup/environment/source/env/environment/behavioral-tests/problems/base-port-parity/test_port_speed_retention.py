"""Speed-retention measurement for base-port-parity (never a scored test).

Measures the agent's ``lat_fast`` against the GRADER-OWNED pristine
``lat_fast`` on the mixed no-cap fixtures (same discipline as the base
speedup harness: subprocess per side, single-thread process_time, warmup +
best-of-3) and writes ``port_speed_retention`` = t_pristine / t_agent to
the metrics sidecar. The grader applies it as a MULTIPLIER on all f2p
credit — a port "fixed" by reverting to the slow reference passes every
parity anchor and still scores ~0. This test itself always passes when the
measurement completes; a crashed measurement errors the test and leaves the
metric missing, which the grader fails closed.

Self-gated like the anchors: pristine tree must carry the port.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _lib.grade_env import pristine_src, record_metric as _record_metric

_PRISTINE = pristine_src()


def _pristine_has_port() -> bool:
    return bool(_PRISTINE) and (Path(_PRISTINE) / "lat_fast" / "__init__.py").exists()


pytestmark = pytest.mark.skipif(
    not _pristine_has_port(),
    reason="port speed-retention harness (pristine tree does not carry the port)",
)

_RUNNER = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(1)
import lat_fast
from lat.ir import (LinearizedAttention, LinearizedAttribution,
                    LinearizedLayer, LinearizedMLP, NormSnapshot)


class Tok:
    def decode(self, ids):
        if isinstance(ids, int):
            return f"<t{ids}>"
        return " ".join(f"<t{int(i)}>" for i in torch.as_tensor(ids).flatten())


def norm(n_pos, d, g):
    return NormSnapshot(scale=torch.rand(n_pos, generator=g) + 0.5,
                        gamma=torch.randn(d, generator=g) * 0.2 + 1.0,
                        mean=None, beta=None)


def build(n_layers, n_pos, d, fpp, seed, heads=4, dh=8):
    g = torch.Generator().manual_seed(seed)
    pattern = torch.tril(torch.ones(n_pos, n_pos))
    pattern = pattern / pattern.sum(-1, keepdim=True)
    n_active = fpp * n_pos
    positions = torch.arange(n_pos, dtype=torch.long).repeat_interleave(fpp)
    layers = []
    for i in range(n_layers):
        attention = LinearizedAttention(
            pattern=pattern.unsqueeze(0).repeat(heads, 1, 1) * 0.3,
            W_V=torch.randn(heads, d, dh, generator=g) * 0.2,
            W_O=torch.randn(heads, dh, d, generator=g) * 0.2,
            b_V=None, b_O=None, pre_norm=norm(n_pos, d, g), is_causal=True)
        mlp = LinearizedMLP(
            active_positions=positions,
            active_feature_idx=torch.arange(n_active, dtype=torch.long),
            active_activations=torch.rand(n_active, generator=g) + 0.2,
            W_enc_active=torch.randn(n_active, d, generator=g) * 0.3,
            W_dec_active=torch.randn(n_active, d, generator=g) * 0.3,
            b_enc_active=torch.zeros(n_active), b_dec=torch.zeros(d),
            d_transcoder=n_active, pre_norm=norm(n_pos, d, g),
            error_vec=torch.randn(n_pos, d, generator=g) * 0.05)
        layers.append(LinearizedLayer(layer_idx=i, attention=attention, mlp=mlp))
    return LinearizedAttribution(
        input_tokens=torch.arange(n_pos, dtype=torch.long) % 11,
        resid_0=torch.randn(n_pos, d, generator=g) * 0.3,
        embed_vecs=torch.randn(n_pos, d, generator=g) * 0.3,
        layers=tuple(layers), final_norm=norm(n_pos, d, g),
        W_unembed=torch.randn(d, 32, generator=g), b_unembed=None,
        logits=torch.randn(n_pos, 32, generator=g))


for name, kw in [("long", dict(n_layers=6, n_pos=96, d=40, fpp=6, seed=101)),
                 ("deep", dict(n_layers=32, n_pos=16, d=40, fpp=8, seed=202))]:
    ir = build(**kw)
    lat_fast.attribute_from_linearized(ir, Tok(), show_progress=False, batch_size=64)
    best = float("inf")
    for _ in range(3):
        t0 = time.process_time()
        lat_fast.attribute_from_linearized(ir, Tok(), show_progress=False, batch_size=64)
        best = min(best, time.process_time() - t0)
    print(f"{name}={best}")
"""


def _time_side(src: str, tmp: Path, tag: str) -> dict[str, float]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    script = tmp / f"retention_runner_{tag}.py"
    script.write_text(_RUNNER)
    proc = subprocess.run(
        [sys.executable, str(script), src],
        capture_output=True, text=True, timeout=15 * 60, env=env,
    )
    assert proc.returncode == 0, (
        f"retention runner failed for src={src}:\n{proc.stdout}\n{proc.stderr}"
    )
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=")
            out[k] = float(v)
    assert set(out) == {"long", "deep"}, f"unexpected runner output: {proc.stdout}"
    return out


def _agent_src() -> str:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat_fast" / "__init__.py").exists():
            return entry
    raise AssertionError("agent lat_fast src not found on PYTHONPATH")


def test_port_speed_retention(tmp_path):
    """Measure agent-port vs pristine-port speed; the grader converts the
    recorded ratio to a multiplier. This test only fails if the measurement
    itself cannot run (which fails the grader closed via the missing metric)."""
    pristine = _time_side(_PRISTINE, tmp_path, "pristine")
    agent = _time_side(_agent_src(), tmp_path, "agent")
    ratios = {k: pristine[k] / max(agent[k], 1e-9) for k in pristine}
    combined = 1.0
    for r in ratios.values():
        combined *= r
    combined = combined ** (1.0 / len(ratios))
    for k, r in ratios.items():
        _record_metric(f"port_retention_{k}", r)
    _record_metric("port_speed_retention", combined)
