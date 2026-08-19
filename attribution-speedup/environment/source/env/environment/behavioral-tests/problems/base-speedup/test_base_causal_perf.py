"""Mechanism discriminators for base-* problems: does sweep cost track causality?

Grades the OBSERVABLE property, not the implementation: in a causal model,
features that sit early in the prompt (or low in the layer stack) have small
causal cones, so attributing them should be cheaper than attributing
late-position (or high-layer) features on an otherwise identical graph. A
backward sweep that always pays full prompt length x full depth shows no such
asymmetry.

Base-shaped twin of test_causal_backward_perf.py: same methodology
(same-size synthetic IRs differing only in where the active features live;
self-normalizing t(expensive)/t(cheap) ratio; single-thread process_time;
warmup + best-of-3), fixtures constructed with the base IR signature, and
DIFFERENT test names so the two files' junit entries never collide.

Gated on LAT_BENCH_PRISTINE_SRC (set only for base-* problem ids): on every
other problem this file skips (neutral p2p). Ratios are recorded to the
metrics sidecar before the assert so the grader can score them continuously.
"""

import functools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

from _lib.grade_env import gold_src, pristine_src, record_metric as _record_metric


@functools.lru_cache(maxsize=1)
def _gold_asym() -> dict:
    """Time the GRADER-OWNED gold reference's causal asymmetries on THIS host,
    in a clean subprocess (the agent's ``lat`` is already imported here, so gold
    cannot share this process). Returns {} when gold was not staged — the grader
    then keeps the x86-calibrated constant targets. Cached so the two asymmetry
    tests share one gold run."""
    gold = gold_src()
    if not (gold and Path(gold, "lat", "__init__.py").exists()):
        return {}
    runner = Path(__file__).parents[2] / "_lib" / "asym_gold_runner.py"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    proc = subprocess.run(
        [sys.executable, str(runner), "--src", gold, "--out", out],
        capture_output=True, text=True, timeout=15 * 60,
    )
    if proc.returncode != 0:
        return {}
    try:
        return json.load(open(out))
    except (OSError, ValueError):
        return {}

pytestmark = pytest.mark.skipif(
    not pristine_src(),
    reason="base-* discriminator (LAT_BENCH_PRISTINE_SRC not set for this problem)",
)

import lat  # noqa: E402
from lat.ir import (  # noqa: E402
    LinearizedAttention,
    LinearizedAttribution,
    LinearizedLayer,
    LinearizedMLP,
    NormSnapshot,
)

# Calibrated on the naive base + the gold-causality port (2026-07-15):
# see the grader's CONTINUOUS_PERF for the partial-credit targets.
_MIN_TEMPORAL_ASYMMETRY = 1.5
_MIN_LAYER_ASYMMETRY = 3.0


class _StubTokenizer:
    def decode(self, ids):
        if isinstance(ids, int):
            return f"<t{ids}>"
        return " ".join(f"<t{int(i)}>" for i in torch.as_tensor(ids).flatten())


def _norm(n_pos: int, d_model: int) -> NormSnapshot:
    return NormSnapshot(
        scale=torch.ones(n_pos), gamma=torch.ones(d_model), mean=None, beta=None
    )


def _build_ir(*, n_layers, n_pos, d_model, feats_per_layer, feat_positions,
              feat_layers=None, seed=0):
    """Synthetic causal IR whose active features live only at
    ``feat_positions`` (and only in ``feat_layers`` when given)."""
    g = torch.Generator().manual_seed(seed)
    base_pattern = torch.tril(torch.ones(n_pos, n_pos))
    base_pattern = base_pattern / base_pattern.sum(-1, keepdim=True)
    layers = []
    for i in range(n_layers):
        has_feats = feat_layers is None or i in feat_layers
        if has_feats:
            reps = max(1, feats_per_layer // len(feat_positions))
            positions = torch.tensor(sorted(feat_positions * reps), dtype=torch.long)
            n_active = positions.shape[0]
            w = torch.randn(n_active, d_model, generator=g) * 0.3
            mlp = LinearizedMLP(
                active_positions=positions,
                active_feature_idx=torch.arange(n_active, dtype=torch.long),
                active_activations=torch.ones(n_active),
                W_enc_active=w,
                W_dec_active=w * 0.9,
                b_enc_active=torch.zeros(n_active),
                b_dec=torch.zeros(d_model),
                d_transcoder=max(n_active, 64),
                pre_norm=_norm(n_pos, d_model),
                error_vec=torch.zeros(n_pos, d_model),
            )
        else:
            mlp = LinearizedMLP(
                active_positions=torch.zeros(0, dtype=torch.long),
                active_feature_idx=torch.zeros(0, dtype=torch.long),
                active_activations=torch.zeros(0),
                W_enc_active=torch.zeros(0, d_model),
                W_dec_active=torch.zeros(0, d_model),
                b_enc_active=torch.zeros(0),
                b_dec=torch.zeros(d_model),
                d_transcoder=64,
                pre_norm=_norm(n_pos, d_model),
                error_vec=torch.zeros(n_pos, d_model),
            )
        attention = LinearizedAttention(
            pattern=base_pattern.unsqueeze(0).repeat(2, 1, 1) * 0.05,
            W_V=torch.randn(2, d_model, 4, generator=g) * 0.1,
            W_O=torch.randn(2, 4, d_model, generator=g) * 0.1,
            b_V=None,
            b_O=None,
            pre_norm=_norm(n_pos, d_model),
            is_causal=True,
        )
        layers.append(LinearizedLayer(layer_idx=i, attention=attention, mlp=mlp))

    return LinearizedAttribution(
        input_tokens=torch.zeros(n_pos, dtype=torch.long),
        resid_0=torch.randn(n_pos, d_model, generator=g) * 0.1,
        embed_vecs=torch.randn(n_pos, d_model, generator=g) * 0.1,
        layers=tuple(layers),
        final_norm=_norm(n_pos, d_model),
        W_unembed=torch.randn(d_model, 32, generator=g),
        b_unembed=None,
        logits=torch.randn(n_pos, 32, generator=g),
    )


def _t_run(ir, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.process_time()
        lat.attribute_from_linearized(
            ir, _StubTokenizer(), show_progress=False, batch_size=16
        )
        best = min(best, time.process_time() - t0)
    return best


def test_base_backward_cost_respects_temporal_causality():
    """Early-position features have tiny causal cones; attributing them must
    be substantially cheaper than attributing late-position features on an
    identical-size graph."""
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        early = _build_ir(n_layers=4, n_pos=256, d_model=16, feats_per_layer=256,
                          feat_positions=[1, 2, 3, 4])
        late = _build_ir(n_layers=4, n_pos=256, d_model=16, feats_per_layer=256,
                         feat_positions=[252, 253, 254, 255])
        _t_run(early, reps=1)  # warmup
        t_early = _t_run(early)
        t_late = _t_run(late)
    finally:
        torch.set_num_threads(n_threads)
    ratio = t_late / t_early
    _record_metric("temporal_asymmetry", ratio)
    g = _gold_asym().get("temporal_asymmetry")
    if isinstance(g, (int, float)):
        _record_metric("temporal_asymmetry_gold", g)
    assert ratio >= _MIN_TEMPORAL_ASYMMETRY, (
        f"backward cost ignores temporal causality: late={t_late:.2f}s vs "
        f"early={t_early:.2f}s (ratio {ratio:.2f} < {_MIN_TEMPORAL_ASYMMETRY}). "
        f"An early-position feature's causal cone is a few tokens; attributing "
        f"it should not cost the same as a full-prompt sweep."
    )


def test_base_backward_cost_respects_layer_causality():
    """Low-layer features can only be influenced from below; attributing them
    must be substantially cheaper than attributing high-layer features."""
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        low = _build_ir(n_layers=64, n_pos=8, d_model=16, feats_per_layer=128,
                        feat_positions=list(range(8)), feat_layers=[0, 1, 2, 3])
        high = _build_ir(n_layers=64, n_pos=8, d_model=16, feats_per_layer=128,
                         feat_positions=list(range(8)),
                         feat_layers=[60, 61, 62, 63])
        _t_run(low, reps=1)  # warmup
        t_low = _t_run(low)
        t_high = _t_run(high)
    finally:
        torch.set_num_threads(n_threads)
    ratio = t_high / t_low
    _record_metric("layer_asymmetry", ratio)
    g = _gold_asym().get("layer_asymmetry")
    if isinstance(g, (int, float)):
        _record_metric("layer_asymmetry_gold", g)
    assert ratio >= _MIN_LAYER_ASYMMETRY, (
        f"backward cost ignores layer causality: high={t_high:.2f}s vs "
        f"low={t_low:.2f}s (ratio {ratio:.2f} < {_MIN_LAYER_ASYMMETRY}). "
        f"Gradient from a layer-3 feature cannot reach layers above it; the "
        f"sweep should not pay for them."
    )
