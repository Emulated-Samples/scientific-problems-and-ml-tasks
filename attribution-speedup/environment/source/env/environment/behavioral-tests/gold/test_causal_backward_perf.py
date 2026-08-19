"""Graded behavioral tests: the backward sweep must exploit causality.

Grades the OBSERVABLE property, not the implementation: in a causal model,
features that sit early in the prompt (or low in the layer stack) have small
causal cones, so attributing them should be cheaper than attributing
late-position (or high-layer) features on an otherwise identical graph. A
backward sweep that always pays full prompt length x full depth shows no such
asymmetry.

Discriminator shape: two same-size synthetic IRs differing only in where the
active features live; the asymmetry ratio t(expensive)/t(cheap) is
self-normalizing (machine speed cancels). Calibrated 2026-07-14:

  temporal (late/early):  gold 2.46x   naive full-matrix 1.03x
  layer   (high/low):     gold 8.48x   naive full-matrix 1.01x

Thresholds sit between the bands; ratios are also recorded to the metrics
sidecar so discovery-style problems can score them continuously.

No circuit_tracer dependency — nothing in this module can skip.
"""

import time

import torch

from _lib.grade_env import record_metric as _record_metric

import lat
from lat.ir import (
    LinearizedAttention,
    LinearizedAttribution,
    LinearizedLayer,
    LinearizedMLP,
    NormSnapshot,
)

_MIN_TEMPORAL_ASYMMETRY = 1.5  # gold 2.46, naive 1.03
_MIN_LAYER_ASYMMETRY = 3.0  # gold 8.48, naive 1.01


class _StubTokenizer:
    def decode(self, ids):
        if isinstance(ids, int):
            return f"<t{ids}>"
        return " ".join(f"<t{int(i)}>" for i in torch.as_tensor(ids).flatten())


def _norm(n_pos: int, d_model: int) -> NormSnapshot:
    return NormSnapshot(
        scale=torch.ones(n_pos), gamma=torch.ones(d_model), mean=None, beta=None
    )


def _groups(positions: torch.Tensor):
    if positions.numel() == 0:
        return ()
    uniq, counts = torch.unique_consecutive(positions, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts[:-1].cumsum(0)])
    return tuple(
        (int(p), int(s), int(c))
        for p, s, c in zip(uniq.tolist(), starts.tolist(), counts.tolist(), strict=True)
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
                position_groups=_groups(positions),
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
                position_groups=(),
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


def test_backward_cost_respects_temporal_causality():
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
    assert ratio >= _MIN_TEMPORAL_ASYMMETRY, (
        f"backward cost ignores temporal causality: late={t_late:.2f}s vs "
        f"early={t_early:.2f}s (ratio {ratio:.2f} < {_MIN_TEMPORAL_ASYMMETRY}). "
        f"An early-position feature's causal cone is a few tokens; attributing "
        f"it should not cost the same as a full-prompt sweep."
    )


def test_backward_cost_respects_layer_causality():
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
    assert ratio >= _MIN_LAYER_ASYMMETRY, (
        f"backward cost ignores layer causality: high={t_high:.2f}s vs "
        f"low={t_low:.2f}s (ratio {ratio:.2f} < {_MIN_LAYER_ASYMMETRY}). "
        f"Gradient from a layer-3 feature cannot reach layers above it; the "
        f"sweep should not pay for them."
    )
