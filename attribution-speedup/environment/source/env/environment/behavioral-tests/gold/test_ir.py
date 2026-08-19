"""Tests for the IR dataclasses.

Two kinds of tests:

- Happy-path: derived properties (``n_active``, ``n_pos``, ...) report the
  right values when the IR is constructed correctly.
- Sad-path: ``__post_init__`` raises ``ValueError`` for each shape
  inconsistency the dataclass is supposed to catch.

Helpers build minimal valid instances; sad-path tests construct invalid
instances inline so the test reads as "here's the bad config".
"""

import pytest
import torch

from lat.ir import (
    LinearizedAttention,
    LinearizedAttribution,
    LinearizedLayer,
    LinearizedMLP,
    NormSnapshot,
)

# ─── helpers (happy-path defaults) ───────────────────────────────────────────


def _norm_snapshot(
    *, n_pos: int = 4, d_model: int = 8, with_mean: bool = False, with_beta: bool = False
) -> NormSnapshot:
    return NormSnapshot(
        scale=torch.ones(n_pos),
        gamma=torch.ones(d_model),
        mean=torch.zeros(n_pos) if with_mean else None,
        beta=torch.zeros(d_model) if with_beta else None,
    )


def _attention(
    *, n_pos: int = 4, n_heads: int = 2, d_model: int = 8, d_head: int = 4
) -> LinearizedAttention:
    return LinearizedAttention(
        pattern=torch.zeros(n_heads, n_pos, n_pos),
        W_V=torch.zeros(n_heads, d_model, d_head),
        W_O=torch.zeros(n_heads, d_head, d_model),
        b_V=None,
        b_O=None,
        pre_norm=_norm_snapshot(n_pos=n_pos, d_model=d_model),
    )


def _mlp(*, n_pos: int = 4, d_model: int = 8, n_active: int = 3) -> LinearizedMLP:
    """Build a valid MLP with `n_active` features at positions 0, 1, 2, ... (capped at n_pos)."""
    positions = torch.arange(min(n_active, n_pos), dtype=torch.long)
    # If n_active > n_pos, pad by repeating the last valid position so the
    # invariant "ascending" still holds.
    if n_active > n_pos:
        tail = torch.full((n_active - n_pos,), n_pos - 1, dtype=torch.long)
        positions = torch.cat([positions, tail])
    return LinearizedMLP(
        active_positions=positions,
        active_feature_idx=torch.arange(n_active, dtype=torch.long),
        active_activations=torch.ones(n_active),
        W_enc_active=torch.zeros(n_active, d_model),
        W_dec_active=torch.zeros(n_active, d_model),
        b_enc_active=torch.zeros(n_active),
        b_dec=torch.zeros(d_model),
        d_transcoder=64,
        pre_norm=_norm_snapshot(n_pos=n_pos, d_model=d_model),
        error_vec=torch.zeros(n_pos, d_model),
        position_groups=_groups_one_per_position(positions),
    )


def _groups_one_per_position(positions: torch.Tensor) -> tuple[tuple[int, int, int], ...]:
    """Build position_groups from a sorted positions tensor."""
    if positions.numel() == 0:
        return ()
    unique_pos, counts = torch.unique_consecutive(positions, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts[:-1].cumsum(0)])
    return tuple(
        (int(p), int(s), int(c))
        for p, s, c in zip(unique_pos.tolist(), starts.tolist(), counts.tolist(), strict=True)
    )


def _attribution(
    *,
    n_pos: int = 4,
    d_model: int = 8,
    n_layers: int = 2,
    n_active_per_layer: int = 3,
    d_vocab: int = 16,
) -> LinearizedAttribution:
    layers = tuple(
        LinearizedLayer(
            layer_idx=i,
            attention=_attention(n_pos=n_pos, d_model=d_model),
            mlp=_mlp(n_pos=n_pos, d_model=d_model, n_active=n_active_per_layer),
        )
        for i in range(n_layers)
    )
    return LinearizedAttribution(
        input_tokens=torch.zeros(n_pos, dtype=torch.long),
        resid_0=torch.zeros(n_pos, d_model),
        embed_vecs=torch.zeros(n_pos, d_model),
        layers=layers,
        final_norm=_norm_snapshot(n_pos=n_pos, d_model=d_model),
        W_unembed=torch.zeros(d_model, d_vocab),
        b_unembed=None,
        logits=torch.zeros(n_pos, d_vocab),
    )


# ─── NormSnapshot ────────────────────────────────────────────────────────────


class TestNormSnapshot:
    def test_has_mean_centering_true_when_mean_present(self):
        assert _norm_snapshot(with_mean=True).has_mean_centering

    def test_has_mean_centering_false_when_mean_absent(self):
        assert not _norm_snapshot(with_mean=False).has_mean_centering


# ─── LinearizedMLP ───────────────────────────────────────────────────────────


class TestLinearizedMLP:
    def test_n_active_reports_count(self):
        mlp = _mlp(n_active=5, n_pos=8)
        assert mlp.n_active == 5

    def test_empty_active_is_valid(self):
        # n_active = 0 must validate cleanly: it's the right answer for layers
        # where no feature fired on this prompt.
        mlp = _mlp(n_active=0)
        assert mlp.n_active == 0
        assert mlp.position_groups == ()

    def test_feature_idx_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="active_feature_idx"):
            LinearizedMLP(
                active_positions=torch.zeros(3, dtype=torch.long),
                active_feature_idx=torch.zeros(2, dtype=torch.long),  # wrong: 2 != 3
                active_activations=torch.zeros(3),
                W_enc_active=torch.zeros(3, 8),
                W_dec_active=torch.zeros(3, 8),
                b_enc_active=torch.zeros(3),
                b_dec=torch.zeros(8),
                d_transcoder=64,
                pre_norm=_norm_snapshot(),
                error_vec=torch.zeros(4, 8),
                position_groups=((0, 0, 3),),
            )

    def test_activations_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="active_activations"):
            LinearizedMLP(
                active_positions=torch.zeros(3, dtype=torch.long),
                active_feature_idx=torch.zeros(3, dtype=torch.long),
                active_activations=torch.zeros(2),  # wrong
                W_enc_active=torch.zeros(3, 8),
                W_dec_active=torch.zeros(3, 8),
                b_enc_active=torch.zeros(3),
                b_dec=torch.zeros(8),
                d_transcoder=64,
                pre_norm=_norm_snapshot(),
                error_vec=torch.zeros(4, 8),
                position_groups=((0, 0, 3),),
            )

    def test_w_enc_first_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="W_enc_active"):
            LinearizedMLP(
                active_positions=torch.zeros(3, dtype=torch.long),
                active_feature_idx=torch.zeros(3, dtype=torch.long),
                active_activations=torch.zeros(3),
                W_enc_active=torch.zeros(2, 8),  # wrong: 2 rows, expected 3
                W_dec_active=torch.zeros(3, 8),
                b_enc_active=torch.zeros(3),
                b_dec=torch.zeros(8),
                d_transcoder=64,
                pre_norm=_norm_snapshot(),
                error_vec=torch.zeros(4, 8),
                position_groups=((0, 0, 3),),
            )

    def test_w_dec_first_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="W_dec_active"):
            LinearizedMLP(
                active_positions=torch.zeros(3, dtype=torch.long),
                active_feature_idx=torch.zeros(3, dtype=torch.long),
                active_activations=torch.zeros(3),
                W_enc_active=torch.zeros(3, 8),
                W_dec_active=torch.zeros(2, 8),  # wrong
                b_enc_active=torch.zeros(3),
                b_dec=torch.zeros(8),
                d_transcoder=64,
                pre_norm=_norm_snapshot(),
                error_vec=torch.zeros(4, 8),
                position_groups=((0, 0, 3),),
            )

    def test_position_groups_undercount_raises(self):
        # Groups sum to fewer features than n_active.
        with pytest.raises(ValueError, match="position_groups"):
            LinearizedMLP(
                active_positions=torch.zeros(3, dtype=torch.long),
                active_feature_idx=torch.zeros(3, dtype=torch.long),
                active_activations=torch.zeros(3),
                W_enc_active=torch.zeros(3, 8),
                W_dec_active=torch.zeros(3, 8),
                b_enc_active=torch.zeros(3),
                b_dec=torch.zeros(8),
                d_transcoder=64,
                pre_norm=_norm_snapshot(),
                error_vec=torch.zeros(4, 8),
                position_groups=((0, 0, 1),),  # only 1, expected 3
            )

    def test_position_groups_overcount_raises(self):
        with pytest.raises(ValueError, match="position_groups"):
            LinearizedMLP(
                active_positions=torch.zeros(3, dtype=torch.long),
                active_feature_idx=torch.zeros(3, dtype=torch.long),
                active_activations=torch.zeros(3),
                W_enc_active=torch.zeros(3, 8),
                W_dec_active=torch.zeros(3, 8),
                b_enc_active=torch.zeros(3),
                b_dec=torch.zeros(8),
                d_transcoder=64,
                pre_norm=_norm_snapshot(),
                error_vec=torch.zeros(4, 8),
                position_groups=((0, 0, 5),),  # 5, expected 3
            )


# ─── LinearizedAttribution ───────────────────────────────────────────────────


class TestLinearizedAttribution:
    def test_basic_properties(self):
        ir = _attribution(n_pos=6, d_model=12, n_layers=3, n_active_per_layer=2)
        assert ir.n_pos == 6
        assert ir.n_layers == 3
        assert ir.d_model == 12

    def test_total_active_features_sums_across_layers(self):
        ir = _attribution(n_layers=3, n_active_per_layer=4)
        assert ir.total_active_features == 12

    def test_total_active_features_empty(self):
        ir = _attribution(n_layers=2, n_active_per_layer=0)
        assert ir.total_active_features == 0

    def test_resid_0_n_pos_mismatch_raises(self):
        with pytest.raises(ValueError, match="resid_0 n_pos"):
            LinearizedAttribution(
                input_tokens=torch.zeros(4, dtype=torch.long),
                resid_0=torch.zeros(3, 8),  # wrong: 3 != 4
                embed_vecs=torch.zeros(4, 8),
                layers=(),
                final_norm=_norm_snapshot(),
                W_unembed=torch.zeros(8, 16),
                b_unembed=None,
                logits=torch.zeros(4, 16),
            )

    def test_embed_vecs_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="embed_vecs shape"):
            LinearizedAttribution(
                input_tokens=torch.zeros(4, dtype=torch.long),
                resid_0=torch.zeros(4, 8),
                embed_vecs=torch.zeros(4, 16),  # wrong d_model
                layers=(),
                final_norm=_norm_snapshot(),
                W_unembed=torch.zeros(8, 16),
                b_unembed=None,
                logits=torch.zeros(4, 16),
            )

    def test_layer_error_vec_shape_mismatch_raises(self):
        # Construct a layer whose mlp.error_vec has the wrong shape, and
        # verify the top-level IR's validation catches it.
        bad_mlp = LinearizedMLP(
            active_positions=torch.zeros(0, dtype=torch.long),
            active_feature_idx=torch.zeros(0, dtype=torch.long),
            active_activations=torch.zeros(0),
            W_enc_active=torch.zeros(0, 8),
            W_dec_active=torch.zeros(0, 8),
            b_enc_active=torch.zeros(0),
            b_dec=torch.zeros(8),
            d_transcoder=64,
            pre_norm=_norm_snapshot(),
            error_vec=torch.zeros(3, 8),  # n_pos=3, but top-level uses n_pos=4
            position_groups=(),
        )
        layer = LinearizedLayer(layer_idx=0, attention=_attention(n_pos=4, d_model=8), mlp=bad_mlp)
        with pytest.raises(ValueError, match="error_vec"):
            LinearizedAttribution(
                input_tokens=torch.zeros(4, dtype=torch.long),
                resid_0=torch.zeros(4, 8),
                embed_vecs=torch.zeros(4, 8),
                layers=(layer,),
                final_norm=_norm_snapshot(),
                W_unembed=torch.zeros(8, 16),
                b_unembed=None,
                logits=torch.zeros(4, 16),
            )
