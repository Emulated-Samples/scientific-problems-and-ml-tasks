"""Shape/consistency validation for the IR dataclasses.

The IR is the contract between linearize and everything downstream, so its
``__post_init__`` checks are load-bearing: a malformed IR should fail at
construction, not deep inside a backward sweep.
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

_N_POS = 3
_D_MODEL = 4
_D_VOCAB = 8


def _norm(n_pos=_N_POS, d_model=_D_MODEL, *, layernorm=False) -> NormSnapshot:
    return NormSnapshot(
        scale=torch.ones(n_pos),
        gamma=torch.ones(d_model),
        mean=torch.zeros(n_pos) if layernorm else None,
        beta=torch.zeros(d_model) if layernorm else None,
    )


def _attention(n_heads=2, d_head=2) -> LinearizedAttention:
    return LinearizedAttention(
        pattern=torch.zeros(n_heads, _N_POS, _N_POS),
        W_V=torch.zeros(n_heads, _D_MODEL, d_head),
        W_O=torch.zeros(n_heads, d_head, _D_MODEL),
        b_V=None,
        b_O=None,
        pre_norm=_norm(),
    )


def _mlp(n_active=2, **overrides) -> LinearizedMLP:
    kwargs = dict(
        active_positions=torch.zeros(n_active, dtype=torch.long),
        active_feature_idx=torch.zeros(n_active, dtype=torch.long),
        active_activations=torch.zeros(n_active),
        W_enc_active=torch.zeros(n_active, _D_MODEL),
        W_dec_active=torch.zeros(n_active, _D_MODEL),
        b_enc_active=torch.zeros(n_active),
        b_dec=torch.zeros(_D_MODEL),
        d_transcoder=16,
        pre_norm=_norm(),
        error_vec=torch.zeros(_N_POS, _D_MODEL),
    )
    kwargs.update(overrides)
    return LinearizedMLP(**kwargs)


def _ir(layers=None, **overrides) -> LinearizedAttribution:
    if layers is None:
        layers = (LinearizedLayer(layer_idx=0, attention=_attention(), mlp=_mlp()),)
    kwargs = dict(
        input_tokens=torch.zeros(_N_POS, dtype=torch.long),
        resid_0=torch.zeros(_N_POS, _D_MODEL),
        embed_vecs=torch.zeros(_N_POS, _D_MODEL),
        layers=layers,
        final_norm=_norm(),
        W_unembed=torch.zeros(_D_MODEL, _D_VOCAB),
        b_unembed=None,
        logits=torch.zeros(_N_POS, _D_VOCAB),
    )
    kwargs.update(overrides)
    return LinearizedAttribution(**kwargs)


# ─── NormSnapshot ───────────────────────────────────────────────────────────


def test_has_mean_centering_true_when_mean_present():
    assert _norm(layernorm=True).has_mean_centering


def test_has_mean_centering_false_when_mean_absent():
    assert not _norm(layernorm=False).has_mean_centering


# ─── LinearizedAttention ────────────────────────────────────────────────────


def test_attention_head_count_mismatch_raises():
    with pytest.raises(ValueError, match="heads"):
        LinearizedAttention(
            pattern=torch.zeros(2, _N_POS, _N_POS),
            W_V=torch.zeros(2, _D_MODEL, 2),
            W_O=torch.zeros(3, 2, _D_MODEL),  # 3 heads vs W_V's 2
            b_V=None,
            b_O=None,
            pre_norm=_norm(),
        )


def test_has_qk_false_by_default():
    assert not _attention().has_qk


# ─── LinearizedMLP ──────────────────────────────────────────────────────────


def test_n_active_reports_count():
    assert _mlp(n_active=5).n_active == 5


def test_empty_active_is_valid():
    assert _mlp(n_active=0).n_active == 0


def test_feature_idx_shape_mismatch_raises():
    with pytest.raises(ValueError, match="active_feature_idx"):
        _mlp(active_feature_idx=torch.zeros(3, dtype=torch.long))


def test_activations_shape_mismatch_raises():
    with pytest.raises(ValueError, match="active_activations"):
        _mlp(active_activations=torch.zeros(3))


def test_b_enc_shape_mismatch_raises():
    with pytest.raises(ValueError, match="b_enc_active"):
        _mlp(b_enc_active=torch.zeros(3))


def test_w_enc_first_dim_mismatch_raises():
    with pytest.raises(ValueError, match="W_enc_active"):
        _mlp(W_enc_active=torch.zeros(3, _D_MODEL))


def test_w_dec_first_dim_mismatch_raises():
    with pytest.raises(ValueError, match="W_dec_active"):
        _mlp(W_dec_active=torch.zeros(3, _D_MODEL))


# ─── LinearizedAttribution ──────────────────────────────────────────────────


def test_basic_properties():
    ir = _ir()
    assert ir.n_pos == _N_POS
    assert ir.n_layers == 1
    assert ir.d_model == _D_MODEL


def test_total_active_features_sums_across_layers():
    layers = (
        LinearizedLayer(layer_idx=0, attention=_attention(), mlp=_mlp(n_active=2)),
        LinearizedLayer(layer_idx=1, attention=_attention(), mlp=_mlp(n_active=3)),
    )
    assert _ir(layers=layers).total_active_features == 5


def test_total_active_features_empty():
    layers = (LinearizedLayer(layer_idx=0, attention=_attention(), mlp=_mlp(n_active=0)),)
    assert _ir(layers=layers).total_active_features == 0


def test_resid_0_n_pos_mismatch_raises():
    with pytest.raises(ValueError, match="resid_0"):
        _ir(resid_0=torch.zeros(_N_POS + 1, _D_MODEL))


def test_embed_vecs_shape_mismatch_raises():
    with pytest.raises(ValueError, match="embed_vecs"):
        _ir(embed_vecs=torch.zeros(_N_POS, _D_MODEL + 1))


def test_layer_error_vec_shape_mismatch_raises():
    bad = LinearizedLayer(
        layer_idx=0,
        attention=_attention(),
        mlp=_mlp(error_vec=torch.zeros(_N_POS + 1, _D_MODEL)),
    )
    with pytest.raises(ValueError, match="error_vec"):
        _ir(layers=(bad,))
