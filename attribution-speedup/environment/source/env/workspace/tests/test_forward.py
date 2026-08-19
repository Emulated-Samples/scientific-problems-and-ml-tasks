"""``linearize``: IR construction from a real model.

Shape and consistency of the captured IR, prompt-input flexibility, the
faithfulness contract (``forward(ir)`` reproduces the model when nothing is
filtered), and the ``min_activation_threshold`` semantics (filtered features
flow into ``error_vec``; the decomposition is preserved).
"""

import pytest
import torch

import lat
from lat.linearize import linearize

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def ir(tiny_attribution_model, sample_prompt):
    return linearize(tiny_attribution_model, sample_prompt, show_progress=False)


def test_linearize_returns_ir_with_expected_shape(ir, tiny_attribution_model):
    cfg = tiny_attribution_model.transformer.cfg
    assert ir.n_layers == cfg.n_layers
    assert ir.d_model == cfg.d_model
    assert ir.resid_0.shape == (ir.n_pos, ir.d_model)
    assert ir.embed_vecs.shape == (ir.n_pos, ir.d_model)
    assert ir.logits.shape == (ir.n_pos, cfg.d_vocab)
    assert ir.W_unembed.shape == (ir.d_model, cfg.d_vocab)


def test_linearize_active_features_have_d_model_rows(ir):
    for layer in ir.layers:
        assert layer.mlp.W_enc_active.shape == (layer.mlp.n_active, ir.d_model)
        assert layer.mlp.W_dec_active.shape == (layer.mlp.n_active, ir.d_model)


def test_linearize_error_vec_shape_matches_n_pos(ir):
    for layer in ir.layers:
        assert layer.mlp.error_vec.shape == (ir.n_pos, ir.d_model)


def test_linearize_logits_match_model_forward(faithful_attribution_model, sample_prompt):
    ir = linearize(faithful_attribution_model, sample_prompt, show_progress=False)
    with torch.no_grad():
        expected = faithful_attribution_model.transformer(ir.input_tokens.unsqueeze(0)).squeeze(0)
    assert torch.allclose(ir.logits, expected, atol=1e-3)


def test_linearize_accepts_token_tensor_prompt(tiny_attribution_model, sample_prompt):
    ir_str = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    ir_tensor = linearize(
        tiny_attribution_model, ir_str.input_tokens, show_progress=False
    )
    assert torch.equal(ir_str.input_tokens, ir_tensor.input_tokens)
    assert torch.allclose(ir_str.logits, ir_tensor.logits, atol=1e-5)


def test_linearize_accepts_token_list_prompt(tiny_attribution_model, sample_prompt):
    ir_str = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    ir_list = linearize(
        tiny_attribution_model, ir_str.input_tokens.tolist(), show_progress=False
    )
    assert torch.equal(ir_str.input_tokens, ir_list.input_tokens)


def test_linearize_logits_consistent_across_calls(tiny_attribution_model, sample_prompt):
    ir1 = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    ir2 = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    assert torch.equal(ir1.logits, ir2.logits)
    for l1, l2 in zip(ir1.layers, ir2.layers):
        assert torch.equal(l1.mlp.active_positions, l2.mlp.active_positions)
        assert torch.equal(l1.mlp.active_feature_idx, l2.mlp.active_feature_idx)


def test_linearize_faithful_ir_has_no_zeroed_error(faithful_attribution_model, sample_prompt):
    """With zero_positions=None nothing is masked: every position's error_vec
    reflects the true reconstruction gap (position 0 included)."""
    ir = linearize(faithful_attribution_model, sample_prompt, show_progress=False)
    assert any(layer.mlp.error_vec[0].abs().sum() > 0 for layer in ir.layers)


# ─── min_activation_threshold ───────────────────────────────────────────────


def test_linearize_threshold_zero_is_noop(tiny_attribution_model, sample_prompt):
    a = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    b = linearize(
        tiny_attribution_model, sample_prompt, show_progress=False, min_activation_threshold=0.0
    )
    for la, lb in zip(a.layers, b.layers):
        assert torch.equal(la.mlp.active_feature_idx, lb.mlp.active_feature_idx)


def test_linearize_threshold_filters_below_threshold_features(
    tiny_attribution_model, sample_prompt
):
    threshold = 0.5
    ir = linearize(
        tiny_attribution_model,
        sample_prompt,
        show_progress=False,
        min_activation_threshold=threshold,
    )
    for layer in ir.layers:
        if layer.mlp.n_active:
            assert bool((layer.mlp.active_activations >= threshold).all())


def test_linearize_threshold_strictly_reduces_active_features(
    tiny_attribution_model, sample_prompt
):
    full = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    filtered = linearize(
        tiny_attribution_model, sample_prompt, show_progress=False, min_activation_threshold=0.5
    )
    assert filtered.total_active_features < full.total_active_features


def test_linearize_threshold_preserves_mlp_decomposition(
    faithful_attribution_model, sample_prompt
):
    """Filtered features fold into error_vec: reconstruction + error is
    unchanged, so forward(ir) still reproduces the model."""
    ir = linearize(
        faithful_attribution_model,
        sample_prompt,
        show_progress=False,
        min_activation_threshold=0.5,
    )
    with torch.no_grad():
        expected = faithful_attribution_model.transformer(ir.input_tokens.unsqueeze(0)).squeeze(0)
    assert torch.allclose(lat.forward(ir), expected, atol=2e-2)


def test_linearize_threshold_negative_raises(tiny_attribution_model, sample_prompt):
    with pytest.raises(ValueError, match="min_activation_threshold"):
        linearize(
            tiny_attribution_model,
            sample_prompt,
            show_progress=False,
            min_activation_threshold=-0.1,
        )


def test_linearize_invalid_norm_mode_raises(tiny_attribution_model, sample_prompt):
    with pytest.raises(ValueError, match="norm_mode"):
        linearize(
            tiny_attribution_model, sample_prompt, show_progress=False, norm_mode="banana"
        )


def test_linearize_invalid_qk_mode_raises(tiny_attribution_model, sample_prompt):
    with pytest.raises(ValueError, match="qk_mode"):
        linearize(tiny_attribution_model, sample_prompt, show_progress=False, qk_mode="banana")
