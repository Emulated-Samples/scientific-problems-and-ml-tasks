"""Forward primitives against the real model's own numbers.

``linearize`` captures a snapshot; each ``apply_*`` primitive must reproduce
the corresponding piece of the model's forward pass on that snapshot's own
prompt. These are the "the IR means what it says" tests: hand math for the
norm formulas, model-delta equality for attention/MLP, end-to-end logits for
``forward``.
"""

import dataclasses

import pytest
import torch

import lat
from lat.linearize import linearize
from lat.ir import NormSnapshot
from lat.primitives import apply_attention, apply_mlp, apply_norm, forward

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def faithful_ir(faithful_attribution_model, sample_prompt):
    return linearize(faithful_attribution_model, sample_prompt, show_progress=False)


@pytest.fixture(scope="module")
def model_cache(faithful_attribution_model, faithful_ir):
    transformer = faithful_attribution_model.transformer
    with torch.no_grad():
        logits, cache = transformer.run_with_cache(faithful_ir.input_tokens.unsqueeze(0))
    return logits.squeeze(0), {k: v.squeeze(0) for k, v in cache.items()}


# ─── apply_norm vs the model's norm sites ───────────────────────────────────


def test_apply_norm_pre_attn_matches_model_ln1(
    faithful_ir, model_cache, faithful_attribution_model
):
    # TL's hook_normalized fires before gamma/beta; apply the affine params
    # to get the norm's actual output.
    _, cache = model_cache
    blocks = faithful_attribution_model.transformer.blocks
    for i, layer in enumerate(faithful_ir.layers):
        out = apply_norm(layer.attention.pre_norm, cache[f"blocks.{i}.hook_resid_pre"])
        ln = blocks[i].ln1
        expected = cache[f"blocks.{i}.ln1.hook_normalized"] * ln.w + ln.b
        assert torch.allclose(out, expected, atol=1e-4)


def test_apply_norm_pre_mlp_matches_model_ln2(faithful_ir, model_cache, faithful_attribution_model):
    _, cache = model_cache
    blocks = faithful_attribution_model.transformer.blocks
    for i, layer in enumerate(faithful_ir.layers):
        out = apply_norm(layer.mlp.pre_norm, cache[f"blocks.{i}.hook_resid_mid"])
        ln = blocks[i].ln2
        expected = cache[f"blocks.{i}.ln2.hook_normalized"] * ln.w + ln.b
        assert torch.allclose(out, expected, atol=1e-4)


def test_apply_norm_final_matches_model_ln_final(
    faithful_ir, model_cache, faithful_attribution_model
):
    _, cache = model_cache
    last = faithful_ir.n_layers - 1
    ln = faithful_attribution_model.transformer.ln_final
    out = apply_norm(faithful_ir.final_norm, cache[f"blocks.{last}.hook_resid_post"])
    expected = cache["ln_final.hook_normalized"] * ln.w + ln.b
    assert torch.allclose(out, expected, atol=1e-4)


def test_apply_norm_layernorm_centers_and_scales():
    snapshot = NormSnapshot(
        scale=torch.tensor([2.0]),
        gamma=torch.full((4,), 3.0),
        mean=torch.tensor([0.0]),
        beta=torch.full((4,), 0.5),
    )
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # centered = [-1.5, -0.5, 0.5, 1.5]; / 2 * 3 + 0.5
    expected = torch.tensor([[-1.75, -0.25, 1.25, 2.75]])
    assert torch.allclose(apply_norm(snapshot, x), expected)


def test_apply_norm_rmsnorm_skips_mean_and_beta():
    snapshot = NormSnapshot(
        scale=torch.tensor([2.0]),
        gamma=torch.full((4,), 3.0),
        mean=None,
        beta=None,
    )
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = x / 2.0 * 3.0
    assert torch.allclose(apply_norm(snapshot, x), expected)


# ─── apply_attention / apply_mlp vs model deltas ────────────────────────────


def test_apply_attention_matches_model_delta(faithful_ir, model_cache):
    _, cache = model_cache
    for i, layer in enumerate(faithful_ir.layers):
        delta = cache[f"blocks.{i}.hook_resid_mid"] - cache[f"blocks.{i}.hook_resid_pre"]
        out = apply_attention(layer.attention, cache[f"blocks.{i}.hook_resid_pre"])
        assert torch.allclose(out, delta, atol=1e-3)


def test_apply_attention_zero_pattern_yields_only_b_O(faithful_ir):
    attn = faithful_ir.layers[0].attention
    zeroed = dataclasses.replace(attn, pattern=torch.zeros_like(attn.pattern))
    out = apply_attention(zeroed, torch.randn(faithful_ir.n_pos, faithful_ir.d_model))
    expected = attn.b_O.expand_as(out) if attn.b_O is not None else torch.zeros_like(out)
    assert torch.allclose(out, expected, atol=1e-6)


def test_apply_mlp_matches_model_delta(faithful_ir, model_cache):
    _, cache = model_cache
    for i, layer in enumerate(faithful_ir.layers):
        delta = cache[f"blocks.{i}.hook_resid_post"] - cache[f"blocks.{i}.hook_resid_mid"]
        out = apply_mlp(layer.mlp, cache[f"blocks.{i}.hook_resid_mid"])
        assert torch.allclose(out, delta, atol=1e-3)


def test_apply_mlp_recomputed_activations_match_snapshot(faithful_ir, model_cache):
    """On the snapshot input, recomputing activations from W_enc reproduces
    the captured activation values."""
    _, cache = model_cache
    for i, layer in enumerate(faithful_ir.layers):
        mlp = layer.mlp
        if mlp.n_active == 0:
            continue
        normed = apply_norm(mlp.pre_norm, cache[f"blocks.{i}.hook_resid_mid"])
        normed_at = normed.index_select(0, mlp.active_positions)
        recomputed = (normed_at * mlp.W_enc_active).sum(dim=-1) + mlp.b_enc_active
        assert torch.allclose(recomputed, mlp.active_activations, atol=1e-4)


def test_apply_mlp_zero_W_enc_uses_only_biases(faithful_ir):
    mlp = faithful_ir.layers[0].mlp
    zeroed = dataclasses.replace(
        mlp,
        W_enc_active=torch.zeros_like(mlp.W_enc_active),
        b_enc_active=torch.zeros_like(mlp.b_enc_active),
        W_dec_active=torch.zeros_like(mlp.W_dec_active),
    )
    x = torch.randn(faithful_ir.n_pos, faithful_ir.d_model)
    out = apply_mlp(zeroed, x)
    assert torch.allclose(out, (mlp.b_dec + mlp.error_vec).expand_as(out), atol=1e-6)


# ─── forward ────────────────────────────────────────────────────────────────


def test_forward_matches_model_logits(faithful_ir, model_cache):
    logits, _ = model_cache
    assert torch.allclose(forward(faithful_ir), logits, atol=2e-2)


def test_forward_is_linear_in_resid_0_perturbation(faithful_ir):
    """The linearized model's response to a resid_0 perturbation is linear:
    f(x0 + a·d) - f(x0) == a · (f(x0 + d) - f(x0))."""
    base = forward(faithful_ir)
    d = torch.randn_like(faithful_ir.resid_0) * 0.1

    def perturbed(scale: float) -> torch.Tensor:
        ir2 = dataclasses.replace(faithful_ir, resid_0=faithful_ir.resid_0 + scale * d)
        return forward(ir2)

    resp_1 = perturbed(1.0) - base
    resp_3 = perturbed(3.0) - base
    assert torch.allclose(resp_3, 3.0 * resp_1, atol=1e-2)
