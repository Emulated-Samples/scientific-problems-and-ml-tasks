"""Correctness for total (no-cut) logit influence.

Two gates:

- ``cut_features=True`` must reproduce the DIRECT map bit-for-bit (mod
  reduction order): the walk (seeding, attention adjoint, write-site
  recording) is thereby validated against ``component_attribute`` /
  ``forward_inject_batch``, which are themselves validated against
  ``circuit_tracer``.
- The no-cut result must equal a naive reference FORWARD of the same no-cut
  linearized model (inject each cell's write, propagate through attention AND
  every transcoder's frozen-open response, read the logit) — adjoint ==
  forward, per cell, per logit.
"""

import pytest
import torch

from lat.attribution._targets import _resolve_targets
from lat.attribution.components import component_attribute, component_writes
from lat.attribution.forward_inject import TOKEN_SOURCE_LAYER, forward_inject_batch
from lat.attribution.influence import total_logit_influence
from lat.linearize import linearize
from lat.primitives import apply_attention_linear, apply_norm_linear

_ATOL = 1e-4
_RTOL = 1e-3


@pytest.fixture(scope="module")
def ir(faithful_attribution_model, sample_prompt):
    return linearize(faithful_attribution_model, sample_prompt, show_progress=False)


@pytest.fixture(scope="module")
def resolved(ir, faithful_attribution_model):
    return _resolve_targets(
        ir,
        tokenizer=faithful_attribution_model.transformer.tokenizer,
        targets=None,
        max_n_logits=10,
        desired_logit_prob=0.95,
    )


def test_cut_features_equals_direct_map(ir, resolved):
    """Severing the responses must give exactly the direct component->logit map."""
    total = total_logit_influence(ir, resolved=resolved, cut_features=True)
    direct = component_attribute(ir, resolved=resolved)
    torch.testing.assert_close(
        total.component_influence, direct.logit_edges, atol=_ATOL, rtol=_RTOL
    )

    n_pos = ir.n_pos
    emb = forward_inject_batch(
        ir,
        inject_layers=torch.full((n_pos,), TOKEN_SOURCE_LAYER, dtype=torch.long),
        inject_positions=torch.arange(n_pos),
        inject_vectors=ir.embed_vecs,
        resolved=resolved,
    )
    torch.testing.assert_close(
        total.embedding_influence, emb.logit_edges, atol=_ATOL, rtol=_RTOL
    )


def _no_cut_forward_readout(ir, resolved, delta: torch.Tensor, start_layer: int) -> torch.Tensor:
    """Reference forward of the no-cut linearized model.

    ``delta`` is seeded at resid_post of ``start_layer`` (``-1`` = resid_0).
    Returns the per-logit readout ``[n_logits]``.
    """
    for layer in ir.layers[start_layer + 1 :]:
        delta = delta + apply_attention_linear(layer.attention, delta)
        normed = apply_norm_linear(layer.mlp.pre_norm, delta)
        response = torch.zeros_like(delta)
        for pos, start, length in layer.mlp.position_groups:
            W_enc = layer.mlp.W_enc_active.narrow(0, start, length)
            W_dec = layer.mlp.W_dec_active.narrow(0, start, length)
            response[pos] = (normed[pos] @ W_enc.T) @ W_dec
        delta = delta + response

    normed_final = apply_norm_linear(ir.final_norm, delta)
    at_logits = normed_final[resolved.logit_positions]  # [n_logits, d_model]
    return (at_logits * resolved.logit_vectors).sum(dim=-1)


def test_no_cut_matches_reference_forward(ir, resolved):
    """Backward totals == the naive no-cut forward, for every cell and logit."""
    src = component_writes(ir)
    total = total_logit_influence(ir, resolved=resolved)

    n_pos = ir.n_pos
    for cell in range(src.layers.shape[0]):
        layer = int(src.layers[cell])
        position = int(src.positions[cell])
        delta = torch.zeros(n_pos, ir.d_model, dtype=src.vectors.dtype)
        delta[position] = src.vectors[cell]
        expected = _no_cut_forward_readout(ir, resolved, delta, layer)
        torch.testing.assert_close(
            total.component_influence[cell], expected, atol=_ATOL, rtol=_RTOL
        )


def test_no_cut_embeddings_match_reference_forward(ir, resolved):
    """Embedding totals == the no-cut forward seeded at resid_0."""
    total = total_logit_influence(ir, resolved=resolved)
    n_pos = ir.n_pos
    for position in range(n_pos):
        delta = torch.zeros(n_pos, ir.d_model, dtype=ir.embed_vecs.dtype)
        delta[position] = ir.embed_vecs[position]
        expected = _no_cut_forward_readout(ir, resolved, delta, -1)
        torch.testing.assert_close(
            total.embedding_influence[position], expected, atol=_ATOL, rtol=_RTOL
        )


def test_indirect_influence_is_nonzero_somewhere(ir, resolved):
    """The responses must actually change something (total != direct), or the
    no-cut pass is silently a no-op on this fixture."""
    total = total_logit_influence(ir, resolved=resolved)
    direct = total_logit_influence(ir, resolved=resolved, cut_features=True)
    assert not torch.allclose(
        total.component_influence, direct.component_influence, atol=1e-6
    )


# ─── True-MLP-Jacobian response (mlp_response="raw") ─────────────────────────


@pytest.fixture(scope="module")
def raw_ir(faithful_attribution_model, sample_prompt):
    return linearize(
        faithful_attribution_model,
        sample_prompt,
        show_progress=False,
        capture_mlp_jacobian=True,
    )


def test_raw_requires_captured_jacobian(ir, resolved):
    with pytest.raises(ValueError, match="capture_mlp_jacobian"):
        total_logit_influence(ir, resolved=resolved, mlp_response="raw")


def test_captured_jacobian_matches_autograd_jvp(raw_ir, faithful_attribution_model):
    """The factored capture == autograd JVP of the model's OWN MLP module at
    the captured input, per layer."""
    transformer = faithful_attribution_model.transformer
    tokens = raw_ir.input_tokens.unsqueeze(0)
    _, cache = transformer.run_with_cache(
        tokens, names_filter=lambda name: name.endswith("mlp.hook_in")
    )

    for layer_idx, layer in enumerate(raw_ir.layers):
        jacobian = layer.mlp.jacobian
        assert jacobian is not None
        block_mlp = transformer.blocks[layer_idx].mlp
        inner = getattr(block_mlp, "old_mlp", block_mlp)
        key = next(k for k in cache if k.startswith(f"blocks.{layer_idx}.") and "hook_in" in k)
        x0 = cache[key].squeeze(0)

        tangent = torch.randn_like(x0)
        _, expected = torch.autograd.functional.jvp(inner, x0, tangent)

        hidden = torch.zeros(x0.shape[0], jacobian.branches[0].factor.shape[-1])
        for branch in jacobian.branches:
            hidden = hidden + (tangent @ branch.W_in) * branch.factor
        got = hidden @ jacobian.W_out
        torch.testing.assert_close(got, expected, atol=_ATOL, rtol=_RTOL)


def _raw_forward_readout(ir, resolved, delta: torch.Tensor, start_layer: int) -> torch.Tensor:
    """Reference forward of the no-cut model with the TRUE MLP responses."""
    for layer in ir.layers[start_layer + 1 :]:
        delta = delta + apply_attention_linear(layer.attention, delta)
        normed = apply_norm_linear(layer.mlp.pre_norm, delta)
        jacobian = layer.mlp.jacobian
        assert jacobian is not None
        hidden = torch.zeros(delta.shape[0], jacobian.branches[0].factor.shape[-1])
        for branch in jacobian.branches:
            hidden = hidden + (normed @ branch.W_in) * branch.factor
        delta = delta + hidden @ jacobian.W_out

    normed_final = apply_norm_linear(ir.final_norm, delta)
    at_logits = normed_final[resolved.logit_positions]
    return (at_logits * resolved.logit_vectors).sum(dim=-1)


def test_raw_total_matches_reference_forward(raw_ir, resolved):
    """Backward raw totals == the naive raw-response forward, per cell/logit."""
    src = component_writes(raw_ir)
    total = total_logit_influence(raw_ir, resolved=resolved, mlp_response="raw")

    n_pos = raw_ir.n_pos
    for cell in range(src.layers.shape[0]):
        layer = int(src.layers[cell])
        position = int(src.positions[cell])
        delta = torch.zeros(n_pos, raw_ir.d_model, dtype=src.vectors.dtype)
        delta[position] = src.vectors[cell]
        expected = _raw_forward_readout(raw_ir, resolved, delta, layer)
        torch.testing.assert_close(
            total.component_influence[cell], expected, atol=_ATOL, rtol=_RTOL
        )


def test_raw_differs_from_transcoder_response(raw_ir, resolved):
    """The two response operators must disagree somewhere, or 'raw' is not
    actually reading the true Jacobian on this fixture."""
    raw = total_logit_influence(raw_ir, resolved=resolved, mlp_response="raw")
    transcoder = total_logit_influence(raw_ir, resolved=resolved)
    assert not torch.allclose(
        raw.component_influence, transcoder.component_influence, atol=1e-6
    )
