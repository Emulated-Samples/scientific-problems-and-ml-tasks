"""Correctness for ``forward_inject_batch``, via transpose parity with backward.

The gate: forward-mode columns must equal reverse-mode rows, transposed. If
``forward_edges[source, target] == backward_row(target)[source]`` over every
node-pair class (feature/token/error source x feature/logit target), then
forward is the exact adjoint of ``backward_inject_batch`` — which is itself
validated against ``circuit_tracer`` in ``test_oracle.py``. Forward correctness
is thus transitive, without a second oracle.

The remaining tests cover the local contracts (shape, batched == serial,
linearity in the write vector) the way ``test_adjoint.py`` does for backward.
"""

import pytest
import torch

from lat.attribution._targets import _flatten_active_features, _resolve_targets
from lat.attribution.forward_inject import (
    TOKEN_SOURCE_LAYER,
    ForwardEdgeColumns,
    forward_inject_batch,
)
from lat.linearize import linearize
from lat.primitives import backward_inject_batch

# fp32 gelu-2l; forward and backward group their matmuls differently, so allow a
# little reduction-order drift. Real transpose bugs are order-1, far outside this.
_ATOL = 1e-4
_RTOL = 1e-3


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    torch.testing.assert_close(got, want, atol=_ATOL, rtol=_RTOL)


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


def _feature_source_writes(ir) -> torch.Tensor:
    """Per-active-feature write vector ``a_f * W_dec[f]``, in flat layer order.

    Built by the same layer walk (and same skip-empty-layers rule) as
    ``_flatten_active_features`` so it lines up with ``flat.activations``.
    """
    chunks = [layer.mlp.W_dec_active for layer in ir.layers if layer.mlp.n_active > 0]
    W_dec_flat = torch.cat(chunks, dim=0)
    flat = _flatten_active_features(ir)
    return flat.activations.unsqueeze(-1) * W_dec_flat


def _error_source_batch(ir):
    """(layers, positions, vectors) for every ``(layer, position)`` error node,
    indexed ``l * n_pos + p`` — the layout ``BatchedAttributionRow.errors`` uses
    once flattened."""
    n_pos = ir.n_pos
    layers = torch.arange(ir.n_layers).repeat_interleave(n_pos)
    positions = torch.arange(n_pos).repeat(ir.n_layers)
    vectors = torch.cat([layer.mlp.error_vec for layer in ir.layers], dim=0)
    return layers, positions, vectors


# ─── the transpose-parity gate ──────────────────────────────────────────────


def test_feature_sources_match_backward_transpose(ir, resolved):
    """feature -> feature and feature -> logit edges equal backward rows, transposed."""
    flat = _flatten_active_features(ir)

    fwd = forward_inject_batch(
        ir,
        inject_layers=flat.layers,
        inject_positions=flat.positions,
        inject_vectors=_feature_source_writes(ir),
        resolved=resolved,
    )
    bwd_feat = backward_inject_batch(
        ir,
        inject_layers=flat.layers,
        inject_positions=flat.positions,
        inject_vectors=flat.encoder_vecs,
    )
    bwd_logit = backward_inject_batch(
        ir,
        inject_layers=torch.full((resolved.logit_vectors.shape[0],), ir.n_layers),
        inject_positions=resolved.logit_positions,
        inject_vectors=resolved.logit_vectors,
    )

    # fwd.feature_edges[src, tgt] == bwd_feat.features[tgt, src]
    _close(fwd.feature_edges, bwd_feat.features.T)
    # fwd.logit_edges[src, j]     == bwd_logit.features[j, src]
    _close(fwd.logit_edges, bwd_logit.features.T)


def test_token_sources_match_backward_transpose(ir, resolved):
    """token -> feature and token -> logit edges equal backward rows, transposed."""
    n_pos = ir.n_pos
    flat = _flatten_active_features(ir)

    fwd = forward_inject_batch(
        ir,
        inject_layers=torch.full((n_pos,), TOKEN_SOURCE_LAYER),
        inject_positions=torch.arange(n_pos),
        inject_vectors=ir.embed_vecs,
        resolved=resolved,
    )
    bwd_feat = backward_inject_batch(
        ir,
        inject_layers=flat.layers,
        inject_positions=flat.positions,
        inject_vectors=flat.encoder_vecs,
    )
    bwd_logit = backward_inject_batch(
        ir,
        inject_layers=torch.full((resolved.logit_vectors.shape[0],), ir.n_layers),
        inject_positions=resolved.logit_positions,
        inject_vectors=resolved.logit_vectors,
    )

    # fwd.feature_edges[p, tgt] == bwd_feat.tokens[tgt, p]
    _close(fwd.feature_edges, bwd_feat.tokens.T)
    _close(fwd.logit_edges, bwd_logit.tokens.T)


def test_error_sources_match_backward_transpose(ir, resolved):
    """error -> feature and error -> logit edges equal backward rows, transposed."""
    n_layers, n_pos = ir.n_layers, ir.n_pos
    flat = _flatten_active_features(ir)
    err_layers, err_positions, err_vectors = _error_source_batch(ir)

    fwd = forward_inject_batch(
        ir,
        inject_layers=err_layers,
        inject_positions=err_positions,
        inject_vectors=err_vectors,
        resolved=resolved,
    )
    bwd_feat = backward_inject_batch(
        ir,
        inject_layers=flat.layers,
        inject_positions=flat.positions,
        inject_vectors=flat.encoder_vecs,
    )
    bwd_logit = backward_inject_batch(
        ir,
        inject_layers=torch.full((resolved.logit_vectors.shape[0],), ir.n_layers),
        inject_positions=resolved.logit_positions,
        inject_vectors=resolved.logit_vectors,
    )

    # fwd rows are indexed l*n_pos + p; reshape to [n_layers, n_pos, .] to line up
    # with backward's [target, n_layers, n_pos] errors block.
    fwd_feat = fwd.feature_edges.reshape(n_layers, n_pos, flat.layers.shape[0])
    fwd_logit = fwd.logit_edges.reshape(n_layers, n_pos, resolved.logit_vectors.shape[0])
    torch.testing.assert_close(
        fwd_feat, bwd_feat.errors.permute(1, 2, 0), atol=_ATOL, rtol=_RTOL
    )
    torch.testing.assert_close(
        fwd_logit, bwd_logit.errors.permute(1, 2, 0), atol=_ATOL, rtol=_RTOL
    )


def test_causal_suffix_high_position_sources_match_backward(ir, resolved):
    """Feature sources confined to late positions exercise the causal-suffix
    path (``pos_start > 0``); their edges must still equal the backward
    transpose. Also checks the host-side ``min_inject_position`` hint agrees."""
    flat = _flatten_active_features(ir)
    writes = _feature_source_writes(ir)

    p_thresh = int(flat.positions.max().item()) - 1
    assert p_thresh > 0, "fixture too short to exercise a non-trivial suffix"
    sel = (flat.positions >= p_thresh).nonzero(as_tuple=False).squeeze(-1)
    min_pos = int(flat.positions[sel].min().item())
    assert min_pos > 0

    fwd = forward_inject_batch(
        ir,
        inject_layers=flat.layers[sel],
        inject_positions=flat.positions[sel],
        inject_vectors=writes[sel],
        resolved=resolved,
        min_inject_position=min_pos,  # host-side hint; must match the derived path
    )
    fwd_derived = forward_inject_batch(
        ir,
        inject_layers=flat.layers[sel],
        inject_positions=flat.positions[sel],
        inject_vectors=writes[sel],
        resolved=resolved,
    )
    assert torch.equal(fwd.feature_edges, fwd_derived.feature_edges)
    assert torch.equal(fwd.logit_edges, fwd_derived.logit_edges)

    bwd_feat = backward_inject_batch(
        ir, inject_layers=flat.layers, inject_positions=flat.positions,
        inject_vectors=flat.encoder_vecs,
    )
    bwd_logit = backward_inject_batch(
        ir, inject_layers=torch.full((resolved.logit_vectors.shape[0],), ir.n_layers),
        inject_positions=resolved.logit_positions, inject_vectors=resolved.logit_vectors,
    )
    # fwd row i (source flat idx sel[i]) vs backward column sel[i].
    _close(fwd.feature_edges, bwd_feat.features[:, sel].T)
    _close(fwd.logit_edges, bwd_logit.features[:, sel].T)


# ─── local contracts ────────────────────────────────────────────────────────


def test_forward_inject_batch_shape(ir, resolved):
    batch_size = 3
    out = forward_inject_batch(
        ir,
        inject_layers=torch.zeros(batch_size, dtype=torch.long),
        inject_positions=torch.zeros(batch_size, dtype=torch.long),
        inject_vectors=torch.zeros(batch_size, ir.d_model),
        resolved=resolved,
    )
    assert isinstance(out, ForwardEdgeColumns)
    assert out.feature_edges.shape == (batch_size, ir.total_active_features)
    assert out.logit_edges.shape == (batch_size, resolved.logit_vectors.shape[0])


def test_forward_inject_batch_rejects_mismatched_shapes(ir, resolved):
    with pytest.raises(ValueError, match="same shape"):
        forward_inject_batch(
            ir,
            inject_layers=torch.zeros(3, dtype=torch.long),
            inject_positions=torch.zeros(4, dtype=torch.long),
            inject_vectors=torch.zeros(3, ir.d_model),
            resolved=resolved,
        )


def test_forward_inject_batch_rejects_wrong_vector_shape(ir, resolved):
    with pytest.raises(ValueError, match="inject_vectors"):
        forward_inject_batch(
            ir,
            inject_layers=torch.zeros(3, dtype=torch.long),
            inject_positions=torch.zeros(3, dtype=torch.long),
            inject_vectors=torch.zeros(3, ir.d_model + 1),
            resolved=resolved,
        )


def test_forward_inject_batch_rejects_non_1d_layers(ir, resolved):
    with pytest.raises(ValueError, match="must be 1-D"):
        forward_inject_batch(
            ir,
            inject_layers=torch.zeros(3, 2, dtype=torch.long),
            inject_positions=torch.zeros(3, 2, dtype=torch.long),
            inject_vectors=torch.zeros(3, ir.d_model),
            resolved=resolved,
        )


def test_forward_inject_batch_batched_equals_serial(ir, resolved):
    """One batched call equals stacked single-source calls: catches cross-row
    contamination and adaptive-depth (per-layer ``start_layer``) bugs."""
    g = torch.Generator().manual_seed(3)
    inject_layers = torch.tensor([TOKEN_SOURCE_LAYER, 0, 1, 0], dtype=torch.long)
    inject_positions = torch.tensor([0, 1, ir.n_pos - 1, 2], dtype=torch.long)
    inject_vectors = torch.randn(4, ir.d_model, generator=g, dtype=ir.resid_0.dtype)

    batched = forward_inject_batch(
        ir,
        inject_layers=inject_layers,
        inject_positions=inject_positions,
        inject_vectors=inject_vectors,
        resolved=resolved,
    )
    for b in range(4):
        serial = forward_inject_batch(
            ir,
            inject_layers=inject_layers[b : b + 1],
            inject_positions=inject_positions[b : b + 1],
            inject_vectors=inject_vectors[b : b + 1],
            resolved=resolved,
        )
        _close(batched.feature_edges[b], serial.feature_edges[0])
        _close(batched.logit_edges[b], serial.logit_edges[0])


def test_forward_inject_batch_host_layer_list_matches_derived(ir, resolved):
    """Passing ``inject_layer_list`` host-side is a pure sync-avoidance
    optimization: identical output to deriving it from the device tensor."""
    g = torch.Generator().manual_seed(5)
    inject_layers = torch.tensor([TOKEN_SOURCE_LAYER, 0, 1], dtype=torch.long)
    inject_positions = torch.tensor([0, 1, 2], dtype=torch.long)
    inject_vectors = torch.randn(3, ir.d_model, generator=g, dtype=ir.resid_0.dtype)

    derived = forward_inject_batch(
        ir, inject_layers=inject_layers, inject_positions=inject_positions,
        inject_vectors=inject_vectors, resolved=resolved,
    )
    explicit = forward_inject_batch(
        ir, inject_layers=inject_layers, inject_positions=inject_positions,
        inject_vectors=inject_vectors, resolved=resolved,
        inject_layer_list=[TOKEN_SOURCE_LAYER, 0, 1],
    )
    assert torch.equal(derived.feature_edges, explicit.feature_edges)
    assert torch.equal(derived.logit_edges, explicit.logit_edges)


def test_forward_inject_batch_is_linear_in_write_vector(ir, resolved):
    """``forward_inject_batch(alpha * v) == alpha * forward_inject_batch(v)``."""
    g = torch.Generator().manual_seed(7)
    write = torch.randn(ir.d_model, generator=g, dtype=ir.resid_0.dtype)
    alpha = 2.5
    layers = torch.tensor([0], dtype=torch.long)
    positions = torch.tensor([1], dtype=torch.long)

    base = forward_inject_batch(
        ir, inject_layers=layers, inject_positions=positions,
        inject_vectors=write.unsqueeze(0), resolved=resolved,
    )
    scaled = forward_inject_batch(
        ir, inject_layers=layers, inject_positions=positions,
        inject_vectors=(alpha * write).unsqueeze(0), resolved=resolved,
    )
    _close(scaled.feature_edges, alpha * base.feature_edges)
    _close(scaled.logit_edges, alpha * base.logit_edges)
