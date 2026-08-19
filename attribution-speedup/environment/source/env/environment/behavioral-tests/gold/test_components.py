"""Correctness for component (supernode) attribution.

The gate (``test_component_edges_equal_aggregated...``): computing the component
adjacency the efficient way — inject each cell's *summed* write, aggregate the
readout columns by cell — must equal block-aggregating the full *per-source*
feature+error edges. That equality IS the supernode thesis: supernode edges =
summed feature edges, exactly. It holds by linearity because every feature in a
cell shares ``(layer, position)`` so the frozen path Jacobian factors out.

The full per-source edges come from ``forward_inject_batch`` itself (validated
against ``circuit_tracer`` transitively via ``test_forward_inject`` /
``test_oracle``), so this test needs no separate oracle.
"""

import pytest
import torch

from lat.attribution._targets import _flatten_active_features, _resolve_targets
from lat.attribution.components import (
    _flat_decoder_rows,
    component_attribute,
    component_writes,
)
from lat.attribution.forward_inject import forward_inject_batch
from lat.linearize import linearize
from lat.primitives import apply_norm_linear

# fp32 gelu-2l. The two paths sum in different orders (sum-then-inject vs.
# inject-then-sum), so allow reduction-order drift; a real linearity bug is
# order-1, far outside this.
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


def _feature_source_writes(ir) -> torch.Tensor:
    """Per-active-feature write ``a_f * W_dec[f]`` in flat layer order."""
    chunks = [layer.mlp.W_dec_active for layer in ir.layers if layer.mlp.n_active > 0]
    w_dec_flat = (
        torch.cat(chunks, dim=0)
        if chunks
        else torch.empty(0, ir.d_model, dtype=ir.resid_0.dtype)
    )
    flat = _flatten_active_features(ir)
    return flat.activations.unsqueeze(-1) * w_dec_flat


def _error_source_batch(ir):
    """(layers, positions, vectors) for every ``(layer, position)`` error node,
    laid out layer-major so source index ``l * n_pos + p`` == component index."""
    n_pos = ir.n_pos
    layers = torch.arange(ir.n_layers).repeat_interleave(n_pos)
    positions = torch.arange(n_pos).repeat(ir.n_layers)
    vectors = torch.cat([layer.mlp.error_vec for layer in ir.layers], dim=0)
    return layers, positions, vectors


def test_component_writes_match_features_plus_error(ir):
    """w(l, p) == error_vec(l, p) + sum of the cell's feature decoder-writes."""
    src = component_writes(ir)
    n_pos = ir.n_pos

    expected = torch.zeros_like(src.vectors)
    for layer_idx, layer in enumerate(ir.layers):
        expected[layer_idx * n_pos : (layer_idx + 1) * n_pos] += layer.mlp.error_vec
    flat = _flatten_active_features(ir)
    cell = (flat.layers * n_pos + flat.positions).long()
    expected.index_add_(0, cell, _feature_source_writes(ir))

    torch.testing.assert_close(src.vectors, expected, atol=_ATOL, rtol=_RTOL)
    # Grid bookkeeping: component index == layer * n_pos + position.
    assert torch.equal(src.layers, torch.arange(ir.n_layers).repeat_interleave(n_pos))
    assert torch.equal(src.positions, torch.arange(n_pos).repeat(ir.n_layers))


def test_component_edges_equal_aggregated_feature_and_error_edges(ir, resolved):
    """The thesis: component edges == feature+error edges block-summed per cell."""
    n_pos = ir.n_pos
    n_components = ir.n_layers * n_pos
    n_feat = ir.total_active_features

    flat = _flatten_active_features(ir)
    cell_of_feature = (flat.layers * n_pos + flat.positions).long()

    # Full per-source edges (features as sources, then errors as sources).
    ff = forward_inject_batch(
        ir,
        inject_layers=flat.layers,
        inject_positions=flat.positions,
        inject_vectors=_feature_source_writes(ir),
        resolved=resolved,
    )
    err_layers, err_positions, err_vectors = _error_source_batch(ir)
    fe = forward_inject_batch(
        ir,
        inject_layers=err_layers,
        inject_positions=err_positions,
        inject_vectors=err_vectors,
        resolved=resolved,
    )
    error_cell = (err_layers * n_pos + err_positions).long()

    # Aggregate SOURCE rows by cell (features + errors), then TARGET columns.
    comp_rows = torch.zeros(n_components, n_feat, dtype=ff.feature_edges.dtype)
    comp_rows.index_add_(0, cell_of_feature, ff.feature_edges)
    comp_rows.index_add_(0, error_cell, fe.feature_edges)
    expected_edges = torch.zeros(n_components, n_components, dtype=comp_rows.dtype)
    if n_feat > 0:
        expected_edges.index_add_(1, cell_of_feature, comp_rows)

    n_logits = resolved.logit_vectors.shape[0]
    expected_logits = torch.zeros(n_components, n_logits, dtype=ff.logit_edges.dtype)
    expected_logits.index_add_(0, cell_of_feature, ff.logit_edges)
    expected_logits.index_add_(0, error_cell, fe.logit_edges)

    got = component_attribute(ir, resolved=resolved)

    assert got.component_edges.shape == (n_components, n_components)
    assert got.logit_edges.shape == (n_components, n_logits)
    torch.testing.assert_close(got.component_edges, expected_edges, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(got.logit_edges, expected_logits, atol=_ATOL, rtol=_RTOL)


def test_same_layer_component_edges_are_zero(ir, resolved):
    """Strict layer ordering: a cell never edges to another cell in its own layer."""
    got = component_attribute(ir, resolved=resolved)
    same_layer = got.layers[:, None] == got.layers[None, :]
    assert torch.count_nonzero(got.component_edges[same_layer]) == 0


def test_chunking_is_exact(ir, resolved):
    """Chunked source injection (batch_size=1) equals one big batch, edge for edge."""
    whole = component_attribute(ir, resolved=resolved, batch_size=10_000)
    chunked = component_attribute(ir, resolved=resolved, batch_size=1)
    torch.testing.assert_close(
        chunked.component_edges, whole.component_edges, atol=_ATOL, rtol=_RTOL
    )
    torch.testing.assert_close(chunked.logit_edges, whole.logit_edges, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(chunked.abs_edges, whole.abs_edges, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(chunked.proj_edges, whole.proj_edges, atol=_ATOL, rtol=_RTOL)


def _component_feature_columns(ir, resolved) -> torch.Tensor:
    """Reference per-target-feature columns for the component sources: inject
    every cell's summed write in one batch, no aggregation. ``[N_SN, N_feat]``."""
    src = component_writes(ir)
    fwd = forward_inject_batch(
        ir,
        inject_layers=src.layers,
        inject_positions=src.positions,
        inject_vectors=src.vectors,
        resolved=resolved,
    )
    return fwd.feature_edges


def test_abs_channel_is_target_side_magnitude(ir, resolved):
    """abs channel == the same readout columns, |.| before the per-cell sum —
    and it dominates the signed sum elementwise (triangle inequality)."""
    n_pos = ir.n_pos
    n_components = ir.n_layers * n_pos
    flat = _flatten_active_features(ir)
    cell_of_feature = (flat.layers * n_pos + flat.positions).long()

    columns = _component_feature_columns(ir, resolved)
    expected = torch.zeros(n_components, n_components, dtype=columns.dtype)
    expected.index_add_(1, cell_of_feature, columns.abs())

    got = component_attribute(ir, resolved=resolved)
    torch.testing.assert_close(got.abs_edges, expected, atol=_ATOL, rtol=_RTOL)
    assert bool((got.abs_edges >= got.component_edges.abs() - _ATOL).all())


def test_proj_channel_is_write_direction_readout(ir, resolved):
    """proj channel == columns weighted by ``d_f . w_hat_cell(f)`` then cell-summed:
    the transcoder-mediated ``delta mlp_out(t)`` projected on t's write direction."""
    n_pos = ir.n_pos
    n_components = ir.n_layers * n_pos
    src = component_writes(ir)
    flat = _flatten_active_features(ir)
    cell_of_feature = (flat.layers * n_pos + flat.positions).long()

    write_dirs = src.vectors / src.vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    weights = (_flat_decoder_rows(ir) * write_dirs[cell_of_feature]).sum(dim=-1)

    columns = _component_feature_columns(ir, resolved)
    expected = torch.zeros(n_components, n_components, dtype=columns.dtype)
    expected.index_add_(1, cell_of_feature, columns * weights)

    got = component_attribute(ir, resolved=resolved)
    torch.testing.assert_close(got.proj_edges, expected, atol=_ATOL, rtol=_RTOL)


def test_identity_channel_matches_naive_readout(ir, resolved):
    """identity channel == E_t^T pre_norm_linear(w_s) per same-position pair,
    recomputed here with the forward norm (the module uses the adjoint)."""
    n_pos = ir.n_pos
    src = component_writes(ir)
    flat = _flatten_active_features(ir)
    cell_of_feature = (flat.layers * n_pos + flat.positions).long()

    encoders = torch.zeros_like(src.vectors)
    if flat.encoder_vecs.shape[0] > 0:
        encoders.index_add_(0, cell_of_feature, flat.encoder_vecs)

    got = component_attribute(ir, resolved=resolved)
    expected = torch.zeros_like(got.identity_edges)
    for source_cell in range(src.layers.shape[0]):
        source_layer = int(src.layers[source_cell])
        position = int(src.positions[source_cell])
        # Place w_s at its position; read through each later layer's pre-norm.
        delta = torch.zeros(n_pos, ir.d_model, dtype=src.vectors.dtype)
        delta[position] = src.vectors[source_cell]
        for target_layer in range(source_layer + 1, ir.n_layers):
            normed = apply_norm_linear(ir.layers[target_layer].mlp.pre_norm, delta)
            target_cell = target_layer * n_pos + position
            expected[source_cell, target_cell] = normed[position] @ encoders[target_cell]

    torch.testing.assert_close(got.identity_edges, expected, atol=_ATOL, rtol=_RTOL)
    # Cross-position pairs carry no identity path.
    cross = src.positions[:, None] != src.positions[None, :]
    assert torch.count_nonzero(got.identity_edges[cross]) == 0


def test_member_counts_and_encoder_norms(ir, resolved):
    """Display-normalization stats: per-cell feature count and ||sum e_f||."""
    n_pos = ir.n_pos
    flat = _flatten_active_features(ir)
    cell_of_feature = (flat.layers * n_pos + flat.positions).long()

    got = component_attribute(ir, resolved=resolved)
    expected_counts = torch.bincount(cell_of_feature, minlength=got.layers.shape[0])
    assert torch.equal(got.member_counts, expected_counts)

    encoders = torch.zeros(got.layers.shape[0], ir.d_model, dtype=flat.encoder_vecs.dtype)
    if flat.encoder_vecs.shape[0] > 0:
        encoders.index_add_(0, cell_of_feature, flat.encoder_vecs)
    torch.testing.assert_close(
        got.encoder_norms, encoders.norm(dim=-1), atol=_ATOL, rtol=_RTOL
    )
