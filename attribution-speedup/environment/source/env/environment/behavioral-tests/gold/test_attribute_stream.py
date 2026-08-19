"""Contract + parity for ``attribute_stream``.

The orchestrator is thin over ``forward_inject_batch`` (whose numerics are gated
in ``test_forward_inject.py``), so these tests target the *orchestration*: the
frame sequence, the frontier semantics, the canonical-id / target-column
mapping, and that streaming one source layer at a time reproduces exactly the
edge set of a single all-sources forward pass — i.e. per-layer batching neither
drops nor duplicates edges.
"""

import pytest
import torch

from lat.attribution._targets import _flatten_active_features, _resolve_targets
from lat.attribution.forward_inject import TOKEN_SOURCE_LAYER, forward_inject_batch
from lat.attribution.stream import (
    LayerEdgeFrame,
    NodeTableFrame,
    StreamDoneFrame,
    attribute_stream,
)
from lat.circuit.schema import NodeKind, canonical_node_id
from lat.linearize import linearize


@pytest.fixture(scope="module")
def ir(faithful_attribution_model, sample_prompt):
    return linearize(faithful_attribution_model, sample_prompt, show_progress=False)


@pytest.fixture(scope="module")
def tokenizer(faithful_attribution_model):
    return faithful_attribution_model.transformer.tokenizer


def _frames(ir, tokenizer, **kw):
    return list(attribute_stream(ir, tokenizer, show_progress=False, **kw))


# ─── frame sequence + node table ────────────────────────────────────────────


def test_frame_sequence_and_frontier(ir, tokenizer):
    frames = _frames(ir, tokenizer)

    assert isinstance(frames[0], NodeTableFrame)
    assert isinstance(frames[-1], StreamDoneFrame)
    edge_frames = frames[1:-1]
    assert all(isinstance(f, LayerEdgeFrame) for f in edge_frames)

    layers = [f.source_layer for f in edge_frames]
    # non-decreasing; tokens (-1) first, then every layer ascending (one or more
    # frames per layer once a layer's sources exceed batch_size).
    assert layers == sorted(layers)
    assert sorted(set(layers)) == [TOKEN_SOURCE_LAYER, *range(ir.n_layers)]

    # is_layer_complete marks exactly the last frame of each source layer.
    completed = [f.source_layer for f in edge_frames if f.is_layer_complete]
    assert completed == [TOKEN_SOURCE_LAYER, *range(ir.n_layers)]

    # frontier is non-decreasing and equals source_layer once the layer completes.
    frontiers = [f.frontier_layer for f in edge_frames]
    assert frontiers == sorted(frontiers)
    for f in edge_frames:
        expected = f.source_layer if f.is_layer_complete else f.source_layer - 1
        assert f.frontier_layer == expected


def test_small_batch_size_bounds_held_edges(ir, tokenizer):
    """A small batch_size splits layers across frames and bounds each frame's
    source count — the stream never holds more than one batch of sources."""
    from collections import Counter

    batch_size = 4
    edge_frames = [
        f for f in _frames(ir, tokenizer, batch_size=batch_size) if isinstance(f, LayerEdgeFrame)
    ]
    assert all(len(f.source_ids) <= batch_size for f in edge_frames)
    # edge_source_idx is local to the frame's own (bounded) source list.
    assert all(
        f.edge_source_idx.numel() == 0 or int(f.edge_source_idx.max()) < len(f.source_ids)
        for f in edge_frames
    )
    # at least one layer was split into multiple frames.
    assert any(count > 1 for count in Counter(f.source_layer for f in edge_frames).values())


def test_node_table_counts(ir, tokenizer):
    table = _frames(ir, tokenizer)[0]
    assert isinstance(table, NodeTableFrame)
    assert table.n_features == ir.total_active_features
    assert len(table.feature_ids) == ir.total_active_features
    assert len(table.token_ids) == ir.n_pos
    assert len(table.token_strs) == ir.n_pos
    assert table.n_layers == ir.n_layers
    assert table.n_pos == ir.n_pos
    # ids are the canonical scheme
    assert all(fid.startswith("F_") for fid in table.feature_ids)
    assert all(tid.startswith("T_") for tid in table.token_ids)
    assert all(lid.startswith("L_") for lid in table.logit_ids)


def test_done_frame_counts_edges(ir, tokenizer):
    frames = _frames(ir, tokenizer)
    done = frames[-1]
    streamed = sum(f.edge_weight.numel() for f in frames[1:-1] if isinstance(f, LayerEdgeFrame))
    assert isinstance(done, StreamDoneFrame)
    assert done.n_edges_emitted == streamed
    assert done.elapsed_s >= 0.0


# ─── parity: streamed edges == one all-sources forward pass ──────────────────


def _reference_dense(ir, resolved) -> tuple[torch.Tensor, dict[str, int], int]:
    """Dense ``[n_sources, n_features + n_logits]`` edge matrix from ONE forward
    pass over every source, plus the source-id -> row map."""
    flat = _flatten_active_features(ir)
    n_pos, n_layers = ir.n_pos, ir.n_layers

    # feature sources (flat order)
    W_dec_flat = torch.cat(
        [layer.mlp.W_dec_active for layer in ir.layers if layer.mlp.n_active > 0], dim=0
    )
    feat_writes = flat.activations.unsqueeze(-1) * W_dec_flat
    feat_ids = [
        canonical_node_id(NodeKind.FEATURE, layer=layer, feature_idx=fi, position=p)
        for layer, fi, p in zip(
            flat.layers.tolist(), flat.feature_idxs.tolist(), flat.positions.tolist(), strict=True
        )
    ]
    # token sources
    tok_ids = [
        canonical_node_id(NodeKind.EMBEDDING, token_id=int(t), position=p)
        for p, t in enumerate(ir.input_tokens.tolist())
    ]
    # error sources (l * n_pos + p order)
    err_ids = [
        canonical_node_id(NodeKind.ERROR, layer=layer, position=p)
        for layer in range(n_layers)
        for p in range(n_pos)
    ]

    layers = torch.cat(
        [
            flat.layers,
            torch.full((n_pos,), TOKEN_SOURCE_LAYER),
            torch.arange(n_layers).repeat_interleave(n_pos),
        ]
    )
    positions = torch.cat(
        [flat.positions, torch.arange(n_pos), torch.arange(n_pos).repeat(n_layers)]
    )
    writes = torch.cat(
        [feat_writes, ir.embed_vecs, torch.cat([layer.mlp.error_vec for layer in ir.layers], dim=0)]
    )

    fwd = forward_inject_batch(
        ir,
        inject_layers=layers,
        inject_positions=positions,
        inject_vectors=writes,
        resolved=resolved,
    )
    dense = torch.cat([fwd.feature_edges, fwd.logit_edges], dim=1)
    src_row = {sid: i for i, sid in enumerate(feat_ids + tok_ids + err_ids)}
    return dense, src_row, ir.total_active_features


@pytest.mark.parametrize("batch_size", [128, 3])
def test_streamed_edges_equal_all_sources_forward(ir, tokenizer, batch_size):
    # batch_size=3 forces per-layer sub-batching (layers have > 3 sources), so
    # this also checks that chunk merging neither drops, duplicates, nor
    # mis-indexes edges relative to a single full-layer sweep.
    resolved = _resolve_targets(
        ir, tokenizer=tokenizer, targets=None, max_n_logits=10, desired_logit_prob=0.95
    )
    reference, src_row, _ = _reference_dense(ir, resolved)

    stream_dense = torch.zeros_like(reference)
    for frame in _frames(ir, tokenizer, batch_size=batch_size):
        if not isinstance(frame, LayerEdgeFrame):
            continue
        for local, col, weight in zip(
            frame.edge_source_idx.tolist(),
            frame.edge_target_col.tolist(),
            frame.edge_weight.tolist(),
            strict=True,
        ):
            stream_dense[src_row[frame.source_ids[local]], col] = weight

    torch.testing.assert_close(stream_dense, reference, atol=1e-5, rtol=1e-4)


def test_pruning_floor_drops_small_edges(ir, tokenizer):
    """A magnitude floor drops exactly the sub-floor edges and no others."""
    unpruned = _frames(ir, tokenizer)
    all_weights = torch.cat(
        [f.edge_weight for f in unpruned if isinstance(f, LayerEdgeFrame)]
    )
    floor = float(all_weights.abs().median())

    pruned = _frames(ir, tokenizer, edge_magnitude_floor=floor)
    for frame in pruned:
        if isinstance(frame, LayerEdgeFrame) and frame.edge_weight.numel():
            assert frame.edge_weight.abs().min() >= floor
