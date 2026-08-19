"""Edge pruning: ``edge_magnitude_floor`` and ``max_edges_per_row``.

Both apply per attribution row before it lands in the edge matrix; the floor
zeroes small-magnitude entries, the cap keeps the strongest K, and they
compose (cap over what survived the floor).
"""

import pytest
import torch

import lat
from lat.linearize import linearize

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def ir(tiny_attribution_model, sample_prompt):
    return linearize(tiny_attribution_model, sample_prompt, show_progress=False)


@pytest.fixture(scope="module")
def tokenizer(tiny_attribution_model):
    return tiny_attribution_model.transformer.tokenizer


def test_defaults_match_no_kwargs(ir, tokenizer):
    a = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    b = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, edge_magnitude_floor=0.0, max_edges_per_row=None
    )
    assert torch.equal(a.adjacency_matrix, b.adjacency_matrix)


def test_floor_zeros_small_edges(ir, tokenizer):
    floor = 1e-3
    graph = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, edge_magnitude_floor=floor
    )
    nonzero = graph.adjacency_matrix[graph.adjacency_matrix != 0]
    assert bool((nonzero.abs() >= floor).all())


def test_floor_only_removes_edges(ir, tokenizer):
    base = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    pruned = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, edge_magnitude_floor=1e-3
    )
    surviving = pruned.adjacency_matrix != 0
    assert torch.allclose(
        pruned.adjacency_matrix[surviving], base.adjacency_matrix[surviving], atol=1e-6
    )
    assert int(surviving.sum()) <= int((base.adjacency_matrix != 0).sum())


def test_floor_negative_raises(ir, tokenizer):
    with pytest.raises(ValueError, match="edge_magnitude_floor"):
        lat.attribute_from_linearized(
            ir, tokenizer, show_progress=False, edge_magnitude_floor=-1.0
        )


def test_max_edges_per_row_bounds_nonzero_count(ir, tokenizer):
    cap = 5
    graph = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, max_edges_per_row=cap
    )
    row_nonzeros = (graph.adjacency_matrix != 0).sum(dim=-1)
    assert bool((row_nonzeros <= cap).all())


def test_max_edges_per_row_keeps_the_strongest(ir, tokenizer):
    cap = 5
    base = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    capped = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, max_edges_per_row=cap
    )
    # Row 0 of each attributed row keeps a subset of its strongest entries.
    for row in range(base.adjacency_matrix.shape[0]):
        kept = capped.adjacency_matrix[row] != 0
        if not kept.any():
            continue
        kept_min = base.adjacency_matrix[row][kept].abs().min()
        dropped = (~kept) & (base.adjacency_matrix[row] != 0)
        if dropped.any():
            assert kept_min >= base.adjacency_matrix[row][dropped].abs().max() - 1e-6


def test_max_edges_per_row_larger_than_n_cols_is_noop(ir, tokenizer):
    a = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    b = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, max_edges_per_row=10**9
    )
    assert torch.equal(a.adjacency_matrix, b.adjacency_matrix)


def test_max_edges_per_row_zero_raises(ir, tokenizer):
    with pytest.raises(ValueError, match="max_edges_per_row"):
        lat.attribute_from_linearized(ir, tokenizer, show_progress=False, max_edges_per_row=0)


def test_max_edges_per_row_negative_raises(ir, tokenizer):
    with pytest.raises(ValueError, match="max_edges_per_row"):
        lat.attribute_from_linearized(ir, tokenizer, show_progress=False, max_edges_per_row=-3)


def test_floor_and_cap_compose(ir, tokenizer):
    floor, cap = 1e-4, 8
    graph = lat.attribute_from_linearized(
        ir,
        tokenizer,
        show_progress=False,
        edge_magnitude_floor=floor,
        max_edges_per_row=cap,
    )
    nonzero = graph.adjacency_matrix[graph.adjacency_matrix != 0]
    assert bool((nonzero.abs() >= floor).all())
    assert bool(((graph.adjacency_matrix != 0).sum(dim=-1) <= cap).all())
