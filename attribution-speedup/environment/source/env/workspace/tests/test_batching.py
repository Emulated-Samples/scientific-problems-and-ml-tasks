"""Batching semantics of the attribution loop.

Batch size is a memory knob, never a semantics knob: at no-cap, any batch
size must produce the identical graph. Cap-mode structure (row/column
bookkeeping, selected-feature accounting) is checked here; cap-mode *parity*
against the oracle lives in test_oracle.py.
"""

import pytest
import torch

import lat
from lat.linearize import linearize

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def ir(tiny_attribution_model, sample_prompt):
    return linearize(tiny_attribution_model, sample_prompt, show_progress=False)


def test_no_cap_batch_size_does_not_change_the_graph(ir, tiny_attribution_model):
    tokenizer = tiny_attribution_model.transformer.tokenizer
    g_default = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    g_small = lat.attribute_from_linearized(ir, tokenizer, show_progress=False, batch_size=7)
    assert torch.equal(g_default.selected_features, g_small.selected_features)
    assert torch.allclose(g_default.adjacency_matrix, g_small.adjacency_matrix, atol=1e-5)


def test_cap_mode_selects_requested_count(ir, tiny_attribution_model):
    tokenizer = tiny_attribution_model.transformer.tokenizer
    cap = min(10, ir.total_active_features)
    g = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, max_feature_nodes=cap, batch_size=4
    )
    assert g.selected_features.shape[0] == cap
    # Selected indices are valid rows of active_features, without duplicates.
    assert g.selected_features.unique().numel() == cap
    assert int(g.selected_features.max()) < g.active_features.shape[0]


def test_cap_above_total_active_is_clamped(ir, tiny_attribution_model):
    tokenizer = tiny_attribution_model.transformer.tokenizer
    g = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, max_feature_nodes=10**9
    )
    assert g.selected_features.shape[0] == ir.total_active_features


def test_cap_mode_adjacency_is_square_with_selected_rows(ir, tiny_attribution_model):
    tokenizer = tiny_attribution_model.transformer.tokenizer
    cap = min(10, ir.total_active_features)
    g = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, max_feature_nodes=cap, batch_size=4
    )
    n_nodes = cap + ir.n_layers * ir.n_pos + ir.n_pos + len(g.logit_nodes)
    assert g.adjacency_matrix.shape == (n_nodes, n_nodes)
    # Error and token rows are leaf sources: all-zero rows.
    error_token_rows = g.adjacency_matrix[cap : cap + ir.n_layers * ir.n_pos + ir.n_pos]
    assert torch.equal(error_token_rows, torch.zeros_like(error_token_rows))
