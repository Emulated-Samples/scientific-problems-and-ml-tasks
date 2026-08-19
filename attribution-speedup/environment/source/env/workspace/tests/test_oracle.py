"""Dual-run parity against circuit-tracer, and the Target API battery.

The end-to-end correctness gate: on the same (model, transcoders, prompt),
``lat.attribute_linear`` and ``circuit_tracer.attribute`` must produce the
same graph (edge correlation + active/selected feature agreement via
``_compare.compare_graphs``). Plus the ``Target`` resolution contract:
auto-pick, explicit ids, multi-position, and validation errors.
"""

import pytest
import torch

import lat
from lat.linearize import linearize
from _compare import compare_graphs  # ``tests/`` is on sys.path via the conftest mechanism

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _oracle_graph(model, prompt, **kwargs):
    pytest.importorskip("circuit_tracer")
    from circuit_tracer import attribute as oracle_attribute

    return oracle_attribute(prompt=prompt, model=model.transformer, verbose=False, **kwargs)


def test_attribute_linear_matches_oracle_gelu2l(tiny_attribution_model, sample_prompt):
    graph = lat.attribute_linear(tiny_attribution_model, sample_prompt, show_progress=False)
    oracle = _oracle_graph(tiny_attribution_model, sample_prompt)
    report = compare_graphs(oracle, graph)
    assert report.passes(edge_corr_min=0.99, feature_jaccard_min=0.99), report


@pytest.mark.parametrize("max_feature_nodes", [None, 8])
def test_attribute_linear_matches_oracle_with_cap(
    tiny_attribution_model, sample_prompt, max_feature_nodes
):
    """Parity holds both at no-cap and with the greedy ranker active."""
    graph = lat.attribute_linear(
        tiny_attribution_model,
        sample_prompt,
        show_progress=False,
        max_feature_nodes=max_feature_nodes,
    )
    oracle = _oracle_graph(
        tiny_attribution_model, sample_prompt, max_feature_nodes=max_feature_nodes
    )
    report = compare_graphs(oracle, graph)
    assert report.passes(edge_corr_min=0.99, feature_jaccard_min=0.99), report
    assert report.selected_feature_jaccard >= 0.95, report


def test_attribute_linear_returns_attribution_graph(tiny_attribution_model, sample_prompt):
    graph = lat.attribute_linear(tiny_attribution_model, sample_prompt, show_progress=False)
    assert isinstance(graph, lat.AttributionGraph)
    assert graph.adjacency_matrix.device.type == "cpu"
    assert len(graph.logit_nodes) >= 1


# ─── Target resolution ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ir(tiny_attribution_model, sample_prompt):
    return linearize(tiny_attribution_model, sample_prompt, show_progress=False)


@pytest.fixture(scope="module")
def tokenizer(tiny_attribution_model):
    return tiny_attribution_model.transformer.tokenizer


def test_default_target_is_last_position(ir, tokenizer):
    graph = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    assert all(node.position == ir.n_pos - 1 for node in graph.logit_nodes)


def test_target_at_explicit_position_auto_picks(ir, tokenizer):
    pos = ir.n_pos // 2
    graph = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, targets=[lat.Target(position=pos)]
    )
    assert len(graph.logit_nodes) >= 1
    assert all(node.position == pos for node in graph.logit_nodes)


def test_target_counterfactual_explicit_token_ids(ir, tokenizer):
    ids = [11, 42]
    graph = lat.attribute_from_linearized(
        ir,
        tokenizer,
        show_progress=False,
        targets=[lat.Target(position=ir.n_pos - 1, token_ids=ids)],
    )
    assert [node.vocab_idx for node in graph.logit_nodes] == ids


def test_multi_position_attribution(ir, tokenizer):
    positions = [1, ir.n_pos - 1]
    graph = lat.attribute_from_linearized(
        ir,
        tokenizer,
        show_progress=False,
        targets=[lat.Target(position=p) for p in positions],
    )
    assert {node.position for node in graph.logit_nodes} == set(positions)


def test_target_position_out_of_range_raises(ir, tokenizer):
    with pytest.raises(ValueError, match="out of range"):
        lat.attribute_from_linearized(
            ir, tokenizer, show_progress=False, targets=[lat.Target(position=ir.n_pos)]
        )


def test_target_token_id_out_of_range_raises(ir, tokenizer):
    d_vocab = ir.logits.shape[-1]
    with pytest.raises(ValueError, match="out-of-range"):
        lat.attribute_from_linearized(
            ir,
            tokenizer,
            show_progress=False,
            targets=[lat.Target(position=0, token_ids=[d_vocab])],
        )


def test_empty_targets_sequence_raises(ir, tokenizer):
    with pytest.raises(ValueError, match="non-empty"):
        lat.attribute_from_linearized(ir, tokenizer, show_progress=False, targets=[])


def test_empty_token_ids_raises(ir, tokenizer):
    with pytest.raises(ValueError, match="empty"):
        lat.attribute_from_linearized(
            ir, tokenizer, show_progress=False, targets=[lat.Target(position=0, token_ids=[])]
        )


def test_logit_node_probability_matches_softmax(ir, tokenizer):
    graph = lat.attribute_from_linearized(ir, tokenizer, show_progress=False)
    probs = torch.softmax(ir.logits[ir.n_pos - 1], dim=-1)
    for node in graph.logit_nodes:
        assert node.probability == pytest.approx(float(probs[node.vocab_idx]), rel=1e-4)


# ─── circuit-tracer viewer export ───────────────────────────────────────────


def test_to_circuit_tracer_graph_roundtrips(tiny_attribution_model, sample_prompt):
    pytest.importorskip("circuit_tracer")
    graph = lat.attribute_linear(tiny_attribution_model, sample_prompt, show_progress=False)
    ct_graph = graph.to_circuit_tracer_graph(cfg=tiny_attribution_model.transformer.cfg)
    assert torch.equal(ct_graph.adjacency_matrix, graph.adjacency_matrix)
    assert len(ct_graph.logit_targets) == len(graph.logit_nodes)


def test_to_circuit_tracer_graph_warns_on_multi_position(
    tiny_attribution_model, sample_prompt, caplog
):
    pytest.importorskip("circuit_tracer")
    ir = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    tokenizer = tiny_attribution_model.transformer.tokenizer
    graph = lat.attribute_from_linearized(
        ir, tokenizer, show_progress=False, targets=[lat.Target(position=1)]
    )
    with caplog.at_level("WARNING", logger="lat.attribution.graph"):
        graph.to_circuit_tracer_graph(cfg=tiny_attribution_model.transformer.cfg)
    assert any("non-last position" in message for message in caplog.messages)
