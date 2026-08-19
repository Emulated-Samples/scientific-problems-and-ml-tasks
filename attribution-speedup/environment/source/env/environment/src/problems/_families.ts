
export const BASE_WORKSPACE_DESELECTS = [
  "core/test_adjoint.py::test_attention_flat_weight_path_matches_einsum",
  "core/test_adjoint.py::test_apply_attention_linear_equals_apply_attention_minus_bias",
  "core/test_adjoint.py::test_backward_attention_causal_prefix_matches_full_zero_padded",
  "core/test_adjoint.py::test_backward_attention_prefix_requires_causal_attention",
  "core/test_adjoint.py::test_backward_inject_batch_host_metadata_kwargs_match_derived",
  "core/test_batching.py::test_cap_mode_feature_batches_are_layer_sorted",
  "core/test_forward.py::test_linearize_position_groups_cover_n_active",
  "core/test_forward.py::test_linearize_position_groups_match_actual_positions",
  "core/test_pruning.py::test_sparse_requires_explicit_device",
  "core/test_pruning.py::test_sparse_requires_pruning",
  "core/test_pruning.py::test_sparse_cpu_matches_dense_with_floor",
  "core/test_pruning.py::test_sparse_cpu_matches_dense_with_cap",
  "core/test_pruning.py::test_sparse_gpu_matches_dense_with_cap",
  "core/test_pruning.py::test_sparse_output_is_dense_tensor",
  "core/test_pruning.py::test_sparse_with_max_feature_nodes_cap_invokes_ranker",
  "core/test_pruning.py::test_sparse_ranker_matches_dense_ranker",
  "core/test_activation_cache.py::test_from_ir_projects_active_features",
  "core/test_activation_cache.py::test_from_ir_round_trips_through_safetensors",
  "core/test_activation_cache.py::test_from_ir_without_tokenizer_leaves_token_strs_empty",
  "core/test_circuit_artifact.py::test_attribution_names_hint_to_extra_without_torch",
];

