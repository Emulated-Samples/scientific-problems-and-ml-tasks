import type { ProblemGrading } from "./types.js";
import { BASE_WORKSPACE_DESELECTS } from "./_families.js";

export const baseBugs: ProblemGrading = {
  suites: ["core", "base", "problems/base-bugs"],
  f2p: {
    "base/test_base_qk_truth.py::test_qk_backward_matches_autograd_vjp_with_qk_norm": 0.9091,
    "base/test_base_qk_truth.py::test_qk_backward_is_adjoint_of_forward_with_qk_norm": 0.5455,
    "core/test_artifact_query_truth.py::test_max_edges_per_row_keeps_exactly_k": 1.5,
    "core/test_artifact_query_truth.py::test_edges_from_orders_by_magnitude_with_mixed_signs": 0.3636,
    "core/test_artifact_query_truth.py::test_edges_into_orders_by_magnitude_with_mixed_signs": 0.3636,
    "core/test_artifact_query_truth.py::test_edges_top_k_takes_magnitude_strongest": 0.3636,
    "core/test_circuit_artifact.py::test_edges_into_strongest_first": 0.1818,
    "core/test_circuit_artifact.py::test_edges_from_and_top_k": 0.1818,
    "core/test_oracle.py::test_attribute_linear_matches_oracle_with_cap[8]": 0.8333,
    "core/test_to_circuit.py::test_node_influence_seeded_from_logits": 0.3333,
    "core/test_batching.py::test_cap_mode_still_matches_oracle": 0.3333,
    "core/test_artifact_query_truth.py::test_neighbors_includes_incoming_sources": 1.5,
    "core/test_artifact_query_truth.py::test_find_is_case_insensitive_over_ids_and_labels": 1.5,
    "problems/base-bugs/test_base_bugs_coverage.py::test_agent_suite_catches[bug1]": 0.5455,
    "problems/base-bugs/test_base_bugs_coverage.py::test_agent_suite_catches[bug2]": 0.5,
    "problems/base-bugs/test_base_bugs_coverage.py::test_agent_suite_catches[bug3]": 0.5455,
    "problems/base-bugs/test_base_bugs_coverage.py::test_agent_suite_catches[bug4]": 0.5,
    "problems/base-bugs/test_base_bugs_coverage.py::test_agent_suite_catches[bug5]": 0.5,
    "problems/base-bugs/test_base_bugs_coverage.py::test_agent_suite_catches[bug6]": 0.5,
  },
  deselect: BASE_WORKSPACE_DESELECTS,
  pristineBaseline: true,
  baseFamily: true,
};
