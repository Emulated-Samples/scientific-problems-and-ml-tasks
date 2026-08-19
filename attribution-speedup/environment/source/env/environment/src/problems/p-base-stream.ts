import type { ProblemGrading } from "./types.js";
import { BASE_WORKSPACE_DESELECTS } from "./_families.js";

export const baseStream: ProblemGrading = {
  suites: ["core", "base", "problems/base-stream"],
  f2p: {
    "problems/base-stream/test_v2_contract.py::test_v2_stream_reconstructs_reference_graph": 1.5,
    "problems/base-stream/test_v2_contract.py::test_v2_node_edges_match_reference": 1,
    "problems/base-stream/test_v2_contract.py::test_v2_never_materializes_the_matrix": 2,
    "problems/base-stream/test_v2_contract.py::test_v2_out_query_survives_the_huge_fixture": 1,
    "problems/base-stream/test_v2_contract.py::test_v2_first_frame_is_online": 2.5,
    "problems/base-stream/test_v2_contract.py::test_v2_out_query_is_incremental": 2,
    "problems/base-stream/test_v2_fwd_perf.py::test_v2_query_cost_respects_temporal_causality": 3,
    "problems/base-stream/test_v2_fwd_perf.py::test_v2_query_cost_respects_layer_causality": 3,
  },
  continuous: {
    "problems/base-stream/test_v2_contract.py::test_v2_first_frame_is_online": {
      metric: "v2_frame_speedup",
      target: 2.3,
    },
    "problems/base-stream/test_v2_contract.py::test_v2_out_query_is_incremental": {
      metric: "v2_out_query_speedup",
      target: 200,
    },
    "problems/base-stream/test_v2_fwd_perf.py::test_v2_query_cost_respects_temporal_causality": {
      metric: "v2_fwd_temporal_asymmetry",
      target: 2.6,
    },
    "problems/base-stream/test_v2_fwd_perf.py::test_v2_query_cost_respects_layer_causality": {
      metric: "v2_fwd_layer_asymmetry",
      target: 8.0,
    },
  },
  deselect: BASE_WORKSPACE_DESELECTS,
  pristineBaseline: true,
  baseFamily: true,
  metricMultiplier: {
    metric: "v2_bwd_retention",
    floor: 0.5,
    ceiling: 0.8,
  },
};
