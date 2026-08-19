import type { ProblemGrading } from "./types.js";
import { BASE_WORKSPACE_DESELECTS } from "./_families.js";

export const baseSpeedup: ProblemGrading = {
  stateBranch: "problem/base-speedup",
  suites: ["core", "base", "problems/base-speedup"],
  f2p: {
    "problems/base-speedup/test_base_attribution_speedup.py::test_attribution_sweep_speedup": 4,
    "problems/base-speedup/test_base_causal_perf.py::test_base_backward_cost_respects_temporal_causality": 2,
    "problems/base-speedup/test_base_causal_perf.py::test_base_backward_cost_respects_layer_causality": 2,
  },
  continuous: {
    "problems/base-speedup/test_base_attribution_speedup.py::test_attribution_sweep_speedup": {
      metric: "attribution_speedup",
      target: 3.0,
    },
    "problems/base-speedup/test_base_causal_perf.py::test_base_backward_cost_respects_temporal_causality": {
      metric: "temporal_asymmetry",
      target: 2.2,
    },
    "problems/base-speedup/test_base_causal_perf.py::test_base_backward_cost_respects_layer_causality": {
      metric: "layer_asymmetry",
      target: 6.0,
    },
  },
  deselect: BASE_WORKSPACE_DESELECTS,
  pristineBaseline: true,
  goldBaseline: true,
  baseFamily: true,
};
