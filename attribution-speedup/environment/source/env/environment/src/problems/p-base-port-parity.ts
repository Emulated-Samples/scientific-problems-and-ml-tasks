import type { ProblemGrading } from "./types.js";
import { BASE_WORKSPACE_DESELECTS } from "./_families.js";

export const basePortParity: ProblemGrading = {
  suites: ["core", "base", "problems/base-port-parity"],
  f2p: {
    "problems/base-port-parity/test_port_parity_anchors.py::test_port_matches_reference_on_mixed_causality_models": 2,
    "problems/base-port-parity/test_port_parity_anchors.py::test_port_matches_reference_error_attribution": 2,
    "problems/base-port-parity/test_port_parity_anchors.py::test_port_matches_reference_on_feature_batches": 2,
    "problems/base-port-parity/test_port_parity_anchors.py::test_port_matches_reference_in_jacobian_mode": 2,
  },
  deselect: BASE_WORKSPACE_DESELECTS,
  pristineBaseline: true,
  baseFamily: true,
  metricMultiplier: { metric: "port_speed_retention", floor: 0.75, ceiling: 0.95 },
};
