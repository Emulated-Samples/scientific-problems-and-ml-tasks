import type { ProblemGrading } from "./types.js";
import { BASE_WORKSPACE_DESELECTS } from "./_families.js";

export const baseBackward: ProblemGrading = {
  suites: ["core", "problems/base-backward"],
  deselect: BASE_WORKSPACE_DESELECTS,
  f2p: {
    "problems/base-backward/test_backward_adjoints.py::test_backward_norm_layernorm_adjoint": 1,
    "problems/base-backward/test_backward_adjoints.py::test_backward_norm_rmsnorm_adjoint": 1,
    "problems/base-backward/test_backward_adjoints.py::test_backward_norm_jacobian_radial_adjoint": 1.5,
    "problems/base-backward/test_backward_adjoints.py::test_backward_attention_mha_adjoint": 1.5,
    "problems/base-backward/test_backward_adjoints.py::test_backward_attention_qk_formation_adjoint": 3,
  },
  pristineBaseline: true,
};
