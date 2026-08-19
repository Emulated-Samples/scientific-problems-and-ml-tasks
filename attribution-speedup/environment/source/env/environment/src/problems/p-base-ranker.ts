import type { ProblemGrading } from "./types.js";
import { BASE_WORKSPACE_DESELECTS } from "./_families.js";

export const baseRanker: ProblemGrading = {
  suites: ["core", "base", "problems/base-ranker"],
  f2p: {
    "problems/base-ranker/test_base_ranker_perf.py::test_ranker_speedup": 4,
  },
  continuous: {
    "problems/base-ranker/test_base_ranker_perf.py::test_ranker_speedup": {
      metric: "ranker_speedup",
      target: 1.6,
    },
  },
  deselect: BASE_WORKSPACE_DESELECTS,
  pristineBaseline: true,
  baseFamily: true,
};
