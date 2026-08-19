import * as path from "path";
import type { ProblemGrading } from "../problems/types.js";
import { PRISTINE_BASE_DIR } from "./provisioning.js";

export function pytestSelectionArgs(grading: ProblemGrading): string[] {
  return [
    ...grading.suites,
    ...(grading.deselect ?? []).map((t) => `--deselect ${t}`),
  ];
}

export function gradeEnv(
  grading: ProblemGrading,
  workspace: string,
  metricsFile: string,
  goldBaseDir?: string
): Record<string, string> {
  return {
    PYTHONPATH: path.join(workspace, "src"),
    CUDA_VISIBLE_DEVICES: "",
    LAT_BENCH_METRICS_FILE: metricsFile,
    ...(grading.pristineBaseline
      ? { LAT_BENCH_PRISTINE_SRC: path.join(PRISTINE_BASE_DIR, "src") }
      : {}),
    ...(goldBaseDir
      ? { LAT_BENCH_GOLD_SRC: path.join(goldBaseDir, "src") }
      : {}),
    ...(grading.baseFamily ? { LAT_BENCH_BASE_FAMILY: "1" } : {}),
  };
}
