import type { ProblemGrading } from "./types.js";
import { baseSpeedup } from "./p-base-speedup.js";
import { baseRanker } from "./p-base-ranker.js";
import { baseBugs } from "./p-base-bugs.js";
import { basePortParity } from "./p-base-port-parity.js";
import { baseBackward } from "./p-base-backward.js";
import { baseStream } from "./p-base-stream.js";

export type { ProblemGrading } from "./types.js";

export const PROBLEM_GRADING: Record<string, ProblemGrading> = {
  "base-speedup": baseSpeedup,
  "base-speedup-hint": baseSpeedup,
  "base-ranker": baseRanker,
  "base-ranker-hint": baseRanker,
  "base-bugs": baseBugs,
  "base-bugs-hint": baseBugs,
  "base-bugs-hint-easy": baseBugs,
  "base-port-parity": basePortParity,
  "base-stream": baseStream,
  "base-backward": baseBackward,
};
