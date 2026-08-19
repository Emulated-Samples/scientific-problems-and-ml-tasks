
export interface ContinuousSpec {
  metric: string;
  target: number;
}

export interface SurgicalGuard {
  maxFiles: number;
  maxLines: number;
  penalty: number;
}

export interface ProblemGrading {
  stateBranch?: string;

  f2p: Record<string, number>;

  continuous?: Record<string, ContinuousSpec>;

  suites: string[];

  deselect?: string[];

  surgicalGuard?: SurgicalGuard;

  metricMultiplier?: { metric: string; floor: number; ceiling: number };

  pristineBaseline?: boolean;

  // Stage the gold reference (base-gold) and calibrate continuous speedup
  // targets against what gold achieves live on the grading host — makes the
  // timing metrics hardware-agnostic. Requires pristineBaseline + continuous.
  goldBaseline?: boolean;

  baseFamily?: boolean;
}
