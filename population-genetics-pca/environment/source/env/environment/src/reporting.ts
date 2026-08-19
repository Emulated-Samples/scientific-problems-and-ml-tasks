import type { TestResult } from "@hyperfocal/env-base";

const BASE_GATES = ["library_scan", "validity"] as const;
const PROBE_GATES = ["coverage", "hwe_norm", "representation_equivalence"] as const;

export const CATEGORY_NAMES = [
  "admixed", "biobank", "biobank_a", "biobank_b", "continental", "crossover", "haploid",
  "high_rank", "ill_conditioned",
  "io_wide", "messy", "observed", "rare_structure", "sample_heavy", "scaling", "spatial_ld",
  "spectral_selection", "subtle", "variable_width", "very_high_rank",
] as const;
type CategoryName = typeof CATEGORY_NAMES[number];

interface MasteryPolicy {
  meanAccuracy: number;
  minimumAccuracy: number;
  representation?: boolean;
}

// Pass/fail is deliberately independent of the bounded continuous reward. A scalar reward threshold
// cannot say *which* capability a submission demonstrated: an exact-but-slow full scan and a fast
// faithful fit can earn the same reward yet prove different skills. The reward stays the continuous
// RL signal; a category "passes" only when its distinct capability -- correct AND as fast as this
// fold permits -- is actually demonstrated. Status is monotone in the category's own accuracy and
// systems quality.
//
// The systems bar is a SINGLE uniform constant, and that is now principled rather than convenient:
// grader.systems_quality is already normalized per fold against what is ACHIEVABLE there. Matching
// the best achievable speed scores 1.0 on every fold; a full scan scores 0.5 wherever the fold
// offers real headroom to win; and as a fold's achievable speedup shrinks toward nothing, a full
// scan rises smoothly toward 1.0, because there was nothing there to forfeit. Fold difficulty
// therefore lives in the metric, not in a thicket of per-category thresholds -- so one bar means the
// same thing everywhere: "you were essentially as fast as this fold allows."
//
// Reachability is measured, not assumed. Through the real grader on the pinned instance type, over
// all 22 deployed datasets (20 categories): the gold reference scores ~0.9978 and masters every
// category; the unoptimized
// expert full scan scores 0.6095, sitting at exactly 0.5 systems quality on every fold with real
// headroom (that is the parity anchor, by construction). 0.75 therefore sits under the reference on
// every fold and above a full scan wherever speed is winnable.
//
// These numbers are post-`c2973c6`, which un-saturated the scale; the figures they replace (naive
// 0.6841, then 0.5757) were each measured under a scoring rule that no longer exists. A mastery bar
// justified by superseded numbers is unfalsifiable later, so they get re-measured, not carried.
//
// Reading that against the metric: the two categories a naive full scan still masters (haploid,
// variable_width) are exactly the folds where a full scan genuinely IS the best available strategy --
// haploid's body is too small for sampling to repay its own setup, and on variable_width the fast
// reference actually LOSES to a plain scan. The bar is correctly not punishing a program for failing
// to achieve a speedup that does not exist. rare_structure is the check on that reasoning: it offers
// only 1.19x, yet a full scan takes 0.648 there and fails -- little headroom is not no headroom, and
// the scale charges the difference in proportion rather than waiving it.
const SYSTEMS_MASTERY_FLOOR = 0.75;

const MASTERY_POLICIES: Record<CategoryName, MasteryPolicy> = {
  admixed: { meanAccuracy: 0.96, minimumAccuracy: 0.92 },
  // Materializing the sample Gram is impossible here; mastering this means choosing marker space.
  // biobank_a/biobank_b are the lower rungs of the same scale ladder (24k/50k samples); the correct
  // marker-space object is identical, so they carry biobank's accuracy policy.
  biobank: { meanAccuracy: 0.97, minimumAccuracy: 0.92 },
  biobank_a: { meanAccuracy: 0.97, minimumAccuracy: 0.92 },
  biobank_b: { meanAccuracy: 0.97, minimumAccuracy: 0.92 },
  continental: { meanAccuracy: 0.97, minimumAccuracy: 0.94 },
  crossover: { meanAccuracy: 0.95, minimumAccuracy: 0.90 },
  haploid: { meanAccuracy: 0.95, minimumAccuracy: 0.90 },
  high_rank: { meanAccuracy: 0.95, minimumAccuracy: 0.90 },
  ill_conditioned: { meanAccuracy: 0.93, minimumAccuracy: 0.87 },
  io_wide: { meanAccuracy: 0.97, minimumAccuracy: 0.92 },
  messy: { meanAccuracy: 0.95, minimumAccuracy: 0.90, representation: true },
  observed: { meanAccuracy: 0.95, minimumAccuracy: 0.90 },
  rare_structure: { meanAccuracy: 0.94, minimumAccuracy: 0.88 },
  sample_heavy: { meanAccuracy: 0.97, minimumAccuracy: 0.92 },
  scaling: { meanAccuracy: 0.97, minimumAccuracy: 0.92 },
  spatial_ld: { meanAccuracy: 0.94, minimumAccuracy: 0.88 },
  spectral_selection: { meanAccuracy: 0.95, minimumAccuracy: 0.90 },
  subtle: { meanAccuracy: 0.94, minimumAccuracy: 0.88 },
  variable_width: { meanAccuracy: 0.95, minimumAccuracy: 0.90, representation: true },
  very_high_rank: { meanAccuracy: 0.93, minimumAccuracy: 0.88 },
};

export interface GateDetail {
  factor: number;
  // Gates carry their own explanation (library_scan: `hits`/`n_hits`; the probes: severities and
  // their Patterson/raw-covariance anchors; validity: `reason`). The grader has always computed
  // these; the report used to drop them, which is how a gate could zero a submission and leave the
  // reader nothing to reason about. Kept open so a new gate's diagnostic surfaces automatically
  // rather than needing this interface widened first.
  [field: string]: unknown;
}

// Fields that describe the VERDICT rather than explain it -- no need to repeat them as a reason.
const GATE_VERDICT_FIELDS = new Set(["factor", "name", "severity"]);

/**
 * Why did this gate fire? Returns the gate's own explanatory fields, or null if it passed.
 *
 * A zero with no stated cause is the failure mode that cost this project the most time: an expert
 * submission scoring 0.000 with a bare diagnostic sent a competent reader to a confident, wrong root
 * cause. `library_scan` was the sharpest instance -- it zeroes a submission on a static AST finding,
 * exits 0, writes NOTHING to stderr, and the report showed only `library_scan: 0`. The gate knew
 * exactly which file and which module it objected to the whole time.
 */
export function gateReason(gate: GateDetail): Record<string, unknown> | null {
  if (!(typeof gate?.factor === "number") || gate.factor >= 1) return null;
  const reason = Object.fromEntries(
    Object.entries(gate).filter(([field, value]) =>
      !GATE_VERDICT_FIELDS.has(field) && value !== null && value !== undefined),
  );
  return Object.keys(reason).length > 0 ? reason : null;
}

export interface DatasetDetail {
  dataset: string;
  category: string;
  k: number;
  weight: number;
  reward: number;
  accuracy: number;
  time_quality: number;
  gate_product: number;
  sub_seconds: number;
  ref_seconds: number;
  fast_ref_seconds: number | null;
  execution_overhead_seconds: number | null;
  gates: Record<string, GateDetail>;
  run: {
    returncode: number;
    stderr_tail: string;
    sampled_primary_invocation_peak_pss_bytes: number | null;
    sampled_primary_invocation_peak_storage_bytes: number | null;
  };
  deadline_exhausted?: boolean;
}

function finiteInRange(value: number, minimum: number, maximum: number): boolean {
  return Number.isFinite(value) && value >= minimum && value <= maximum;
}

export function validDatasetMeasurements(dataset: DatasetDetail): boolean {
  const gateNames = Object.keys(dataset.gates ?? {}).sort();
  const libraryFactor = dataset.gates?.library_scan?.factor;
  const expectedGateNames = (
    libraryFactor === 0 ? [...BASE_GATES] : [...BASE_GATES, ...PROBE_GATES]
  ).sort();
  const unexecutedDeadlineRow = dataset.deadline_exhausted === true;
  const scoredTimingInputsValid = unexecutedDeadlineRow
    ? dataset.fast_ref_seconds === null && dataset.execution_overhead_seconds === null
    : Number.isFinite(dataset.fast_ref_seconds)
      && dataset.fast_ref_seconds! > 0
      && Number.isFinite(dataset.execution_overhead_seconds)
      && dataset.execution_overhead_seconds! >= 0;
  const primaryInvocationTelemetryValid = unexecutedDeadlineRow
    ? dataset.run?.returncode === 124
      && dataset.reward === 0
      && dataset.accuracy === 0
      && dataset.time_quality === 0
      && dataset.gate_product === 0
      && dataset.run.sampled_primary_invocation_peak_pss_bytes === null
      && dataset.run.sampled_primary_invocation_peak_storage_bytes === null
    : Number.isInteger(dataset.run?.sampled_primary_invocation_peak_pss_bytes)
      && dataset.run.sampled_primary_invocation_peak_pss_bytes! >= 0
      && Number.isInteger(dataset.run?.sampled_primary_invocation_peak_storage_bytes)
      && dataset.run.sampled_primary_invocation_peak_storage_bytes! >= 0;
  return typeof dataset.dataset === "string"
    && dataset.dataset.length > 0
    && typeof dataset.category === "string"
    && dataset.category.length > 0
    && Number.isInteger(dataset.k)
    && dataset.k > 0
    && Number.isFinite(dataset.weight)
    && dataset.weight > 0
    && finiteInRange(dataset.reward, 0, 1)
    && finiteInRange(dataset.accuracy, 0, 1)
    && finiteInRange(dataset.time_quality, 0, 1)
    && finiteInRange(dataset.gate_product, 0, 1)
    && Number.isFinite(dataset.sub_seconds)
    && dataset.sub_seconds >= 0
    && Number.isFinite(dataset.ref_seconds)
    && dataset.ref_seconds >= 0
    && scoredTimingInputsValid
    && JSON.stringify(gateNames) === JSON.stringify(expectedGateNames)
    && Object.values(dataset.gates).every((gate) => finiteInRange(gate?.factor, 0, 1))
    && Number.isInteger(dataset.run?.returncode)
    && typeof dataset.run.stderr_tail === "string"
    && (dataset.deadline_exhausted === undefined || dataset.deadline_exhausted === true)
    && primaryInvocationTelemetryValid;
}

function weightedMean(members: DatasetDetail[], value: (dataset: DatasetDetail) => number): number {
  const weight = members.reduce((sum, dataset) => sum + dataset.weight, 0);
  return members.reduce((sum, dataset) => sum + dataset.weight * value(dataset), 0) / weight;
}

function categoryStatus(category: string, members: DatasetDetail[]): TestResult["status"] {
  if (members.length === 0) return "failed";
  if (!CATEGORY_NAMES.includes(category as CategoryName)) return "failed";
  const policy = MASTERY_POLICIES[category as CategoryName];
  const integrityPassed = members.every(
    (dataset) => dataset.run.returncode === 0
      && BASE_GATES.every((name) => dataset.gates[name]?.factor === 1),
  );
  if (!integrityPassed) return "failed";
  const methodPassed = members.every(
    (dataset) => dataset.gates.hwe_norm?.factor >= 0.90
      && dataset.gates.coverage?.factor >= 0.90
      && (!policy.representation || dataset.gates.representation_equivalence?.factor >= 0.90),
  );
  if (!methodPassed) return "failed";
  const accuracyPassed = weightedMean(members, (dataset) => dataset.accuracy)
      >= policy.meanAccuracy
    && members.every((dataset) => dataset.accuracy >= policy.minimumAccuracy);
  if (!accuracyPassed) return "failed";
  // `time_quality` carries grader.systems_quality: speed measured against what this fold actually
  // permits. One uniform bar therefore means the same thing on every fold.
  const systemsPassed = members.every(
    (dataset) => dataset.time_quality >= SYSTEMS_MASTERY_FLOOR,
  );
  return systemsPassed ? "passed" : "failed";
}

function datasetDiagnostic(dataset: DatasetDetail) {
  return {
    dataset: dataset.dataset,
    k: dataset.k,
    accuracy: dataset.accuracy,
    reward: dataset.reward,
    time_quality: dataset.time_quality,
    gate_product: dataset.gate_product,
    gates: Object.fromEntries(
      Object.entries(dataset.gates).sort().map(([name, gate]) => [name, gate.factor]),
    ),
    // Only for gates that actually fired, so a clean grade stays readable.
    gate_reasons: Object.fromEntries(
      Object.entries(dataset.gates).sort()
        .map(([name, gate]) => [name, gateReason(gate)] as const)
        .filter((entry): entry is readonly [string, Record<string, unknown>] => entry[1] !== null),
    ),
    exit_code: dataset.run.returncode,
    sub_seconds: dataset.sub_seconds,
    ref_seconds: dataset.ref_seconds,
    // BOTH anchors, or time_quality is unfalsifiable. The grader scores speed against what is
    // achievable per fold, so `sub` and the full scan alone cannot re-derive it -- the reader also
    // needs the fast anchor and the shared entry cost. Dropping them is why a saturated scale went
    // unnoticed: an Opus run returned time_quality 0.000 on eleven datasets and the artifact could
    // not show that it was only 2-3x off the full scan rather than hopeless. The grader computed
    // these all along.
    fast_ref_seconds: dataset.fast_ref_seconds,
    execution_overhead_seconds: dataset.execution_overhead_seconds,
    sampled_primary_invocation_peak_pss_bytes:
      dataset.run.sampled_primary_invocation_peak_pss_bytes,
    sampled_primary_invocation_peak_storage_bytes:
      dataset.run.sampled_primary_invocation_peak_storage_bytes,
    stderr_tail: dataset.run.stderr_tail,
  };
}

// Exit codes the sealed runner uses to refuse a program on policy grounds, plus the shell's
// "cannot execute". These mean the submission never ran, which is NOT the same thing as running and
// computing the wrong PCA.
const POLICY_REFUSAL_EXITS = new Set([126, 127]);

/**
 * Did EVERY dataset refuse to execute for the same policy reason?
 *
 * A real rollout (run_019f6281) scored a flat 0.000 across all 18 categories because one
 * `multiprocessing.get_context("fork").Array(...)` call tripped the runtime guard: 20/20 datasets
 * exited 126 in ~0.6s, before a byte of VCF was read. The reward was correct -- a program that emits
 * no output has earned nothing, and inferring credit from source would be a reward-hacking surface.
 * The REPORT was not: it looked exactly like "wrote a hopeless PCA", when in fact an implementation
 * whose subspace later measured canonical correlation 1.00000 never got to run.
 *
 * Those two situations must not be indistinguishable. This does not touch the score -- it makes the
 * distinction visible. It deliberately requires the sweep to be TOTAL: a submission that runs on 19
 * folds and refuses on one has a real, dataset-specific defect and should read as such.
 */
function totalExecutionRefusal(perDataset: DatasetDetail[]): string | null {
  if (perDataset.length === 0) return null;
  if (!perDataset.every((dataset) => POLICY_REFUSAL_EXITS.has(dataset.run?.returncode))) return null;
  const codes = [...new Set(perDataset.map((dataset) => dataset.run.returncode))].sort();
  const reasons = [...new Set(
    perDataset.map((dataset) => (dataset.run.stderr_tail ?? "").trim().split("\n").pop()?.trim())
      .filter((line): line is string => Boolean(line)),
  )];
  const detail = reasons.length > 0 ? ` Runner said: ${reasons.join(" | ")}` : "";
  return `EXECUTION CONTRACT: the submission never ran. All ${perDataset.length} invocations exited `
    + `${codes.join("/")} without producing output, so no PCA behaviour was observed and every score `
    + `is zero for that reason alone -- this is NOT evidence about the quality of the algorithm.${detail}`;
}

export function categoryTestResults(
  categories: Record<string, number>,
  perDataset: DatasetDetail[],
): TestResult[] {
  const refusal = totalExecutionRefusal(perDataset);
  return Object.entries(categories).sort().map(([category, score]) => {
    const members = perDataset.filter((dataset) => dataset.category === category);
    const duration = Math.round(
      members.reduce((seconds, dataset) => seconds + dataset.sub_seconds + dataset.ref_seconds, 0) * 1000,
    );
    return {
      id: `category:${category}`,
      name: `${category} PCA recovery`,
      description: `${category} bounded scientific and resource-quality score; status requires integrity and private mastery`,
      status: categoryStatus(category, members),
      duration,
      score,
      ...(refusal ? { error: refusal } : {}),
      // Match grader.rollup(): within-category dataset weights determine the category mean,
      // while the average reviewed dataset weight determines the category's benchmark weight.
      // Keeping that weight here makes the platform's TestResult aggregation equal the sealed
      // grader reward instead of silently reverting to an unweighted category mean.
      weight: members.reduce((sum, dataset) => sum + dataset.weight, 0) / members.length,
      output: JSON.stringify({ datasets: members.map(datasetDiagnostic) }),
    };
  });
}
