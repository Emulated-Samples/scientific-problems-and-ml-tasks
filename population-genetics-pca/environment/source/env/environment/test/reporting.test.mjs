import assert from "node:assert/strict";
import test from "node:test";

import { categoryTestResults, validDatasetMeasurements } from "../dist/reporting.js";

function dataset(overrides = {}) {
  return {
    dataset: "grade_continental_a",
    category: "continental",
    k: 10,
    weight: 1,
    reward: 0.71,
    accuracy: 0.99,
    time_quality: 0.75,
    gate_product: 1,
    sub_seconds: 1.25,
    ref_seconds: 0.5,
    fast_ref_seconds: 0.25,
    execution_overhead_seconds: 0.01,
    gates: {
      coverage: { factor: 1 },
      hwe_norm: { factor: 1 },
      representation_equivalence: { factor: 1 },
      library_scan: { factor: 1 },
      validity: { factor: 1 },
    },
    run: {
      returncode: 0,
      stderr_tail: "",
      sampled_primary_invocation_peak_pss_bytes: 64 * 1024 * 1024,
      sampled_primary_invocation_peak_storage_bytes: 4096,
    },
    ...overrides,
  };
}

test("category status is component-based and reports real grader duration", () => {
  const members = [
    dataset(),
    dataset({ dataset: "grade_continental_b", reward: 0.55, sub_seconds: 2, ref_seconds: 0.25 }),
  ];
  const [result] = categoryTestResults({ continental: 0.63 }, members);

  assert.equal(result.status, "passed");
  assert.equal(result.score, 0.63);
  assert.equal(result.weight, 1);
  assert.equal(result.duration, 4000);
  assert.deepEqual(JSON.parse(result.output), {
    datasets: [
      {
        dataset: "grade_continental_a",
        k: 10,
        accuracy: 0.99,
        reward: 0.71,
        time_quality: 0.75,
        gate_product: 1,
        gates: { coverage: 1, hwe_norm: 1, library_scan: 1, representation_equivalence: 1, validity: 1 },
        gate_reasons: {},
        exit_code: 0,
        sub_seconds: 1.25,
        ref_seconds: 0.5,
        fast_ref_seconds: 0.25,
        execution_overhead_seconds: 0.01,
        sampled_primary_invocation_peak_pss_bytes: 64 * 1024 * 1024,
        sampled_primary_invocation_peak_storage_bytes: 4096,
        stderr_tail: "",
      },
      {
        dataset: "grade_continental_b",
        k: 10,
        accuracy: 0.99,
        reward: 0.55,
        time_quality: 0.75,
        gate_product: 1,
        gates: { coverage: 1, hwe_norm: 1, library_scan: 1, representation_equivalence: 1, validity: 1 },
        gate_reasons: {},
        exit_code: 0,
        sub_seconds: 2,
        ref_seconds: 0.25,
        fast_ref_seconds: 0.25,
        execution_overhead_seconds: 0.01,
        sampled_primary_invocation_peak_pss_bytes: 64 * 1024 * 1024,
        sampled_primary_invocation_peak_storage_bytes: 4096,
        stderr_tail: "",
      },
    ],
  });
});

test("category result weights preserve the grader's reviewed category weighting", () => {
  const [result] = categoryTestResults(
    { observed: 0.8 },
    [
      dataset({ dataset: "observed_a", category: "observed", weight: 2 }),
      dataset({ dataset: "observed_b", category: "observed", weight: 1 }),
    ],
  );

  assert.equal(result.weight, 1.5);
});

test("scientific weakness retains score but does not count as a category pass", () => {
  const [partial] = categoryTestResults(
    { continental: 0.4 },
    [dataset({ accuracy: 0.4, reward: 0.4 })],
  );
  const [zero] = categoryTestResults(
    { continental: 0 },
    [dataset({ accuracy: 0, reward: 0 })],
  );
  const [brokenGate] = categoryTestResults(
    { continental: 0.99 },
    [dataset({ gates: { ...dataset().gates, coverage: { factor: 0.899 } } })],
  );
  const [boundaryGate] = categoryTestResults(
    { continental: 0.4 },
    [dataset({ gates: { ...dataset().gates, coverage: { factor: 0.9 } } })],
  );
  const [badExit] = categoryTestResults(
    { continental: 0.99 },
    [dataset({ run: {
      returncode: 1,
      stderr_tail: "boom",
      sampled_primary_invocation_peak_pss_bytes: 64 * 1024 * 1024,
      sampled_primary_invocation_peak_storage_bytes: 4096,
    } })],
  );

  assert.equal(partial.status, "failed");
  assert.equal(zero.status, "failed");
  assert.equal(brokenGate.status, "failed");
  assert.equal(boundaryGate.status, "passed");
  assert.equal(badExit.status, "failed");
});

test("uniform systems mastery uses achievable-speed quality in every category", () => {
  const exactButParity = dataset({
    dataset: "grade_scaling",
    category: "scaling",
    accuracy: 1,
    reward: 0.65,
    time_quality: 0.5,
  });
  const [scaling] = categoryTestResults({ scaling: 0.65 }, [exactButParity]);
  const [highRank] = categoryTestResults(
    { high_rank: 0.65 },
    [{ ...exactButParity, dataset: "grade_high_rank", category: "high_rank" }],
  );
  const [optimized] = categoryTestResults(
    { scaling: 0.825 },
    [{ ...exactButParity, time_quality: 0.75, reward: 0.825 }],
  );
  const [biobank] = categoryTestResults(
    { biobank: 0.825 },
    [{ ...exactButParity, dataset: "grade_biobank", category: "biobank", time_quality: 0.75 }],
  );

  assert.equal(scaling.status, "failed");
  assert.equal(highRank.status, "failed");
  assert.equal(optimized.status, "passed");
  assert.equal(biobank.status, "passed");
});

test("one weak member blocks category mastery even when the category reward is high", () => {
  const [result] = categoryTestResults(
    { continental: 0.95 },
    [dataset(), dataset({ dataset: "grade_continental_b", accuracy: 0.80, reward: 0.95 })],
  );

  assert.equal(result.status, "failed");
  assert.equal(result.score, 0.95);
});

test("representation mastery is scoped to parser-stress categories", () => {
  const weakRepresentation = {
    ...dataset().gates,
    representation_equivalence: { factor: 0.5 },
  };
  const [continental] = categoryTestResults(
    { continental: 0.5 },
    [dataset({ gates: weakRepresentation })],
  );
  const [variableWidth] = categoryTestResults(
    { variable_width: 0.5 },
    [dataset({ category: "variable_width", gates: weakRepresentation })],
  );

  assert.equal(continental.status, "passed");
  assert.equal(variableWidth.status, "failed");
});

test("dataset result contract requires finite submission and reference timings", () => {
  assert.equal(validDatasetMeasurements(dataset()), true);
  assert.equal(validDatasetMeasurements(dataset({ sub_seconds: -0.001 })), false);
  assert.equal(validDatasetMeasurements(dataset({ sub_seconds: Number.NaN })), false);
  assert.equal(validDatasetMeasurements(dataset({ ref_seconds: Number.POSITIVE_INFINITY })), false);
  assert.equal(validDatasetMeasurements(dataset({ ref_seconds: "1" })), false);
  assert.equal(validDatasetMeasurements(dataset({ fast_ref_seconds: Number.NaN })), false);
  assert.equal(validDatasetMeasurements(dataset({ fast_ref_seconds: -0.001 })), false);
  assert.equal(validDatasetMeasurements(dataset({ fast_ref_seconds: 0 })), false);
  assert.equal(validDatasetMeasurements(dataset({ execution_overhead_seconds: Number.POSITIVE_INFINITY })), false);
  assert.equal(validDatasetMeasurements(dataset({ execution_overhead_seconds: -0.001 })), false);
  assert.equal(validDatasetMeasurements(dataset({
    run: { ...dataset().run, sampled_primary_invocation_peak_pss_bytes: -1 },
  })), false);
  assert.equal(validDatasetMeasurements(dataset({
    deadline_exhausted: true,
    reward: 0,
    accuracy: 0,
    time_quality: 0,
    gate_product: 0,
    fast_ref_seconds: null,
    execution_overhead_seconds: null,
    run: {
      returncode: 124,
      stderr_tail: "global grading deadline exhausted before this dataset",
      sampled_primary_invocation_peak_pss_bytes: null,
      sampled_primary_invocation_peak_storage_bytes: null,
    },
  })), true);
  assert.equal(validDatasetMeasurements(dataset({
    run: {
      ...dataset().run,
      sampled_primary_invocation_peak_pss_bytes: null,
      sampled_primary_invocation_peak_storage_bytes: null,
    },
  })), false);
  assert.equal(validDatasetMeasurements(dataset({
    deadline_exhausted: true,
  })), false);
  assert.equal(validDatasetMeasurements(dataset({
    deadline_exhausted: true,
    reward: 0,
    accuracy: 0,
    time_quality: 0,
    gate_product: 0,
    fast_ref_seconds: null,
    execution_overhead_seconds: null,
    run: {
      returncode: 124,
      stderr_tail: "global grading deadline exhausted before this dataset",
      sampled_primary_invocation_peak_pss_bytes: 0,
      sampled_primary_invocation_peak_storage_bytes: 0,
    },
  })), false);
});

test("dataset result contract requires probe gates unless library scan already failed", () => {
  const { coverage: _coverage, ...missingCoverage } = dataset().gates;
  assert.equal(validDatasetMeasurements(dataset({ gates: missingCoverage })), false);
  assert.equal(validDatasetMeasurements(dataset({
    gate_product: 0,
    gates: { library_scan: { factor: 0 }, validity: { factor: 1 } },
  })), true);
  assert.equal(validDatasetMeasurements(dataset({ gates: { ...dataset().gates, surprise: { factor: 1 } } })), false);
});

test("a total execution refusal is reported as a contract failure, not as a bad PCA", () => {
  // run_019f6281: every invocation exited 126 in ~0.6s because one fork-context Array() call
  // tripped the runtime guard. The reward was correctly zero, but the report was indistinguishable
  // from "wrote a hopeless PCA" -- while the implementation behind it later measured canonical
  // correlation 1.00000 against an independent reference.
  const refused = (name, category) => dataset({
    dataset: name,
    category,
    reward: 0,
    accuracy: 0,
    time_quality: 0,
    gate_product: 0,
    sub_seconds: 0.607,
    gates: {
      coverage: { factor: 0 },
      hwe_norm: { factor: 0 },
      representation_equivalence: { factor: 0 },
      library_scan: { factor: 1 },
      validity: { factor: 0 },
    },
    run: {
      returncode: 126,
      stderr_tail: "ctypes-backed multiprocessing shared objects are disabled\n",
      sampled_primary_invocation_peak_pss_bytes: 0,
      sampled_primary_invocation_peak_storage_bytes: 0,
    },
  });
  const members = [refused("grade_continental_a", "continental"), refused("grade_subtle_a", "subtle")];
  const results = categoryTestResults({ continental: 0, subtle: 0 }, members);

  for (const result of results) {
    assert.equal(result.status, "failed");
    assert.equal(result.score, 0);            // the zero is CORRECT and must not be inflated
    assert.match(result.error, /EXECUTION CONTRACT: the submission never ran/);
    assert.match(result.error, /NOT evidence about the quality of the algorithm/);
    // The runner's actual reason must reach the reader, so the contract breach is actionable.
    assert.match(result.error, /ctypes-backed multiprocessing shared objects are disabled/);
  }
});

test("a genuinely bad PCA is NOT excused as an execution-contract failure", () => {
  // The margin. A program that RUNS and computes garbage must keep reading as a capability result;
  // otherwise the new diagnostic would launder every zero into "probably infra".
  const ran = dataset({
    reward: 0,
    accuracy: 0,
    time_quality: 0.9,
    gate_product: 0,
    gates: {
      coverage: { factor: 0 },
      hwe_norm: { factor: 0 },
      representation_equivalence: { factor: 1 },
      library_scan: { factor: 1 },
      validity: { factor: 0 },
    },
    run: {
      returncode: 0,
      stderr_tail: "",
      sampled_primary_invocation_peak_pss_bytes: 1024,
      sampled_primary_invocation_peak_storage_bytes: 0,
    },
  });
  const [result] = categoryTestResults({ continental: 0 }, [ran]);
  assert.equal(result.status, "failed");
  assert.equal(result.error, undefined);
});

test("a partial refusal is a dataset-specific defect, not a contract sweep", () => {
  // One fold refusing while others run is a real, submission-specific bug. Only a TOTAL sweep is
  // evidence that the program never ran at all.
  const good = dataset();
  const bad = dataset({
    dataset: "grade_subtle_a",
    category: "continental",
    reward: 0,
    accuracy: 0,
    run: {
      returncode: 126,
      stderr_tail: "policy\n",
      sampled_primary_invocation_peak_pss_bytes: 0,
      sampled_primary_invocation_peak_storage_bytes: 0,
    },
  });
  const [result] = categoryTestResults({ continental: 0.3 }, [good, bad]);
  assert.equal(result.error, undefined);
});

test("a gate that fires explains itself in the artifact", () => {
  // library_scan zeroes a submission on a static finding, exits 0, and writes nothing to stderr.
  // The report used to show `library_scan: 0` and nothing else -- the reader could not tell a banned
  // import from a native payload from a computed import name. The gate knew all along.
  const scanned = dataset({
    reward: 0,
    accuracy: 0,
    gate_product: 0,
    gates: {
      coverage: { factor: 1 },
      hwe_norm: { factor: 1 },
      representation_equivalence: { factor: 1 },
      library_scan: {
        factor: 0,
        name: "library_scan",
        severity: 1,
        n_hits: 1,
        hits: [{ file: "pca", kind: "import", module: "sklearn" }],
      },
      validity: { factor: 1 },
    },
  });
  const [result] = categoryTestResults({ continental: 0 }, [scanned]);
  const [diagnostic] = JSON.parse(result.output).datasets;

  assert.equal(diagnostic.gates.library_scan, 0);
  assert.deepEqual(diagnostic.gate_reasons.library_scan.hits,
    [{ file: "pca", kind: "import", module: "sklearn" }]);
  assert.equal(diagnostic.gate_reasons.library_scan.n_hits, 1);
  // The verdict itself is not repeated as its own explanation.
  assert.equal(diagnostic.gate_reasons.library_scan.factor, undefined);
  assert.equal(diagnostic.gate_reasons.library_scan.severity, undefined);
});

test("gates that passed are not given reasons", () => {
  // A clean grade must stay readable; only failures owe an explanation.
  const [result] = categoryTestResults({ continental: 0.71 }, [dataset()]);
  const [diagnostic] = JSON.parse(result.output).datasets;
  assert.deepEqual(diagnostic.gate_reasons, {});
});
