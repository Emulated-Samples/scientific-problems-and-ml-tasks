import {
  EnvironmentDefinition,
  Logger,
  TestResult,
  execute,
  loadProblemsFromDirectory,
} from "@hyperfocal/env-base";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { createHash, randomBytes } from "crypto";
import { fileURLToPath } from "url";
import { createGradeRun, readPrivateReward } from "./grade-run.js";
import {
  CATEGORY_NAMES,
  categoryTestResults,
  DatasetDetail,
  gateReason,
  validDatasetMeasurements,
} from "./reporting.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ENV_DIR = path.join(__dirname, "..");
const REPO_ROOT = path.join(ENV_DIR, "..");
const ROLLOUT_ANALYSIS_DIR = path.join(REPO_ROOT, "rollout_analysis");
const SOURCE_MATH_KEY_DIR = path.join(REPO_ROOT, "grader", "private");
const SOURCE_MATH_KEY = path.join(SOURCE_MATH_KEY_DIR, "release-v3.math-key");
const WORKSPACE = path.join(REPO_ROOT, "workspace");
const SUBMISSION_DIR = path.join(WORKSPACE, "submission");
const PROVISION = path.join(ENV_DIR, "provision.sh");
const PYTHON = "/opt/hyperfocal/pcabench/bin/python";
// Isolation mode for the grader's untrusted submission phases. ZERO-CONFIG
// DEFAULT: verifier-container. This is the harbor-release product branch and must
// grade under stock harbor with NOTHING set. Harbor's Docker verifier container is
// NOT granted CAP_SYS_ADMIN, so bubblewrap cannot create its namespaces and its
// preflight dies "bwrap: No permissions to creating new namespace". The grader's
// verifier-container mode instead chroots into a static root-owned sandbox root
// (built by provision.sh) and drops uid/caps via setpriv -- chroot(2) needs only
// CAP_SYS_CHROOT, which the verifier container DOES have. Read-protection is the
// same sealPrivateBenchmarkState 0700 root seal bwrap mode relies on (asserted
// unreadable as pcasub before grading); the chroot adds defense-in-depth.
//   Native EC2 rollouts, where bwrap CAN create namespaces, opt back into it with
// PCABENCH_ISOLATION=bwrap. Nothing set == the shipped default == verifier-container.
const ISOLATION = process.env.PCABENCH_ISOLATION === "bwrap"
  ? "bwrap"
  : "verifier-container";
const REAL_DATA_ROOT = "/opt/hyperfocal/pcabench-data";
const REAL_VCF = path.join(REAL_DATA_ROOT, "chr22.vcf");
const REAL_TRUTH = `${REAL_VCF}.truth.json`;
const MATH_KEY = "/opt/hyperfocal/pcabench-secrets/release-v3.math-key";
const MATH_RELEASE = "pcabench-eval-v3";
const OBSERVED_DERIVATION = "observed-chr22-v5";
const REAL_SOURCE_SHA256 = "a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5";
const EXPECTED_CATEGORIES = CATEGORY_NAMES;
// data.generate.grade_specs defines 21 keyed synthetic tasks and the adapter adds the mandatory
// observed cohort below. Keep this total fail-closed: accepting a shortened catalogue would make a
// missing capability arm look like a valid, easier evaluation.
const EXPECTED_DATASETS = 22;
const problems = loadProblemsFromDirectory(ENV_DIR);

interface ExpectedDataset {
  category: typeof EXPECTED_CATEGORIES[number];
  k: number;
  weight: number;
}

interface GradeResult {
  submission_status: "completed" | "failed";
  submission_error?: string;
  reward: number;
  reward_detail: {
    execution_sandbox: string;
    category_scores: Record<string, number>;
    per_dataset: DatasetDetail[];
  };
}

function errorResult(message: string): TestResult[] {
  return [{
    id: "grade",
    name: "pcabench grade",
    status: "errored",
    duration: 0,
    score: 0,
    error: message,
  }];
}

function failedSubmissionResult(message: string): TestResult[] {
  return [{
    id: "grade",
    name: "pcabench grade",
    status: "failed",
    duration: 0,
    score: 0,
    error: message,
  }];
}

function runtimeEnvironment(): NodeJS.ProcessEnv {
  return {
    PATH: "/opt/hyperfocal/pcabench/bin:/usr/bin:/bin",
    HOME: "/tmp",
    TMPDIR: "/tmp",
    OMP_NUM_THREADS: "8",
    OPENBLAS_NUM_THREADS: "8",
    MKL_NUM_THREADS: "8",
    VECLIB_MAXIMUM_THREADS: "8",
  };
}

function sealPrivateBenchmarkState(): void {
  // rollout_analysis/ is an internal audit dir removed from the delivery; sealed below only if present.
  const sourceKey = fs.lstatSync(SOURCE_MATH_KEY);
  const sourceKeyDirectory = fs.lstatSync(SOURCE_MATH_KEY_DIR);
  if (!sourceKey.isFile() || sourceKey.isSymbolicLink() || sourceKey.nlink !== 1
      || !sourceKeyDirectory.isDirectory() || sourceKeyDirectory.isSymbolicLink()) {
    throw new Error("tracked private benchmark key has an unsafe filesystem shape");
  }
  if (fs.existsSync(ROLLOUT_ANALYSIS_DIR)) {
    fs.chownSync(ROLLOUT_ANALYSIS_DIR, 0, 0);
    fs.chmodSync(ROLLOUT_ANALYSIS_DIR, 0o700);
  }
  fs.chownSync(SOURCE_MATH_KEY_DIR, 0, 0);
  fs.chmodSync(SOURCE_MATH_KEY_DIR, 0o700);
  fs.chownSync(SOURCE_MATH_KEY, 0, 0);
  fs.chmodSync(SOURCE_MATH_KEY, 0o400);
}

async function assertSubmissionCannotReadPrivateState(): Promise<void> {
  for (const privatePath of [ROLLOUT_ANALYSIS_DIR, MATH_KEY, SOURCE_MATH_KEY]) {
    await execute(
      "/usr/bin/setpriv --reuid pcasub --regid pcasub --clear-groups "
        + "--inh-caps=-all --no-new-privs /usr/bin/test ! -r "
        + JSON.stringify(privatePath),
      {
        cwd: REPO_ROOT,
        timeout: 30_000,
        env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", HOME: "/tmp" },
      },
    );
  }
}

function mathKeyCommitment(): string {
  const info = fs.lstatSync(MATH_KEY);
  if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1
      || info.uid !== 0 || (info.mode & 0o077) !== 0) {
    throw new Error("private benchmark math key has unsafe ownership or permissions");
  }
  const encoded = fs.readFileSync(MATH_KEY, "ascii").trim();
  if (!/^[0-9a-f]{64}$/.test(encoded)) {
    throw new Error("private benchmark math key is malformed");
  }
  return createHash("sha256")
    .update(Buffer.from(`${MATH_RELEASE}\0`, "ascii"))
    .update(Buffer.from(encoded, "hex"))
    .digest("hex");
}

function approximatelyEqual(left: number, right: number): boolean {
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return Math.abs(left - right) <= 1e-12 * scale;
}

function stageObservedDataset(dataDir: string): void {
  const vcf = fs.lstatSync(REAL_VCF);
  const truth = fs.lstatSync(REAL_TRUTH);
  if (!vcf.isFile() || vcf.isSymbolicLink() || vcf.size < 100_000_000
      || vcf.size > 2_000_000_000
      || !truth.isFile() || truth.isSymbolicLink()) {
    throw new Error("reviewed observed-cohort bundle is missing or malformed");
  }
  if (vcf.uid !== 0 || truth.uid !== 0 || (vcf.mode & 0o022) !== 0
      || (truth.mode & 0o077) !== 0) {
    throw new Error("reviewed observed-cohort bundle has unsafe ownership or permissions");
  }
  const contract = JSON.parse(fs.readFileSync(REAL_TRUTH, "utf8")) as {
    spec?: {
      category?: string;
      k?: number;
      source_sha256?: string;
      derivation_version?: string;
      math_release?: string;
      math_key_commitment?: string;
      derived_sha256?: string;
      derived_bytes?: number;
      n_records?: number;
    };
    n_samples?: number;
    sample_ids?: unknown[];
    sample_pop?: unknown[];
    sample_subpop?: unknown[];
    population_names?: unknown[];
  };
  if (contract.spec?.category !== "observed" || contract.spec?.k !== 8
      || contract.spec?.source_sha256 !== REAL_SOURCE_SHA256
      || contract.spec?.derivation_version !== OBSERVED_DERIVATION
      || contract.spec?.math_release !== MATH_RELEASE
      || contract.spec?.math_key_commitment !== mathKeyCommitment()
      || !/^[0-9a-f]{64}$/.test(contract.spec?.derived_sha256 ?? "")
      || contract.spec?.derived_bytes !== vcf.size
      || (contract.spec?.n_records ?? 0) < 50_000
      || contract.n_samples !== 500 || contract.sample_ids?.length !== 500
      || contract.sample_pop?.length !== 500 || contract.sample_subpop?.length !== 500
      || !contract.sample_ids?.every((sample) => typeof sample === "string"
        && /^p[0-9a-f]{20}$/.test(sample))
      || (contract.population_names?.length ?? 0) < 20) {
    throw new Error("reviewed observed-cohort truth contract is invalid");
  }
  const destination = path.join(dataDir, "grade_observed.vcf");
  if (vcf.dev === fs.lstatSync(dataDir).dev) {
    fs.linkSync(REAL_VCF, destination);
  } else {
    fs.copyFileSync(REAL_VCF, destination, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(destination, 0o444);
  }
  fs.copyFileSync(REAL_TRUTH, `${destination}.truth.json`, fs.constants.COPYFILE_EXCL);
  fs.chmodSync(`${destination}.truth.json`, 0o400);
}

function expectedDatasets(dataDir: string): Map<string, ExpectedDataset> {
  const suffix = ".vcf.truth.json";
  const result = new Map<string, ExpectedDataset>();
  for (const filename of fs.readdirSync(dataDir).filter((name) => name.endsWith(suffix)).sort()) {
    const truth = JSON.parse(fs.readFileSync(path.join(dataDir, filename), "utf8")) as {
      spec?: {
        category?: string;
        k?: number;
        weight?: number;
        math_release?: string;
        math_key_commitment?: string;
      };
    };
    const category = truth.spec?.category;
    const k = truth.spec?.k;
    const weight = truth.spec?.weight ?? 1;
    if (!EXPECTED_CATEGORIES.includes(category as typeof EXPECTED_CATEGORIES[number])
        || !Number.isInteger(k) || (k as number) < 1
        || !Number.isFinite(weight) || weight <= 0
        || truth.spec?.math_release !== MATH_RELEASE
        || truth.spec?.math_key_commitment !== mathKeyCommitment()) {
      throw new Error(`invalid generated truth contract: ${filename}`);
    }
    result.set(filename.slice(0, -suffix.length), {
      category: category as typeof EXPECTED_CATEGORIES[number],
      k: k as number,
      weight,
    });
  }
  return result;
}

function validGradeResult(detail: GradeResult, expected: Map<string, ExpectedDataset>): boolean {
  const categories = detail.reward_detail?.category_scores ?? {};
  const perDataset = detail.reward_detail?.per_dataset ?? [];
  const categoryNames = Object.keys(categories).sort();
  const expectedCategoryNames = [...EXPECTED_CATEGORIES].sort();
  if (!(["completed", "failed"] as const).includes(detail.submission_status)
      || !Number.isFinite(detail.reward) || detail.reward < 0 || detail.reward > 1
      // Both reviewed confined execution sandboxes are valid: bwrap (native EC2
      // rollouts) and verifier-container (the zero-config default for packaged
      // harbor tasks, whose container lacks CAP_SYS_ADMIN for bwrap namespaces).
      // The grader stamps which one it used into reward_detail.execution_sandbox;
      // an unknown/absent value is still rejected as unconfined.
      || !(["bwrap", "verifier-container"] as const).includes(
        detail.reward_detail?.execution_sandbox as "bwrap" | "verifier-container")
      || expected.size !== EXPECTED_DATASETS
      || perDataset.length !== expected.size
      || JSON.stringify(categoryNames) !== JSON.stringify(expectedCategoryNames)
      || !Object.values(categories).every((score) => Number.isFinite(score) && score >= 0 && score <= 1)) {
    return false;
  }

  const actual = new Map<string, DatasetDetail>();
  for (const dataset of perDataset) {
    const contract = expected.get(dataset.dataset);
    if (!contract || actual.has(dataset.dataset)
        || dataset.category !== contract.category
        || dataset.k !== contract.k
        || !Number.isFinite(dataset.weight) || !approximatelyEqual(dataset.weight, contract.weight)
        || !Number.isFinite(dataset.reward) || dataset.reward < 0 || dataset.reward > 1) {
      return false;
    }
    actual.set(dataset.dataset, dataset);
  }
  if ([...expected.keys()].some((dataset) => !actual.has(dataset))) {
    return false;
  }

  if (detail.submission_status === "failed") {
    return detail.reward === 0
      && typeof detail.submission_error === "string"
      && detail.submission_error.length > 0
      && Object.values(categories).every((score) => score === 0)
      && perDataset.every((dataset) => dataset.reward === 0);
  }
  if (!perDataset.every(validDatasetMeasurements)) {
    return false;
  }

  // This block makes TypeScript a SECOND AUTHORITY on the reward, and that is a known way to kill an
  // environment: a sibling env is currently down because its TS validator re-derived reward with
  // stale weights while the Python grader had moved on, and the two are required to agree to 1e-12.
  // Every rollout errors, and no unit test on either side can see it -- each is self-consistent.
  //
  // The reason that cannot happen here is worth stating so it is not "simplified" away: this
  // re-derives ONLY THE AGGREGATION -- category means and the benchmark, from per-dataset rewards
  // the grader already computed. It never re-implements `dataset_score`. The accuracy/systems/gate
  // formula and its constants live in exactly one place (`grade.dataset_score`), so changing them
  // cannot desync the two authorities. That was load-bearing when the correctness floor moved
  // 0.30 -> 0.10: nothing here needed to know.
  //
  // So: if you are tempted to check the per-dataset arithmetic too, DON'T. Re-deriving a formula in
  // a second language does not verify it -- it duplicates it, and duplicates drift. Check structure
  // and aggregation, and let the single implementation own the formula.
  const recomputed: Array<{ score: number; weight: number }> = [];
  for (const category of EXPECTED_CATEGORIES) {
    const members = [...actual.values()].filter((dataset) => dataset.category === category);
    const weight = members.reduce((sum, dataset) => sum + expected.get(dataset.dataset)!.weight, 0);
    if (members.length === 0 || weight <= 0) {
      return false;
    }
    const score = members.reduce(
      (sum, dataset) => sum + dataset.reward * expected.get(dataset.dataset)!.weight,
      0,
    ) / weight;
    if (!approximatelyEqual(categories[category], score)) {
      return false;
    }
    recomputed.push({ score, weight: weight / members.length });
  }
  const totalCategoryWeight = recomputed.reduce((sum, category) => sum + category.weight, 0);
  const reward = recomputed.reduce(
    (sum, category) => sum + category.score * category.weight,
    0,
  ) / totalCategoryWeight;
  if (!approximatelyEqual(detail.reward, reward)) {
    return false;
  }
  return detail.submission_error === undefined;
}

class Environment implements EnvironmentDefinition {
  async listProblems() {
    return problems;
  }

  async setupProblem(problemId: string, logger?: Logger): Promise<void> {
    if (problemId !== "from_scratch_pca") {
      throw new Error(`unknown pcabench problem: ${problemId}`);
    }
    logger?.info("Provisioning the pinned Python and bubblewrap runtime");
    await execute(`/bin/bash ${JSON.stringify(PROVISION)}`, {
      cwd: REPO_ROOT,
      timeout: 30 * 60 * 1000,
      env: {
        PATH: "/usr/sbin:/usr/bin:/sbin:/bin",
        HOME: "/root",
      },
    });
    // Materialize the one exact problem state directly. The gold implementation
    // lives on main for environment validation; every rollout removes it here
    // before the kernel-confined solver starts.
    fs.rmSync(WORKSPACE, { recursive: true, force: true });
    fs.mkdirSync(SUBMISSION_DIR, { recursive: true });
    sealPrivateBenchmarkState();
    await assertSubmissionCannotReadPrivateState();
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    if (problemId !== "from_scratch_pca") {
      return errorResult(`unknown pcabench problem: ${problemId}`);
    }
    sealPrivateBenchmarkState();
    await assertSubmissionCannotReadPrivateState();
    logger.info(
      ISOLATION === "verifier-container"
        ? "isolation=verifier-container (default; bwrap self-sandbox skipped — harbor "
          + "grants CAP_SYS_CHROOT but not CAP_SYS_ADMIN). Set PCABENCH_ISOLATION=bwrap "
          + "for the native bubblewrap jail."
        : "isolation=bwrap (PCABENCH_ISOLATION=bwrap; native bubblewrap jail)",
    );
    const { runDir, dataDir, workDir, rewardPath, ownerUid } = createGradeRun(os.tmpdir());
    try {
      const representationKey = path.join(runDir, "synthetic-representation.key");
      fs.writeFileSync(representationKey, randomBytes(32), { mode: 0o600 });
      await execute(
        `${PYTHON} -m data.generate --suite grade --out-dir ${JSON.stringify(dataDir)} `
          + `--math-key-file ${JSON.stringify(MATH_KEY)} `
          + `--representation-key-file ${JSON.stringify(representationKey)}`,
        {
          cwd: REPO_ROOT,
          timeout: 60 * 60 * 1000,
          env: runtimeEnvironment(),
        },
      );
      fs.rmSync(representationKey);
      stageObservedDataset(dataDir);
      const expected = expectedDatasets(dataDir);

      await execute(
        `${PYTHON} ${JSON.stringify(path.join(REPO_ROOT, "grader", "grade.py"))} ` +
          `${JSON.stringify(SUBMISSION_DIR)} --data-dir ${JSON.stringify(dataDir)} ` +
          `--workdir ${JSON.stringify(workDir)} ` +
          `--out ${JSON.stringify(rewardPath)} ` +
          `--math-key-file ${JSON.stringify(MATH_KEY)} --isolation ${ISOLATION} ` +
          `--time-budget-seconds 8400`,
        {
          cwd: REPO_ROOT,
          timeout: 4 * 60 * 60 * 1000,
          env: runtimeEnvironment(),
        },
      );

      const detail = readPrivateReward(rewardPath, ownerUid) as GradeResult;
      const categories = detail.reward_detail?.category_scores ?? {};
      if (!validGradeResult(detail, expected)) {
        return errorResult(`grader produced an incomplete or unconfined result: ${JSON.stringify(detail)}`);
      }
      if (detail.submission_status === "failed") {
        return failedSubmissionResult(detail.submission_error!);
      }

      logger.info(`reward=${detail.reward.toFixed(4)} categories=${JSON.stringify(categories)}`);
      // A flat zero caused by a program that never executed must not read like a flat zero caused by
      // a bad PCA. `categoryTestResults` carries the same finding into every category's `error`.
      if (detail.reward_detail.per_dataset.every(
        (dataset) => dataset.run.returncode === 126 || dataset.run.returncode === 127,
      )) {
        logger.info(
          "EXECUTION CONTRACT: every invocation exited 126/127 without producing output. The zero "
          + "reflects a program that never ran, not a measured PCA. Check the runner's stderr below "
          + "against the task's stated runtime limits before reading this as a capability result.",
        );
      }
      for (const dataset of detail.reward_detail.per_dataset) {
        const gates = Object.entries(dataset.gates)
          .sort()
          .map(([name, gate]) => `${name}=${gate.factor.toFixed(1)}`)
          .join(",");
        // WHY a gate fired, on the human-readable line -- not only inside the result JSON. A run
        // that scores 0.0000 across every dataset shows `gates=0.000` here, and a reader scanning
        // these lines could not tell an empty output from a forbidden import from a wrong object;
        // the cause was already computed and sat one nesting level away in the artifact. An
        // unexplained zero is the single most expensive diagnostic failure this env has produced.
        const failures = Object.entries(dataset.gates)
          .sort()
          .map(([name, gate]) => {
            const reason = gateReason(gate);
            return reason === null ? null : `${name}:${JSON.stringify(reason)}`;
          })
          .filter((entry): entry is string => entry !== null)
          .join(" ");
        logger.info(
          `dataset=${dataset.dataset} reward=${dataset.reward.toFixed(4)} `
          + `accuracy=${dataset.accuracy.toFixed(4)} time=${dataset.time_quality.toFixed(3)} `
          + `gates=${dataset.gate_product.toFixed(3)} [${gates}] `
          + (failures ? `why=${failures} ` : "")
          + `exit=${dataset.run.returncode} `
          + `submission_seconds=${dataset.sub_seconds.toFixed(3)} `
          + `reference_seconds=${dataset.ref_seconds.toFixed(3)} `
          + `sampled_primary_invocation_peak_pss_bytes=`
          + `${dataset.run.sampled_primary_invocation_peak_pss_bytes ?? "unmeasured"} `
          + `sampled_primary_invocation_peak_storage_bytes=`
          + `${dataset.run.sampled_primary_invocation_peak_storage_bytes ?? "unmeasured"}`,
        );
      }
      return categoryTestResults(categories, detail.reward_detail.per_dataset);
    } catch (error) {
      return errorResult((error as Error).message);
    } finally {
      fs.rmSync(runDir, { recursive: true, force: true });
    }
  }
}

export default new Environment();