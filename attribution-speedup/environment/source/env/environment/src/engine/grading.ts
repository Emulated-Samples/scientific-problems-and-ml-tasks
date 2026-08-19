import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { Logger, TestResult } from "@hyperfocal/env-base";
import { executeWithExitCode, runSubprocessTests } from "@hyperfocal/env-base";
import { BEHAVIORAL_TESTS_DIR, graderPython, workspacePath } from "../paths.js";
import type { ProblemGrading } from "../problems/types.js";
import { stageGoldBase } from "./provisioning.js";
import { gradeEnv, pytestSelectionArgs } from "./scoping.js";

const GRADE_TIMEOUT_MS = 20 * 60 * 1000;

// Anti-inflation floor for the live gold target: the host-relative target may
// never fall below this fraction of the calibrated constant (see the guard
// where it is applied).
const GOLD_TARGET_FLOOR_FRAC = 0.5;

function baseName(id: string): string {
  return id.split("[")[0];
}

interface F2pEntry {
  key: string;
  file: string;
  testId: string;
  weight: number;
}

function parseF2p(grading: ProblemGrading, problemId: string): F2pEntry[] {
  return Object.entries(grading.f2p).map(([key, weight]) => {
    const sep = key.indexOf("::");
    if (sep <= 0 || sep === key.length - 2) {
      throw new Error(
        `${problemId}: f2p key ${key} is not a node id ("<file>::<test>")`
      );
    }
    return { key, file: key.slice(0, sep), testId: key.slice(sep + 2), weight };
  });
}

function validateF2p(
  entries: F2pEntry[],
  grading: ProblemGrading,
  problemId: string
): void {
  for (const e of entries) {
    if (!fs.existsSync(path.join(BEHAVIORAL_TESTS_DIR, e.file))) {
      throw new Error(`${problemId}: f2p file not found: ${e.file}`);
    }
    const inSuite = grading.suites.some(
      (s) => e.file === s || e.file.startsWith(`${s}/`)
    );
    if (!inSuite) {
      throw new Error(
        `${problemId}: f2p file ${e.file} is outside the declared suites ` +
          `[${grading.suites.join(", ")}] — it would never be collected`
      );
    }
  }
  const keys = new Set(entries.map((e) => e.key));
  for (const key of Object.keys(grading.continuous ?? {})) {
    if (!keys.has(key)) {
      throw new Error(
        `${problemId}: continuous key ${key} matches no f2p entry`
      );
    }
  }
}

function f2pEntryFor(id: string, entries: F2pEntry[]): F2pEntry | undefined {
  const exact = entries.find((e) => e.testId === id);
  if (exact) {
    return exact;
  }
  const bare = baseName(id);
  return entries.find((e) => e.testId === bare && !e.testId.includes("["));
}

async function measureSrcDiff(
  ws: string
): Promise<{ files: number; lines: number; ref: string }> {
  const tagProbe = await executeWithExitCode(
    "git rev-parse -q --verify refs/tags/hyperfocal-baseline",
    { cwd: ws }
  );
  const ref = tagProbe.success ? "hyperfocal-baseline" : "HEAD";
  const stat = await executeWithExitCode(`git diff --numstat ${ref} -- src/`, {
    cwd: ws,
  });
  let files = 0;
  let lines = 0;
  if (stat.success) {
    for (const row of stat.output.trim().split("\n")) {
      if (!row) continue;
      const [ins, del] = row.split("\t");
      const changed = (parseInt(ins, 10) || 0) + (parseInt(del, 10) || 0);
      if (changed === 0) continue;
      files += 1;
      lines += changed;
    }
  }
  return { files, lines, ref };
}

export async function runProblemTests(
  problemId: string,
  grading: ProblemGrading,
  logger: Logger
): Promise<TestResult[]> {
  const ws = workspacePath();
  const resultsFile = path.join(os.tmpdir(), `lat-bench-grade-${process.pid}.xml`);
  logger.info(`Grading '${problemId}' against ${ws}`);

  const f2pEntries = parseF2p(grading, problemId);
  validateF2p(f2pEntries, grading, problemId);

  const resultsCopy = `${resultsFile}.raw`;
  const metricsFile = path.join(os.tmpdir(), `lat-bench-metrics-${process.pid}.json`);
  if (fs.existsSync(metricsFile)) {
    fs.unlinkSync(metricsFile);
  }
  const command = [
    graderPython(),
    "-m",
    "pytest",
    ...pytestSelectionArgs(grading),
    "-q",
    "--no-header",
    "--continue-on-collection-errors",
    `--junitxml=${resultsFile}`,
    ";",
    `cp ${resultsFile} ${resultsCopy}`,
  ].join(" ");
  // Gold baseline lives ONLY for the duration of this grading run, in an
  // ephemeral random-path dir (see stageGoldBase): never in the image, never
  // at a guessable path, removed as soon as the suite finishes.
  const goldDir = grading.goldBaseline
    ? await stageGoldBase(logger)
    : undefined;
  let raw: TestResult[];
  try {
    raw = await runSubprocessTests(command, {
      resultsFile,
      cwd: BEHAVIORAL_TESTS_DIR,
      timeout: GRADE_TIMEOUT_MS,
      logger,
      env: gradeEnv(grading, ws, metricsFile, goldDir),
    });
  } finally {
    if (goldDir) {
      fs.rmSync(goldDir, { recursive: true, force: true });
    }
  }

  let perfMetrics: Record<string, number> = {};
  if (fs.existsSync(metricsFile)) {
    try {
      perfMetrics = JSON.parse(fs.readFileSync(metricsFile, "utf-8"));
    } catch {
    }
    fs.unlinkSync(metricsFile);
  }
  const continuous = grading.continuous ?? {};

  const isF2p = (r: TestResult) => f2pEntryFor(r.id, f2pEntries) !== undefined;

  let surgicalScale = 1;
  const guard = grading.surgicalGuard;
  if (guard) {
    const diff = await measureSrcDiff(ws);
    if (diff.files > guard.maxFiles || diff.lines > guard.maxLines) {
      surgicalScale = guard.penalty;
      logger.info(
        `surgical guard tripped: ${diff.files} files / ${diff.lines} lines ` +
          `changed under src/ vs ${diff.ref} (bounds ${guard.maxFiles}/` +
          `${guard.maxLines}) — f2p scores scaled by ${guard.penalty}`
      );
    }
  }

  let metricScale = 1;
  const mm = grading.metricMultiplier;
  if (mm) {
    const value = perfMetrics[mm.metric];
    if (typeof value === "number" && Number.isFinite(value)) {
      metricScale = Math.min(
        1,
        Math.max(0, (value - mm.floor) / (mm.ceiling - mm.floor))
      );
      logger.info(
        `metric multiplier '${mm.metric}' = ${value.toFixed(3)} → scale ${metricScale.toFixed(3)}`
      );
    } else {
      metricScale = 0;
      logger.info(
        `metric multiplier '${mm.metric}' missing from sidecar — f2p scores scaled to 0 (fail closed)`
      );
    }
  }
  surgicalScale *= metricScale;

  const skippedNames = new Set<string>();
  if (fs.existsSync(resultsCopy)) {
    const xml = fs.readFileSync(resultsCopy, "utf-8");
    const re =
      /<testcase\s+classname="[^"]*?"\s+name="([^"]*?)"[^>]*>(?:(?!<\/testcase>|<testcase)[\s\S])*?<skipped/g;
    for (let m = re.exec(xml); m !== null; m = re.exec(xml)) {
      skippedNames.add(baseName(m[1]));
    }
    fs.unlinkSync(resultsCopy);
  }

  const p2p = raw.filter((r) => !isF2p(r));
  const p2pScored = p2p.filter((r) => r.status === "passed" || r.status === "failed");
  const p2pPassed = p2pScored.filter((r) => r.status === "passed").length;
  const regressionScale = p2pScored.length > 0 ? p2pPassed / p2pScored.length : 1;
  if (regressionScale < 1) {
    logger.info(
      `regression multiplier ${regressionScale.toFixed(4)} ` +
        `(${p2pScored.length - p2pPassed} of ${p2pScored.length} p2p tests failing)`
    );
  }

  const results = raw.map((r) => {
    const entry = f2pEntryFor(r.id, f2pEntries);
    if (entry !== undefined) {
      if (skippedNames.has(r.id) || skippedNames.has(baseName(r.id))) {
        return {
          ...r,
          status: "failed" as const,
          weight: entry.weight,
          score: 0,
          error: "f2p test was skipped at collection/setup — scored as failed",
        };
      }
      const cont = continuous[entry.key];
      const metric = cont ? perfMetrics[cont.metric] : undefined;
      if (cont && typeof metric === "number" && Number.isFinite(metric)) {
        // Host-relative target (hardware-agnostic grading): when the grade
        // suite measured the GRADER-OWNED gold reference live on this same
        // host in this same run, it records "<metric>_gold". We normalise the
        // agent's ratio against what gold actually achieves HERE rather than
        // against an x86-calibrated constant, so the reference solution scores
        // ~1.0 on any architecture. cont.target stays the fallback for hosts /
        // problems that did not stage gold. Two guards on the live value:
        //  - degenerate gold (<= 1.05x, no measurable effect) falls back to
        //    the constant, so a broken calibration can never zero the metric;
        //  - the live target never drops below half the constant
        //    (GOLD_TARGET_FLOOR_FRAC), bounding how far anything that slows
        //    the gold measurement (adversarial CPU contention included) can
        //    lower the bar. Every legitimately measured host sits well above
        //    this floor (lowest observed: gold/const = 0.78 on Graviton).
        const goldVal = perfMetrics[`${cont.metric}_gold`];
        const hostTarget =
          typeof goldVal === "number" && Number.isFinite(goldVal) && goldVal > 1.05
            ? Math.max(goldVal, GOLD_TARGET_FLOOR_FRAC * cont.target)
            : cont.target;
        const partial = Math.min(1, Math.max(0, (metric - 1) / (hostTarget - 1)));
        const src =
          hostTarget === cont.target ? "target(const)" : "target(gold@host)";
        return {
          ...r,
          weight: entry.weight,
          score: partial * regressionScale * surgicalScale,
          output: `measured ratio ${metric.toFixed(2)} vs ${src} ${hostTarget.toFixed(2)} → continuous score ${partial.toFixed(3)}`,
        };
      }
      const base = r.status === "passed" ? 1 : 0;
      return {
        ...r,
        weight: entry.weight,
        score: base * regressionScale * surgicalScale,
      };
    }
    return { ...r, weight: 0 };
  });

  const seen = new Set<string>();
  for (const r of results) {
    const entry = f2pEntryFor(r.id, f2pEntries);
    if (entry !== undefined) {
      seen.add(entry.key);
    }
  }
  for (const entry of f2pEntries) {
    if (!seen.has(entry.key)) {
      results.push({
        id: entry.testId,
        name: entry.testId,
        status: "failed",
        duration: 0,
        weight: entry.weight,
        score: 0,
        error: "f2p test missing from pytest report (collection failure?)",
      });
    }
  }
  return results;
}
