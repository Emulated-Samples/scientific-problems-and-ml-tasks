import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import type {
  EnvironmentDefinition,
  Logger,
  Problem,
  TestResult,
} from "@hyperfocal/env-base";
import { ConsoleLogger, loadProblemsFromDirectory } from "@hyperfocal/env-base";
import { runProblemTests } from "./engine/grading.js";
import {
  ensureGradeVenv,
  ensurePristineBase,
  ensureSolverVenv,
  materializeProblemState,
} from "./engine/provisioning.js";
import { PROBLEM_GRADING } from "./problems/index.js";
import { workspacePath } from "./paths.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ENV_DIR = path.resolve(__dirname, "..");
const problems: Problem[] = loadProblemsFromDirectory(ENV_DIR);

class Environment implements EnvironmentDefinition {
  async listProblems(): Promise<Problem[]> {
    return problems;
  }

  async setupProblem(problemId?: string, logger?: Logger): Promise<void> {
    const log = logger ?? new ConsoleLogger();
    const grading = problemId ? PROBLEM_GRADING[problemId] : undefined;
    let stateRef: string | undefined;
    if (grading?.stateBranch) {
      stateRef = await materializeProblemState(grading.stateBranch, log);
    }
    const latInit = path.join(workspacePath(), "src", "lat", "__init__.py");
    if (!fs.existsSync(latInit)) {
      throw new Error(`workspace missing lat package at ${latInit}`);
    }
    await ensureGradeVenv(log);
    await ensureSolverVenv(log);
    if (grading?.pristineBaseline) {
      await ensurePristineBase(log, stateRef);
    }
    // NOTE: the gold baseline (goldBaseline flag) is deliberately NOT staged
    // at setup/image-build time — it is the reference solution and must never
    // exist in the image the agent works in. runProblemTests stages it into
    // an ephemeral random-path dir at grade time and removes it after.
    log.info(`setup complete for '${problemId ?? "default"}'`);
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    const grading = PROBLEM_GRADING[problemId];
    if (!grading) {
      throw new Error(
        `no grading definition for problem '${problemId}' — add ` +
          `src/problems/p-${problemId}.ts and register it in src/problems/index.ts`
      );
    }
    await ensureGradeVenv(logger);
    if (grading.pristineBaseline) {
      await ensurePristineBase(logger);
    }
    return runProblemTests(problemId, grading, logger);
  }
}

export default new Environment();
