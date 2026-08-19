import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { fileURLToPath } from "url";
import type { Logger } from "@hyperfocal/env-base";
import { executeWithExitCode } from "@hyperfocal/env-base";
import {
  GRADE_VENV_DIR,
  REQUIREMENTS_TEST,
  graderPython,
  workspacePath,
} from "../paths.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ENV_DIR = path.resolve(__dirname, "..", "..");

export const PRISTINE_BASE_DIR = "/hyperfocal/pristine-base";

// Gold reference branch in the env repo bundle (the optimized reference
// solution). Present in every base-* task's repo.bundle; used only by the
// grader to calibrate the host-relative speedup target. Never on the agent's
// PYTHONPATH, never in the workspace.
const GOLD_REF = "base-gold";

const REPO_ROOT = path.resolve(ENV_DIR, "..");

export async function materializeProblemState(
  branch: string,
  logger: Logger
): Promise<string> {
  let resolved: string | null = null;
  for (const candidate of [branch, `origin/${branch}`]) {
    const probe = await executeWithExitCode(
      `git rev-parse --verify --quiet ${candidate}`,
      { cwd: REPO_ROOT }
    );
    if (probe.success) {
      resolved = candidate;
      break;
    }
  }
  if (!resolved) {
    throw new Error(
      `state branch ${branch} not found (tried ${branch} and ` +
        `origin/${branch}) in ${REPO_ROOT} — cannot materialize the ` +
        `problem workspace state`
    );
  }

  const clean = await executeWithExitCode("git clean -fd -- workspace", {
    cwd: REPO_ROOT,
  });
  if (!clean.success) {
    throw new Error(`git clean of workspace failed: ${clean.output}`);
  }
  const restore = await executeWithExitCode(
    `git restore --source=${resolved} --worktree -- workspace`,
    { cwd: REPO_ROOT }
  );
  if (!restore.success) {
    throw new Error(
      `git restore of workspace from ${resolved} failed: ${restore.output}`
    );
  }
  logger.info(`workspace state materialized from ${resolved}`);
  return resolved;
}

async function resolveBasePython(logger: Logger): Promise<string> {
  for (const candidate of ["python3.12", "python3.11", "python3.10", "python3"]) {
    const probe = await executeWithExitCode(
      `${candidate} -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"`,
      { cwd: ENV_DIR }
    );
    if (probe.success) {
      return candidate;
    }
  }
  logger.info("No python >= 3.10 on PATH; installing python3.11 via dnf");
  const install = await executeWithExitCode("dnf install -y python3.11", {
    cwd: ENV_DIR,
  });
  if (!install.success) {
    throw new Error(`python3.11 install failed: ${install.output}`);
  }
  return "python3.11";
}

export async function ensureGradeVenv(logger: Logger): Promise<void> {
  if (fs.existsSync(graderPython())) {
    return;
  }
  const basePython = await resolveBasePython(logger);
  logger.info(`Provisioning grade venv at ${GRADE_VENV_DIR} (${basePython})`);
  const venv = await executeWithExitCode(`${basePython} -m venv ${GRADE_VENV_DIR}`, {
    cwd: ENV_DIR,
  });
  if (!venv.success) {
    throw new Error(`venv creation failed: ${venv.output}`);
  }
  const install = await executeWithExitCode(
    `${graderPython()} -m pip install -r ${REQUIREMENTS_TEST}`,
    { cwd: ENV_DIR }
  );
  if (!install.success) {
    throw new Error(`grade dependency install failed: ${install.output}`);
  }
}

const SOLVER_VENV_DIR = "/hyperfocal/solver-venv";

export async function ensureSolverVenv(logger: Logger): Promise<void> {
  const venvDir = SOLVER_VENV_DIR;
  const venvPython = path.join(venvDir, "bin", "python");
  if (fs.existsSync(venvPython)) {
    return;
  }
  const basePython = await resolveBasePython(logger);
  logger.info(`Provisioning solver venv at ${venvDir} (${basePython})`);
  const venv = await executeWithExitCode(`${basePython} -m venv ${venvDir}`, {
    cwd: ENV_DIR,
  });
  if (!venv.success) {
    throw new Error(`solver venv creation failed: ${venv.output}`);
  }
  const deps = await executeWithExitCode(
    `${venvPython} -m pip install -r ${REQUIREMENTS_TEST}`,
    { cwd: ENV_DIR }
  );
  if (!deps.success) {
    throw new Error(`solver venv dependency install failed: ${deps.output}`);
  }
  const lat = await executeWithExitCode(
    `${venvPython} -m pip install --no-deps -e ${workspacePath()}`,
    { cwd: ENV_DIR }
  );
  if (!lat.success) {
    throw new Error(`solver venv lat install failed: ${lat.output}`);
  }
  const hasAgentUser = await executeWithExitCode("id -u hyperfocal-agent", {
    cwd: ENV_DIR,
  });
  if (hasAgentUser.success) {
    await executeWithExitCode(
      `chown -R hyperfocal-agent:hyperfocal-agent ${venvDir}`,
      { cwd: ENV_DIR }
    );
  }
}

export async function ensurePristineBase(
  logger: Logger,
  stateRef?: string
): Promise<void> {
  const marker = path.join(PRISTINE_BASE_DIR, "src", "lat", "__init__.py");
  if (fs.existsSync(marker)) {
    return;
  }
  const ws = workspacePath();
  fs.rmSync(PRISTINE_BASE_DIR, { recursive: true, force: true });
  fs.mkdirSync(PRISTINE_BASE_DIR, { recursive: true });

  if (stateRef) {
    const fromRef = await executeWithExitCode(
      `git archive ${stateRef} -- workspace/src | tar -x --strip-components=1 -C ${PRISTINE_BASE_DIR}`,
      { cwd: REPO_ROOT }
    );
    if (fromRef.success && fs.existsSync(marker)) {
      logger.info(`pristine baseline staged from ${stateRef}`);
      return;
    }
    throw new Error(
      `pristine baseline staging from ${stateRef} failed — refusing to ` +
        `fall back (HEAD carries the gold workspace, which must never be ` +
        `the baseline)`
    );
  }

  const tagProbe = await executeWithExitCode(
    "git rev-parse -q --verify refs/tags/hyperfocal-baseline",
    { cwd: ws }
  );
  if (tagProbe.success) {
    const fromTag = await executeWithExitCode(
      `git archive hyperfocal-baseline -- src | tar -x -C ${PRISTINE_BASE_DIR}`,
      { cwd: ws }
    );
    if (fromTag.success && fs.existsSync(marker)) {
      logger.info(`pristine baseline staged from hyperfocal-baseline tag`);
      return;
    }
  }

  const fromHead = await executeWithExitCode(
    `git archive HEAD -- workspace/src | tar -x --strip-components=1 -C ${PRISTINE_BASE_DIR}`,
    { cwd: path.resolve(ENV_DIR, "..") }
  );
  if (fromHead.success && fs.existsSync(marker)) {
    logger.info(`pristine baseline staged from env repo HEAD`);
    return;
  }

  logger.info("pristine baseline: falling back to live workspace copy");
  fs.cpSync(path.join(ws, "src"), path.join(PRISTINE_BASE_DIR, "src"), {
    recursive: true,
  });
}

/**
 * Stage the GOLD reference workspace src (the optimized reference solution)
 * from the env repo bundle branch `base-gold` into a FRESH random temp dir,
 * returning the staged dir (or undefined when the ref is unavailable — the
 * grader then keeps the x86-calibrated constant targets).
 *
 * Grade-time only, never at image build: gold must not exist in the image the
 * agent works in, so even a permission regression can never expose the
 * solution at solve time. The random path (mkdtemp, chmod 700) additionally
 * denies grade-time path-guessing by agent import-time code; the caller
 * removes the dir as soon as grading finishes.
 */
export async function stageGoldBase(
  logger: Logger
): Promise<string | undefined> {
  const goldDir = fs.mkdtempSync(path.join(os.tmpdir(), "hf-gold-"));
  fs.chmodSync(goldDir, 0o700);
  const marker = path.join(goldDir, "src", "lat", "__init__.py");
  for (const ref of [GOLD_REF, `origin/${GOLD_REF}`]) {
    const probe = await executeWithExitCode(
      `git rev-parse --verify --quiet ${ref}`,
      { cwd: REPO_ROOT }
    );
    if (!probe.success) continue;
    const archived = await executeWithExitCode(
      `git archive ${ref} -- workspace/src | tar -x --strip-components=1 -C ${goldDir}`,
      { cwd: REPO_ROOT }
    );
    if (archived.success && fs.existsSync(marker)) {
      // The gold tree IS the reference solution — root-only, fail-safe
      // (grading runs as root; mirrors the Dockerfile's o-rwx on .git).
      await executeWithExitCode(`chmod -R go-rwx ${goldDir}`, {
        cwd: REPO_ROOT,
      });
      logger.info(`gold baseline staged from ${ref} (ephemeral dir)`);
      return goldDir;
    }
  }
  fs.rmSync(goldDir, { recursive: true, force: true });
  logger.info(
    `gold baseline ref '${GOLD_REF}' not available — host-relative speedup ` +
      `calibration disabled, falling back to constant targets`
  );
  return undefined;
}
