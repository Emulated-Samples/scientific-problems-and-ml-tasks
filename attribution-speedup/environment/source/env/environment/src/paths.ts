import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const ENV_DIR = path.resolve(__dirname, "..");

export const BEHAVIORAL_TESTS_DIR = path.join(ENV_DIR, "behavioral-tests");

export function workspacePath(): string {
  return process.env.WORKSPACE_PATH ?? path.resolve(ENV_DIR, "..", "workspace");
}

export const GRADE_VENV_DIR = path.join(ENV_DIR, ".grade-venv");
export const REQUIREMENTS_TEST = path.join(ENV_DIR, "requirements-test.txt");

export function graderPython(): string {
  return process.env.GRADER_PYTHON ?? path.join(GRADE_VENV_DIR, "bin", "python");
}
