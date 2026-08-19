import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const adapter = fs.readFileSync(path.join(root, "environment/src/index.ts"), "utf8");
const provision = fs.readFileSync(path.join(root, "environment/provision.sh"), "utf8");
const agentImage = fs.readFileSync(
  path.join(root, "tasks/from_scratch_pca/environment/Dockerfile"), "utf8",
);
const instruction = fs.readFileSync(
  path.join(root, "tasks/from_scratch_pca/instruction.md"), "utf8",
);

test("rollout analysis and both benchmark keys are root-private and never materialized", () => {
  assert.match(
    adapter,
    /const ROLLOUT_ANALYSIS_DIR = path\.join\(REPO_ROOT, "rollout_analysis"\);/,
  );
  assert.match(adapter, /const SOURCE_MATH_KEY_DIR = path\.join\(REPO_ROOT, "grader", "private"\);/);
  assert.match(adapter, /const SOURCE_MATH_KEY = path\.join\(SOURCE_MATH_KEY_DIR, "release-v3\.math-key"\);/);
  assert.match(adapter, /fs\.chownSync\(ROLLOUT_ANALYSIS_DIR, 0, 0\);/);
  assert.match(adapter, /fs\.chmodSync\(ROLLOUT_ANALYSIS_DIR, 0o700\);/);
  assert.match(adapter, /fs\.chownSync\(SOURCE_MATH_KEY_DIR, 0, 0\);/);
  assert.match(adapter, /fs\.chmodSync\(SOURCE_MATH_KEY_DIR, 0o700\);/);
  assert.match(adapter, /fs\.chownSync\(SOURCE_MATH_KEY, 0, 0\);/);
  assert.match(adapter, /fs\.chmodSync\(SOURCE_MATH_KEY, 0o400\);/);
  assert.match(adapter, /--reuid pcasub --regid pcasub --clear-groups/);
  assert.match(adapter, /\/usr\/bin\/test ! -r/);
  assert.match(adapter,
    /\[ROLLOUT_ANALYSIS_DIR, MATH_KEY, SOURCE_MATH_KEY\]/);

  const setup = adapter.split("async setupProblem", 2)[1].split("async runTests", 1)[0];
  const tests = adapter.split("async runTests", 2)[1];
  for (const source of [setup, tests]) {
    assert.match(source, /sealPrivateBenchmarkState\(\);/);
    assert.match(source, /await assertSubmissionCannotReadPrivateState\(\);/);
  }

  assert.doesNotMatch(setup, /copyFileSync.*ROLLOUT_ANALYSIS_DIR/);
  assert.doesNotMatch(setup, /cpSync.*ROLLOUT_ANALYSIS_DIR/);
});

test("the private release key never enters the solver image or prompt", () => {
  assert.match(adapter, /\/usr\/bin\/test ! -r/);
  assert.match(adapter, /release-v3\.math-key/);
  assert.doesNotMatch(agentImage, /release-v3\.math-key|pcabench-secrets|grader\/private/);
  assert.doesNotMatch(instruction, /release-v3\.math-key|math key|key commitment/);
});

test("the coding agent cannot read evaluator source, retained solutions, or Git history", () => {
  assert.match(provision, /workspace="\$repo_root\/workspace"/);
  assert.match(provision, /\[ ! -d "\$workspace" \] \|\| \[ -L "\$workspace" \]/);
  assert.match(provision, /find "\$repo_root" -mindepth 1 -maxdepth 1 -print0/);
  assert.match(provision, /find "\$repo_root"[^\n]+> "\$top_level_entries"/);
  assert.doesNotMatch(provision, /done < <\(/);
  assert.match(provision, /agent_launcher="\$repo_root\/packages"/);
  assert.match(provision, /if \[ "\$entry" = "\$workspace" \] \|\| \[ "\$entry" = "\$agent_launcher" \]; then/);
  assert.match(provision, /unreviewed top-level environment entry/);
  assert.match(provision, /\/bin\/chmod 0700 "\$entry"/);
  assert.match(provision, /\/bin\/chmod 0600 "\$entry"/);
  assert.match(provision, /\/bin\/chmod 0711 "\$repo_root"/);
  assert.doesNotMatch(provision, /continuing unsealed/);
  assert.doesNotMatch(provision, /chmod .*"\$repo_root\/workspace"/);
});

test("the solve budget is stated plainly without evaluation language", () => {
  const sentence = "You have 24 hours; get something in before then.";
  const problems = fs.readFileSync(path.join(root, "environment/problems.yaml"), "utf8");
  const task = fs.readFileSync(path.join(root, "tasks/from_scratch_pca/task.toml"), "utf8");
  const hyperfocal = fs.readFileSync(path.join(root, "hyperfocal.yaml"), "utf8");
  assert.ok(instruction.includes(sentence));
  assert.ok(problems.includes(sentence));
  assert.match(task, /timeout_sec = 86400/);
  assert.match(hyperfocal, /HYPERFOCAL_AGENT_TIMEOUT_MS: "86400000"/);
  assert.doesNotMatch(task, /HYPERFOCAL_AGENT_TIMEOUT_MS|\[solution\.env\]/);
  assert.doesNotMatch(agentImage, /HYPERFOCAL_AGENT_TIMEOUT_MS/);
  assert.doesNotMatch(instruction, /grader|grading|evaluat(?:e|ed|ion)/i);
});

test("both solver prompt surfaces state the writable runtime contract", () => {
  const problems = fs.readFileSync(path.join(root, "environment/problems.yaml"), "utf8");
  for (const prompt of [instruction, problems]) {
    assert.match(prompt, /write[^\n]+(?:out_path|output file)[^\n]+in place/i);
    assert.match(prompt, /TMPDIR/);
    assert.match(prompt, /only[^\n]+writ(?:e|able)/i);
    assert.match(prompt, /read-only/i);
  }
});
