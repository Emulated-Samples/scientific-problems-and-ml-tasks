import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import test from "node:test";

import { createGradeRun, readPrivateReward } from "../src/grade-run.js";

function permissions(target) {
  return fs.statSync(target).mode & 0o777;
}

test("grade run exposes traversal only while private artifacts remain sealed", () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "pcabench-grade-run-test-"));
  try {
    const run = createGradeRun(temporaryRoot);

    assert.equal(permissions(run.runDir), 0o711);
    assert.equal(permissions(run.dataDir), 0o700);
    assert.equal(permissions(run.runDir) & 0o044, 0);
    assert.equal(permissions(run.runDir) & 0o011, 0o011);

    fs.writeFileSync(run.rewardPath, '{"reward":1}', { mode: 0o600, flag: "wx" });
    assert.deepEqual(readPrivateReward(run.rewardPath, run.ownerUid), { reward: 1 });
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
});

test("reward reader rejects public files and symlinks", () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "pcabench-reward-test-"));
  try {
    const publicReward = path.join(temporaryRoot, "public.json");
    fs.writeFileSync(publicReward, "{}", { mode: 0o644, flag: "wx" });
    assert.throws(() => readPrivateReward(publicReward, fs.statSync(temporaryRoot).uid), /owner-private/);

    const privateReward = path.join(temporaryRoot, "private.json");
    const rewardLink = path.join(temporaryRoot, "reward-link.json");
    fs.writeFileSync(privateReward, "{}", { mode: 0o600, flag: "wx" });
    assert.throws(
      () => readPrivateReward(privateReward, fs.statSync(privateReward).uid + 1),
      /unexpected owner/,
    );
    fs.symlinkSync(privateReward, rewardLink);
    assert.throws(() => readPrivateReward(rewardLink, fs.statSync(temporaryRoot).uid));
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
});
