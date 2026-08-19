import * as fs from "fs";
import * as path from "path";

/**
 * @typedef {object} GradeRun
 * @property {string} runDir
 * @property {string} dataDir
 * @property {string} workDir
 * @property {string} rewardPath
 * @property {number} ownerUid
 */

/**
 * Create the root-owned filesystem layout shared with the dropped sandbox uid.
 *
 * Bubblewrap opens bind sources after `setpriv` drops to `pcasub`, so every
 * ancestor of `work/` must be searchable by that uid. The random run root is
 * therefore execute-only to non-owners: it can be crossed when its full path is
 * already known, but not listed or read. Hidden datasets remain in a separate
 * owner-only directory.
 *
 * @param {string} temporaryRoot
 * @returns {GradeRun}
 */
export function createGradeRun(temporaryRoot) {
  const runDir = fs.mkdtempSync(path.join(temporaryRoot, "pcabench-grade-"));
  try {
    fs.chmodSync(runDir, 0o711);
    const dataDir = path.join(runDir, "data");
    fs.mkdirSync(dataDir, { mode: 0o700 });
    return {
      runDir,
      dataDir,
      workDir: path.join(runDir, "work"),
      rewardPath: path.join(runDir, "reward_detail.json"),
      ownerUid: fs.statSync(runDir).uid,
    };
  } catch (error) {
    fs.rmSync(runDir, { recursive: true, force: true });
    throw error;
  }
}

/**
 * Read a root-produced reward without following a substituted path.
 *
 * @param {string} rewardPath
 * @param {number} ownerUid
 * @returns {unknown}
 */
export function readPrivateReward(rewardPath, ownerUid) {
  const fd = fs.openSync(
    rewardPath,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW,
  );
  try {
    const metadata = fs.fstatSync(fd);
    if (!metadata.isFile()) {
      throw new Error("grader reward is not a regular file");
    }
    if (metadata.uid !== ownerUid) {
      throw new Error("grader reward has an unexpected owner");
    }
    if ((metadata.mode & 0o077) !== 0) {
      throw new Error("grader reward is not owner-private");
    }
    return JSON.parse(fs.readFileSync(fd, "utf8"));
  } finally {
    fs.closeSync(fd);
  }
}
