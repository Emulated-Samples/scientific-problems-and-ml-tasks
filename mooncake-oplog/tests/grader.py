#!/usr/bin/env python3
"""Verifier for mooncake-oplog.

Restores the pristine golden test sources over the agent's tree (agent edits
to tests are discarded by construction), rebuilds the graded gtest targets,
runs each binary once, and writes a weighted reward to
/logs/verifier/reward.json.

Groups map to one or more binaries; a group scores weight * (passed / total)
over the pinned test set of its binaries. A cmake configure failure zeroes
everything; a binary that fails to (re)build zeroes only the groups that
contain it — on a task this wide, one broken corner must not erase credit
for the rest. Graded binaries are deleted before the build so a stale or
pre-cooked binary can never be executed in place of a failed rebuild.

The graded binaries link the agent's own code, so it runs during grading
(including via static initializers). Mitigations: binaries run as an
unprivileged non-agent user that cannot write the root-owned reward file or
signal the grader; each binary's enumerated test set must equal a pinned
manifest, so agent-registered extra tests are rejected; and a missing gtest
JSON result counts every test in that binary as failed rather than crediting
anything parsed from stdout.
"""

import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REWARD_PATH = Path("/logs/verifier/reward.json")
REPORT_PATH = Path("/logs/verifier/report.json")
LOG_DIR = Path("/logs/verifier")
GTEST_OUT_DIR = LOG_DIR / "gtests"
# World-writable temp for the unprivileged test process (testing::TempDir()).
RUN_TMP_DIR = LOG_DIR / "tmp"
RUN_AS_USER = "nobody"


def _drop_privileges():
    try:
        ent = pwd.getpwnam(RUN_AS_USER)
    except KeyError:
        return None
    uid, gid = ent.pw_uid, ent.pw_gid

    def preexec():
        os.setgid(gid)
        os.setgroups([])
        os.setuid(uid)

    return preexec


RUN_AS = _drop_privileges()


def sh(cmd, cwd=None, timeout=None, log_name=None, env=None, drop=False):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        text=True,
        preexec_fn=RUN_AS if (drop and RUN_AS) else None,
    )
    if log_name:
        (LOG_DIR / log_name).write_text(proc.stdout)
    return proc


def emit(reward: float, detail: dict) -> None:
    """reward.json carries flat scalars only (harbor schema); full detail
    goes to report.json next to it."""
    REPORT_PATH.write_text(json.dumps({"reward": reward, **detail}, indent=1))
    flat: dict = {"reward": reward}
    for gname, g in detail.get("groups", {}).items():
        if isinstance(g, dict) and "fraction" in g:
            flat[f"group_{gname}"] = g["fraction"]
    for key in ("build_failed", "static_check_failed", "verifier_error"):
        if key in detail:
            flat[key] = 1
    REWARD_PATH.write_text(json.dumps(flat))


def run_binary(binary: Path, out_json: Path, env, timeout: int,
               excluded: set[str]) -> dict:
    """Run every non-excluded test in the binary once; per-test pass/fail
    from gtest JSON. gtest options travel via GTEST_FILTER/GTEST_OUTPUT env
    vars, NOT argv — two upstream suites run strict gflags parsing before
    InitGoogleTest and hard-exit on unknown --gtest_* flags. Excluded tests
    are filtered out so a crash in one can never take down the counted
    tests. Missing JSON (crash/timeout) yields an empty dict: callers count
    absent tests as failures. Never credit stdout, which the agent's linked
    code controls."""
    out_json.unlink(missing_ok=True)
    run_env = dict(env)
    run_env["GTEST_FILTER"] = "-" + ":".join(sorted(excluded))
    run_env["GTEST_OUTPUT"] = f"json:{out_json}"
    try:
        sh(
            [str(binary)],
            timeout=timeout, log_name=f"run_{binary.name}.log", env=run_env,
            drop=True,
        )
    except subprocess.TimeoutExpired:
        out_json.unlink(missing_ok=True)
        return {"__timeout__": True}
    results: dict = {}
    if out_json.exists():
        data = json.loads(out_json.read_text())
        for ts in data.get("testsuites", []):
            for t in ts.get("testsuite", []):
                # Pass = ran without failure records; SKIPPED (no failures)
                # also counts as pass rather than penalizing
                # environment-conditional tests.
                results[f"{ts['name']}.{t['name']}"] = not t.get("failures")
    else:
        results["__crashed__"] = True
    return results


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    GTEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(GTEST_OUT_DIR, 0o777)
    os.chmod(RUN_TMP_DIR, 0o777)
    cfg = json.loads((TESTS_DIR / "grading.json").read_text())
    repo = Path(cfg["repo_root"])
    build_dir = repo / cfg["build_dir"]
    binaries = cfg["binaries"]
    detail: dict = {"groups": {}}

    # Unprivileged test env: writes land in a world-writable temp, not the tree.
    run_env = os.environ.copy()
    run_env["TMPDIR"] = str(RUN_TMP_DIR)
    run_env["TEST_TMPDIR"] = str(RUN_TMP_DIR)

    # 1. Restore trusted test sources (discard any agent edits under tests/).
    for entry in cfg["restore"]:
        src = TESTS_DIR / entry["from"]
        dst = repo / entry["to"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o644)

    # 1b. Static check: reject vendored headers shadowing the test framework
    # (a fake gtest.h in the include path could trivially "pass" every test).
    shadows = [
        str(p)
        for p in repo.rglob("*")
        if p.name in {"gtest.h", "gtest-spi.h", "gmock.h"}
        and "build" not in p.parts
        and not p.is_relative_to(repo / "extern")
    ]
    if shadows:
        detail["static_check_failed"] = {"gtest_shadow": shadows}
        emit(0.0, detail)
        return

    # 2a. Reconfigure: the golden CMakeLists registers test targets that the
    # existing Makefiles don't know about, and `make <new-target>` fails with
    # "No rule to make target" before it ever re-runs cmake. Offline-safe:
    # the gtest FetchContent tarball is already populated in build/_deps.
    conf = sh(["cmake", "."], cwd=build_dir, timeout=600, log_name="configure.log")
    if conf.returncode != 0:
        detail["build_failed"] = True
        detail["configure_failed"] = True
        emit(0.0, detail)
        return

    # 2b. Delete every graded binary first: a target whose rebuild fails must
    # count as missing, never fall back to a stale (possibly agent-cooked)
    # executable left on disk.
    for b in binaries.values():
        (build_dir / b["path"]).unlink(missing_ok=True)

    # 2c. Rebuild all graded targets, keeping going past per-target failures
    # (-k): one broken corner zeroes its own groups, not the whole task.
    # os.cpu_count() sees the HOST's cores in a Modal sandbox, not the cgroup
    # limit — cap it, or a header-touching rebuild OOM-kills the sandbox.
    jobs = min(os.cpu_count() or 8, 16)
    build = sh(
        ["make", "-k", f"-j{jobs}"] + [b["target"] for b in binaries.values()],
        cwd=build_dir,
        timeout=cfg.get("build_timeout_sec", 2400),
        log_name="build.log",
    )
    missing = sorted(
        name for name, b in binaries.items()
        if not (build_dir / b["path"]).exists()
    )
    detail["missing_binaries"] = missing
    if len(missing) == len(binaries):
        detail["build_failed"] = True
        emit(0.0, detail)
        return
    # The unprivileged test user must reach and exec the built binaries.
    sh(["chmod", "-R", "a+rX", str(build_dir)])
    # Some suites create and clear hardcoded data dirs under /workspace; the
    # unprivileged runner needs to manage entries there. (The reward file
    # lives under root-owned /logs, so grading integrity is unaffected.)
    try:
        os.chmod("/workspace", 0o777)
    except OSError:
        pass

    # 3. A no-op or shadowed gtest could make every graded assertion look
    # green. Every golden source carries a verifier-only test that must FAIL;
    # run it on the first surviving binary and validate its structured result
    # before trusting the framework for reward-bearing tests.
    canary_name = "VerifierIntegrityCanary.AssertionMustFailWhenEnabled"
    canary_owner = next(
        (n for n in sorted(binaries) if n not in missing), None)
    canary_bin = build_dir / binaries[canary_owner]["path"]
    canary_json = GTEST_OUT_DIR / "gtest_canary.json"
    canary_json.unlink(missing_ok=True)
    canary_env = dict(run_env)
    canary_env["MOONCAKE_VERIFIER_RUN_CANARY"] = "1"
    canary_env["GTEST_FILTER"] = canary_name
    canary_env["GTEST_OUTPUT"] = f"json:{canary_json}"
    canary = sh(
        [str(canary_bin)],
        timeout=120, log_name="run_canary.log", env=canary_env, drop=True,
    )
    canary_failed_as_expected = False
    if canary_json.exists():
        canary_data = json.loads(canary_json.read_text())
        for ts in canary_data.get("testsuites", []):
            for test in ts.get("testsuite", []):
                if (f"{ts['name']}.{test['name']}" == canary_name
                        and test.get("failures")):
                    canary_failed_as_expected = True
    if canary.returncode == 0 or not canary_failed_as_expected:
        detail["static_check_failed"] = {
            "gtest_canary": True, "canary_binary": canary_owner}
        emit(0.0, detail)
        return

    # 4. Run each surviving binary once. The observed test set (the run's
    # gtest JSON keys) must equal the pinned manifest exactly: extra
    # agent-registered tests or silently dropped ones are an integrity
    # failure that zeroes that binary, not partial credit. (There is no
    # separate --gtest_list_tests enumeration pass — two upstream suites
    # reject unknown argv flags, so listing and running both go through
    # GTEST_* env vars, and one full run supplies both signals.)
    excluded = set(cfg.get("exclude", []))
    manifest_name = cfg.get("manifest")
    manifest = None
    if manifest_name and (TESTS_DIR / manifest_name).exists():
        manifest = json.loads((TESTS_DIR / manifest_name).read_text())
    else:
        # First-run mode (manifest not frozen yet): report loudly so a
        # shipped task can never silently grade without the pin.
        detail["manifest_missing"] = True

    per_binary_results: dict[str, dict] = {}
    observed_by_binary: dict[str, list] = {}
    bad_manifest = set()
    for name, b in binaries.items():
        if name in missing:
            per_binary_results[name] = {}
            observed_by_binary[name] = []
            continue
        results = run_binary(
            build_dir / b["path"],
            GTEST_OUT_DIR / f"gtest_{name}.json",
            run_env,
            timeout=cfg.get("run_timeout_sec", 1200),
            excluded=excluded,
        )
        if results.pop("__timeout__", None):
            detail.setdefault("timed_out_binaries", []).append(name)
        if results.pop("__crashed__", None):
            detail.setdefault("crashed_binaries", []).append(name)
        observed = sorted(results)
        observed_by_binary[name] = observed
        if manifest is not None and observed:
            expected_set = set(manifest.get(name, []))
            if set(observed) != expected_set:
                bad_manifest.add(name)
                detail.setdefault("test_set_mismatch", {})[name] = {
                    "missing": sorted(expected_set - set(observed))[:50],
                    "unexpected": sorted(set(observed) - expected_set)[:50],
                }
                results = {}
        per_binary_results[name] = results
    detail["observed_by_binary"] = observed_by_binary

    reward = 0.0
    for gname, group in cfg["groups"].items():
        expected: list[str] = []
        results: dict[str, bool] = {}
        for bname in group["binaries"]:
            names = (
                manifest.get(bname, []) if manifest is not None
                else observed_by_binary[bname]
            )
            expected.extend(names)
            results.update(per_binary_results.get(bname, {}))
        passed = sum(1 for t in expected if results.get(t))
        detail.setdefault("tests", {}).update(
            {t: bool(results.get(t, False)) for t in expected}
        )
        frac = passed / len(expected) if expected else 0.0
        reward += group["weight"] * frac
        detail["groups"][gname] = {
            "weight": group["weight"],
            "passed": passed,
            "total": len(expected),
            "fraction": round(frac, 4),
        }

    emit(round(reward, 4), detail)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — any grader crash is reward 0
        REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REWARD_PATH.write_text(json.dumps({"reward": 0, "verifier_error": 1}))
        REPORT_PATH.write_text(
            json.dumps({"reward": 0, "verifier_error": 1, "exception": str(exc)})
        )
        sys.exit(1)
