"""Mirror the canonical grader into the packaged Harbor task tree.

The task under ``tasks/from_scratch_pca/`` must be self-contained at deploy time: its hidden
``tests/`` directory needs the grader, the reference implementations, the metrics/gates, and
the dataset generator so the verifier can grade ``/app/submission`` with no repo checkout.

Rather than keep a second copy in git (drift risk + repo bloat), we mirror on demand from the
canonical top-level packages and drift-check. Run this before packaging/deploying the task;
CI can run it with ``--check`` to assert the shipped mirror matches source.

Usage:
  python scripts/build_task.py            # (re)build the mirror
  python scripts/build_task.py --check    # fail if the mirror is stale
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "tasks" / "from_scratch_pca"
MIRROR = TASK / "tests" / "grader_pkg"

# Top-level packages the grader needs at grade time.
PACKAGES = ["grader", "reference", "data"]
# ``reference/fast_pca.py`` is REQUIRED scoring state, not an optional extra: the grader measures
# speed against what is achievable per fold, and ``ReferenceCache`` times BOTH the full-scan and the
# fast reference through the submission's own execution path. Excluding it from the mirror does not
# make the verifier safer -- it makes it non-functional, because a missing fast anchor is a
# fail-closed infrastructure error on every dataset.
#
# Shipping it is exactly as safe as shipping ``full_scan_pca.py``, which is the scoring TRUTH and has
# always been mirrored -- a strictly more valuable secret than a fast implementation of it. Three
# independent boundaries, none of which depend on the file's absence: the mirror lands only in the
# sealed verifier image (tests/Dockerfile ``COPY . /tests`` + ``chmod -R go-rwx``) and never in the
# agent image; ``grade._seal_grader_package`` makes ``reference/`` root-only 0700 with a preflight
# that verifies the dropped uid cannot read it; and anchor runs materialize a throwaway copy inside a
# per-invocation bubblewrap sandbox that exposes only that invocation's own submission directory.
#
# ``fast_submission/`` stays excluded: it is a directly copyable gold answer, the grader never needs
# it (``grade._reference_submission_dir`` ignores it when snapshotting the package), and keeping it
# out costs nothing. Keep this list in sync with ``_ORACLE_ONLY``.
_ORACLE_ONLY = {"fast_submission"}
_NON_RUNTIME_DATA = {"REAL_DATA.md"}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "generated", "big", "dev", "real",
                                *_ORACLE_ONLY, *_NON_RUNTIME_DATA)


def build():
    if MIRROR.exists():
        shutil.rmtree(MIRROR)
    MIRROR.mkdir(parents=True)
    for pkg in PACKAGES:
        shutil.copytree(ROOT / pkg, MIRROR / pkg, ignore=IGNORE)
    (MIRROR / "__init__.py").write_text("")
    print(f"mirrored {PACKAGES} -> {MIRROR.relative_to(ROOT)}")


def _diff(a: Path, b: Path) -> list[str]:
    """Recursively list files that differ, are missing, or are STALE (mirror-only) between a
    (source) and b (mirror). Skips intentionally-excluded artefacts (bytecode, datasets, oracle)
    so the check flags only real drift."""
    out = []
    skip = {"generated", "big", "dev", "real"} | _ORACLE_ONLY | _NON_RUNTIME_DATA
    cmp = filecmp.dircmp(a, b, ignore=["__pycache__"])
    for name in cmp.left_only:
        if name.endswith(".pyc") or name in skip:
            continue                                   # excluded on purpose -> not a drift
        out.append(f"missing in mirror: {(a / name)}")
    # right_only: files present ONLY in the mirror -- stale copies of since-renamed/deleted source
    # (or an injected extra) that would otherwise ship and be graded. A --check that ignores these
    # accepts exactly the drift it exists to catch, so we report them.
    for name in cmp.right_only:
        if name.endswith(".pyc") or name == "__pycache__":
            continue
        out.append(f"stale in mirror (not in source): {(b / name)}")
    for name in cmp.diff_files:
        out.append(f"differs: {(a / name)}")
    for sub in cmp.common_dirs:
        out.extend(_diff(a / sub, b / sub))
    return out


def check() -> int:
    if not MIRROR.exists():
        print("mirror missing; run build_task.py")
        return 1
    problems = []
    for pkg in PACKAGES:
        problems.extend(_diff(ROOT / pkg, MIRROR / pkg))
    if problems:
        print("MIRROR DRIFT:")
        for p in problems:
            print("  " + p)
        return 1
    print("mirror is up to date")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
