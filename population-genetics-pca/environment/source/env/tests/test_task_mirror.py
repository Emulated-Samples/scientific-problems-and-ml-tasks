"""The packaged Harbor mirror must be able to RUN the grader, not merely match it byte-for-byte.

CI already runs ``scripts/build_task.py --check``, but a drift check only proves the mirror equals
what the build rule produces -- it cannot notice that the build rule itself excludes a file the
grader requires. That gap shipped a real defect: ``ReferenceCache`` scores speed against what is
achievable per fold and therefore times BOTH ``reference.full_scan_pca`` and ``reference.fast_pca``
as anchors, while the mirror's ignore list still excluded ``fast_pca.py`` as an "oracle". The
mirrored verifier could not measure its own fast anchor, so every dataset would raise a fail-closed
infrastructure error -- for the gold submission exactly as for a broken one, which is precisely the
kind of failure a byte-comparison cannot see.

These tests pin the causal property instead: every anchor the grader names must exist and import
inside the mirror tree, and the anchor list is READ OUT OF the grader's own source, so adding a new
anchor without mirroring it fails here rather than in a deployed verifier.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MIRROR = ROOT / "tasks" / "from_scratch_pca" / "tests" / "grader_pkg"


def _anchor_modules() -> list[str]:
    """Every module the grader times through ``ReferenceCache._time_anchor``.

    Derived from the source so this test tracks the grader instead of restating it.
    """
    tree = ast.parse((ROOT / "grader" / "grade.py").read_text())
    modules = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_time_anchor"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            modules.append(node.args[0].value)
    return sorted(set(modules))


def test_the_grader_times_more_than_one_anchor():
    """Guards the test above: if anchor discovery silently found nothing, it would vacuously pass."""
    modules = _anchor_modules()
    assert "reference.full_scan_pca" in modules
    assert "reference.fast_pca" in modules


@pytest.mark.parametrize("module", _anchor_modules())
def test_mirror_contains_every_anchor_the_grader_requires(module: str):
    path = MIRROR / Path(*module.split(".")).with_suffix(".py")
    assert path.is_file(), (
        f"the grader times {module} as a scoring anchor, but the packaged mirror does not ship it; "
        f"the mirrored verifier would fail closed on every dataset"
    )


@pytest.mark.parametrize("module", _anchor_modules())
def test_mirrored_anchor_imports_inside_the_mirror_tree(module: str):
    """Presence is not enough -- the anchor must import with ONLY the mirror on the path.

    ``_reference_submission_dir`` snapshots the reference package into a sandbox and runs the module
    as ``__main__``; an anchor whose imports reach outside the mirrored packages would resolve here
    in the repo and fail in the sealed verifier image.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
        cwd=MIRROR, capture_output=True, text=True,
        env={"PYTHONPATH": str(MIRROR), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, f"{module} does not import from the mirror alone: {proc.stderr}"


def test_mirror_still_withholds_the_directly_copyable_gold_submission():
    """The functional fix must not become a licence to ship everything.

    ``fast_submission/pca`` is a ready-made answer the grader never needs: anchors snapshot the
    ``reference`` package and ignore it. Its exclusion is defence-in-depth that costs nothing, so it
    stays excluded even though the mirror now (correctly) carries the fast anchor's source.
    """
    assert not (MIRROR / "reference" / "fast_submission").exists()


def test_mirror_matches_the_build_rule():
    """The drift check itself, so a stale mirror fails the suite and not only CI."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_task.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, f"packaged mirror is stale: {proc.stdout}{proc.stderr}"
