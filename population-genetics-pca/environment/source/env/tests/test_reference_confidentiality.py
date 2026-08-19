"""The hidden full-scan reference is the exact scoring truth; a submission that can read/exec it
aces the benchmark without doing the task. These tests pin the confidentiality boundary:

  * the static library scan is NOT that boundary -- a subprocess-based leak imports only
    os/subprocess and passes the scan clean;
  * ``_seal_grader_package`` makes only the hidden source subtrees un-traversable under mandatory
    UID separation, without locking the reusable environment repository root;
  * ``_assert_grader_package_sealed`` fails the grade CLOSED when the reference is still readable.

They run without root by locating the seal machinery at a temp package root and neutralising the
real ``setuid`` drop, so the OS read/deny decision is exercised as the current uid.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grader import grade
from grader.gates import library_scan


def _write_pca(d: Path, body: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "pca"
    p.write_text(body)
    return d


def test_grader_package_root_holds_the_reference():
    root = grade._grader_package_root()
    assert root is not None
    assert (root / "reference" / "full_scan_pca.py").exists()


def test_subprocess_leak_passes_the_static_library_scan():
    """The exploit imports only allowlisted modules, so the AST scan cannot be the defence: it
    reports a clean factor 1.0. This is precisely why the filesystem seal is required."""
    sub = _write_pca(Path(tmpdir_for("leaklib")),
                     "#!/usr/bin/env python3\n"
                     "import os, subprocess, sys\n"
                     "subprocess.run([sys.executable, '-m', 'reference.full_scan_pca'])\n")
    res = library_scan.scan(sub)
    assert res["factor"] == 1.0
    assert res["n_hits"] == 0


def test_seal_makes_only_hidden_source_directories_owner_only(monkeypatch, tmp_path):
    pkg = tmp_path / "grader_pkg"
    (pkg / "reference").mkdir(parents=True)
    (pkg / "reference" / "full_scan_pca.py").write_text("# truth\n")
    (pkg / "grader").mkdir()
    pkg.chmod(0o755)

    monkeypatch.setattr(grade, "_UNPRIV", (12345, 12345))
    monkeypatch.setattr(grade, "_grader_package_root", lambda: pkg)
    # chown to root needs privilege we don't have off the deployed box; the mode change is the
    # traversal barrier we assert here.
    monkeypatch.setattr(grade.os, "chown", lambda *a, **k: None)

    grade._seal_grader_package()
    assert (pkg.stat().st_mode & 0o777) == 0o755
    assert ((pkg / "reference").stat().st_mode & 0o777) == 0o700
    assert ((pkg / "grader").stat().st_mode & 0o777) == 0o700


def test_preflight_raises_when_reference_is_readable(monkeypatch, tmp_path):
    pkg = tmp_path / "grader_pkg"
    (pkg / "reference").mkdir(parents=True)
    (pkg / "reference" / "full_scan_pca.py").write_text("# truth\n")

    monkeypatch.setattr(grade, "_UNPRIV", (12345, 12345))
    monkeypatch.setattr(grade, "_grader_package_root", lambda: pkg)
    # Neutralise the real setuid drop (needs root); the child then reads as the current uid, which
    # can read the world-readable reference -> the preflight must detect the leak and fail closed.
    monkeypatch.setattr(grade, "_drop_privileges", lambda: None)

    with pytest.raises(RuntimeError, match="readable by the submission uid"):
        grade._assert_grader_package_sealed()


def test_preflight_passes_when_reference_is_unreadable(monkeypatch, tmp_path):
    pkg = tmp_path / "grader_pkg"
    (pkg / "reference").mkdir(parents=True)
    ref = pkg / "reference" / "full_scan_pca.py"
    ref.write_text("# truth\n")
    (pkg / "grader").mkdir()
    (pkg / "reference").chmod(0o000)                    # hidden subtree traversal is denied
    (pkg / "grader").chmod(0o000)

    monkeypatch.setattr(grade, "_UNPRIV", (12345, 12345))
    monkeypatch.setattr(grade, "_grader_package_root", lambda: pkg)
    monkeypatch.setattr(grade, "_drop_privileges", lambda: None)

    try:
        import os
        if hasattr(os, "geteuid") and os.geteuid() == 0:  # root ignores permission bits
            pytest.skip("running as root; owner permission bits do not deny reads")
        grade._assert_grader_package_sealed()            # must NOT raise
    finally:
        # The fixture deliberately removes traversal from the package root. Restore
        # owner access and remove the fixture contents ourselves. On macOS, pytest's
        # concurrent garbage collector can otherwise race a previously inaccessible
        # directory and emit a misleading ``Directory not empty`` warning even though
        # the confidentiality assertion passed.
        (pkg / "reference").chmod(0o700)
        (pkg / "grader").chmod(0o700)
        ref.unlink(missing_ok=True)
        (pkg / "reference").rmdir()
        (pkg / "grader").rmdir()
        pkg.rmdir()


def test_root_always_resolves_an_unprivileged_identity(monkeypatch):
    """Root grading has no opt-out or environment-controlled compatibility path."""
    import pwd
    from types import SimpleNamespace

    monkeypatch.setattr(grade.os, "geteuid", lambda: 0)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1234, pw_gid=1235))
    uid, gid = grade._resolve_unpriv()
    assert (uid, gid) == (1234, 1235)


def test_grading_rejects_same_uid_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(grade, "_UNPRIV", None)
    with pytest.raises(RuntimeError, match="must run as root"):
        grade.grade_suite(
            tmp_path / "submission",
            tmp_path / "data",
            tmp_path / "work",
            isolation="bwrap",
            math_key=bytes(range(32)),
            probes=False,
        )


def test_submission_runner_has_no_unconfined_command_path(monkeypatch, tmp_path):
    monkeypatch.setattr(grade, "_UNPRIV", None)
    with pytest.raises(RuntimeError, match="requires root"):
        grade._sandbox_argv(
            ["python", "pca"], tmp_path / "submission", tmp_path / "input", tmp_path / "output",
        )


def test_verifier_container_runner_requires_root_linux(monkeypatch):
    monkeypatch.setattr(grade, "_UNPRIV", (1234, 1235))
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade.os, "geteuid", lambda: 1000)

    with pytest.raises(RuntimeError, match="requires root on Linux"):
        grade._verifier_container_argv([grade._RUNTIME_PYTHON, "pca"])


def test_verifier_container_runner_drops_identity_groups_and_capabilities(monkeypatch, tmp_path):
    monkeypatch.setattr(grade, "_UNPRIV", (1234, 1235))
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade.os, "geteuid", lambda: 0)
    monkeypatch.setattr(grade.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(grade.os, "access", lambda path, mode: True)
    root_path = tmp_path / "chroot"
    guard = root_path / grade._GUARD_RUNNER.lstrip("/")
    guard.parent.mkdir(parents=True)
    guard.write_text("# guard\n")

    class Root:
        def stat(self):
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o555, st_uid=0)

        def __truediv__(self, relative):
            return root_path / relative

        def __str__(self):
            return str(root_path)

    root = Root()
    monkeypatch.setattr(grade, "_CHROOT_ROOT", root)
    command = [grade._RUNTIME_PYTHON, "-I", "pca", "input.vcf", "3", "output.tsv"]

    argv = grade._verifier_container_argv(command)

    assert argv[:3] == [grade._CHROOT, str(root_path), grade._SETPRIV]
    assert ["--reuid", "1234"] in [argv[i:i + 2] for i in range(len(argv) - 1)]
    assert ["--regid", "1235"] in [argv[i:i + 2] for i in range(len(argv) - 1)]
    assert "--clear-groups" in argv
    assert "--inh-caps=-all" in argv
    assert "--ambient-caps=-all" in argv
    assert "--bounding-set=-all" in argv
    assert "--no-new-privs" in argv
    assert grade._BWRAP not in argv
    assert argv[-len(command):] == command


def test_submission_argv_dispatch_is_explicit_and_rejects_unknown_modes(
    monkeypatch, tmp_path,
):
    bwrap = object()
    verifier_container = object()
    monkeypatch.setattr(grade, "_sandbox_argv", lambda *args: bwrap)
    monkeypatch.setattr(grade, "_verifier_container_argv", lambda command: verifier_container)
    args = (["python", "pca"], tmp_path / "submission", tmp_path / "input.vcf",
            tmp_path / "output.tsv")

    assert grade._submission_argv(*args, "bwrap") is bwrap
    assert grade._submission_argv(*args, "verifier-container") is verifier_container
    with pytest.raises(ValueError, match="unknown isolation mode"):
        grade._submission_argv(*args, "unconfined")


def test_execution_boundary_dispatch_is_explicit_and_rejects_unknown_modes(
    monkeypatch, tmp_path,
):
    calls = []
    monkeypatch.setattr(
        grade, "_assert_bwrap_boundary", lambda workdir: calls.append(("bwrap", workdir)),
    )
    monkeypatch.setattr(
        grade,
        "_assert_verifier_container_boundary",
        lambda workdir: calls.append(("verifier-container", workdir)),
    )

    grade._assert_execution_boundary(tmp_path, "bwrap")
    grade._assert_execution_boundary(tmp_path, "verifier-container")

    assert calls == [("bwrap", tmp_path), ("verifier-container", tmp_path)]
    with pytest.raises(ValueError, match="unknown isolation mode"):
        grade._assert_execution_boundary(tmp_path, "unconfined")


def test_sandbox_mounts_an_allowlist_not_the_host_root(monkeypatch, tmp_path):
    monkeypatch.setattr(grade, "_UNPRIV", (1234, 1235))
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade.os, "geteuid", lambda: 0)
    monkeypatch.setattr(grade.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(grade.os, "access", lambda path, mode: True)
    submission = tmp_path / "submission"
    staged_input = tmp_path / "input.vcf"
    staged_output = tmp_path / "output.tsv"

    argv = grade._sandbox_argv(
        [grade._RUNTIME_PYTHON, "-c", "pass"], submission, staged_input, staged_output,
    )

    assert ["--ro-bind", "/", "/"] not in [argv[i:i + 3] for i in range(len(argv) - 2)]
    assert ["--ro-bind", "/usr", "/usr"] in [argv[i:i + 3] for i in range(len(argv) - 2)]
    assert ["--ro-bind", str(submission), "/submission"] in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert ["--ro-bind", str(staged_input), "/work/input.vcf"] in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert ["--dir", "/work"] in [argv[i:i + 2] for i in range(len(argv) - 1)]
    assert ["--ro-bind", str(staged_input.parent), "/work"] not in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert ["--bind", str(staged_output), "/work/output.tsv"] in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert ["--bind", str(tmp_path / "tmp"), "/tmp"] in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert ["--tmpfs", "/tmp"] not in [
        argv[i:i + 2] for i in range(len(argv) - 1)
    ]
    assert ["--bind", str(tmp_path), "/sandbox"] not in [
        argv[i:i + 3] for i in range(len(argv) - 2)
    ]
    assert "--unshare-net" in argv
    assert "--unshare-pid" in argv


def test_bwrap_work_root_must_be_root_owned_and_not_listable(monkeypatch, tmp_path):
    state = {"mode": stat.S_IFDIR | 0o711}

    class Root:
        def stat(self, *, follow_symlinks=True):
            del follow_symlinks
            return SimpleNamespace(st_mode=state["mode"], st_uid=0)

        def __str__(self):
            return str(tmp_path / "bwrap-root")

    root = Root()
    monkeypatch.setattr(grade, "_BWRAP_WORK_ROOT", root)

    assert grade._bwrap_work_root() == root

    state["mode"] = stat.S_IFDIR | 0o755
    with pytest.raises(RuntimeError, match="invalid bubblewrap work root"):
        grade._bwrap_work_root()


def test_sandbox_leaves_proc_empty_instead_of_exposing_process_memory(monkeypatch, tmp_path):
    """Mount flags cannot hide every dynamic /proc/<pid>/task/<tid>/mem alias.

    The boundary therefore exposes an empty conventional directory, not procfs.  This closes the
    write-to-existing-code-page route around the no-new-executable-memory seccomp filter.
    """
    monkeypatch.setattr(grade, "_UNPRIV", (1234, 1235))
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade.os, "geteuid", lambda: 0)
    monkeypatch.setattr(grade.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(grade.os, "access", lambda path, mode: True)

    argv = grade._sandbox_argv(
        [grade._RUNTIME_PYTHON, "-c", "pass"],
        tmp_path / "submission",
        tmp_path / "input.vcf",
        tmp_path / "output.tsv",
    )
    pairs = [argv[i:i + 2] for i in range(len(argv) - 1)]

    assert ["--dir", "/proc"] in pairs
    assert ["--proc", "/proc"] not in pairs


def test_submission_limits_are_all_mandatory(monkeypatch):
    applied = []
    monkeypatch.setattr(grade.resource, "setrlimit", lambda resource_id, value: applied.append((resource_id, value)))

    grade._apply_submission_limits(17)()

    resources = {resource_id for resource_id, _ in applied}
    assert resources == {
        grade.resource.RLIMIT_AS,
        grade.resource.RLIMIT_FSIZE,
        grade.resource.RLIMIT_NPROC,
        grade.resource.RLIMIT_NOFILE,
        grade.resource.RLIMIT_CPU,
        grade.resource.RLIMIT_CORE,
    }
    assert all(soft == hard for _, (soft, hard) in applied)
    assert dict(applied)[grade.resource.RLIMIT_CPU] == (146, 146)


def test_cpu_guard_scales_the_wall_budget_by_the_disclosed_thread_count():
    assert grade._SUBMISSION_NUMERICAL_THREADS == 8
    assert grade._submission_cpu_limit_seconds(17) == 17 * 8 + 10
    with pytest.raises(ValueError, match="positive and finite"):
        grade._submission_cpu_limit_seconds(0)


def test_cwd_plant_cannot_hijack_grader_under_safe_path():
    """``python -m grader.grade`` searches the cwd before PYTHONPATH, so a planted cwd/grader is a
    root-code-exec vector; PYTHONSAFEPATH=1 (set in test.sh) must neutralise it while the real
    grader still resolves from grader_pkg."""
    import os
    import subprocess
    import tempfile

    pkg = grade._grader_package_root()
    assert pkg is not None
    plant = Path(tempfile.mkdtemp(prefix="pcabench_hijack_"))
    _TMP_ROOTS.append(plant)
    (plant / "grader").mkdir()
    (plant / "grader" / "__init__.py").write_text("")
    (plant / "grader" / "grade.py").write_text("import sys; sys.stderr.write('HIJACKED'); "
                                               "raise SystemExit(0)\n")

    base = dict(os.environ)
    base["PYTHONPATH"] = str(pkg) + os.pathsep + base.get("PYTHONPATH", "")

    # Unsafe (cwd searched first): the plant wins -- documents the vulnerability.
    unsafe = dict(base)
    unsafe.pop("PYTHONSAFEPATH", None)
    r_unsafe = subprocess.run([sys.executable, "-m", "grader.grade", "--help"],
                              cwd=str(plant), env=unsafe, capture_output=True, text=True)
    assert "HIJACKED" in r_unsafe.stderr

    # Safe path: cwd dropped from sys.path -> the real grader resolves, plant is ignored.
    safe = dict(base)
    safe["PYTHONSAFEPATH"] = "1"
    r_safe = subprocess.run([sys.executable, "-m", "grader.grade", "--help"],
                            cwd=str(plant), env=safe, capture_output=True, text=True)
    assert "HIJACKED" not in r_safe.stderr
    assert "usage" in (r_safe.stdout + r_safe.stderr).lower()


# small local temp-dir helper so the scan test doesn't depend on the tmp_path fixture ordering
_TMP_ROOTS: list = []


def tmpdir_for(name: str) -> str:
    import tempfile
    d = Path(tempfile.mkdtemp(prefix=f"pcabench_{name}_"))
    _TMP_ROOTS.append(d)
    return str(d)


def teardown_module(_module):
    import shutil
    for d in _TMP_ROOTS:
        shutil.rmtree(d, ignore_errors=True)
