import errno
import gzip
import json
import mmap
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import grader.grade as grade
import grader.resource_watchdog as resource_watchdog
from grader.grade import (InvalidSubmission, _MAX_SCORES_BYTES,
                          _invalid_submission_result, _normalize_vcf, _stage_input,
                          _stage_submission, grade_suite, run_submission)


VCF = (
    b"##fileformat=VCFv4.2\n"
    b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
    b"1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t1/1\n"
)
MATH_KEY = bytes(range(32))


def test_watchdog_applies_the_cpu_limit_supplied_by_the_trusted_grader(monkeypatch):
    applied = []
    monkeypatch.setattr(
        resource_watchdog.resource,
        "setrlimit",
        lambda resource_id, value: applied.append((resource_id, value)),
    )
    resource_watchdog._apply_limits(SimpleNamespace(
        address_space_limit=1,
        file_size_limit=2,
        process_limit=3,
        open_file_limit=4,
        cpu_time_limit=146,
    ))
    assert dict(applied)[resource_watchdog.resource.RLIMIT_CPU] == (146, 146)


def immutable_snapshot(tmp_path, submission):
    snapshot = tmp_path / f"snapshot-{len(list(tmp_path.glob('snapshot-*')))}"
    _stage_submission(submission, snapshot)
    return snapshot

production_sandbox = pytest.mark.skipif(
    not (
        sys.platform.startswith("linux")
        and hasattr(os, "geteuid")
        and os.geteuid() == 0
        and Path("/usr/bin/bwrap").is_file()
        and Path("/opt/hyperfocal/pcabench/bin/python").is_file()
        and grade._UNPRIV is not None
    ),
    reason="requires the provisioned production sandbox",
)


def test_gzip_is_normalized_by_magic_to_cached_plain_vcf(tmp_path):
    source = tmp_path / "opaque.data"
    with gzip.open(source, "wb") as fh:
        fh.write(VCF)
    workdir = tmp_path / "work"

    first = _normalize_vcf(source, workdir)
    second = _normalize_vcf(source, workdir)

    assert first == second
    assert first.suffix == ".vcf"
    assert first.read_bytes() == VCF
    assert first.stat().st_mode & 0o222 == 0


def test_gzip_cache_invalidates_when_equal_size_source_is_rewritten(tmp_path):
    source = tmp_path / "source.gz"
    source.write_bytes(gzip.compress(b"A" * 100, compresslevel=1, mtime=0))
    original_times = source.stat()
    workdir = tmp_path / "work"
    first = _normalize_vcf(source, workdir)

    source.write_bytes(gzip.compress(b"B" * 100, compresslevel=1, mtime=0))
    os.utime(source, ns=(original_times.st_atime_ns, original_times.st_mtime_ns))
    second = _normalize_vcf(source, workdir)

    assert first != second
    assert first.read_bytes() == b"A" * 100
    assert second.read_bytes() == b"B" * 100


def test_deployed_staging_uses_distinct_metadata_normalized_clone(tmp_path, monkeypatch):
    source = tmp_path / "source.vcf"
    source.write_bytes(VCF)
    os.chmod(source, 0o444)
    destination = tmp_path / "opaque.vcf"
    monkeypatch.setattr(grade, "_UNPRIV", (65534, 65534))

    _stage_input(str(source), destination)

    source_stat = source.stat()
    staged_stat = destination.stat()
    assert source_stat.st_ino != staged_stat.st_ino
    assert staged_stat.st_nlink == 1
    assert staged_stat.st_size == source_stat.st_size
    assert staged_stat.st_atime_ns == grade._NORMALIZED_INPUT_TIME_NS
    assert staged_stat.st_mtime_ns == grade._NORMALIZED_INPUT_TIME_NS
    assert staged_stat.st_mode & 0o222 == 0


@production_sandbox
def test_submission_gets_private_readable_inode_and_cannot_corrupt_source(tmp_path):
    source = tmp_path / "private.vcf"
    source.write_bytes(VCF)
    os.chmod(source, 0o444)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import os, pathlib, sys\n"
        "staged = pathlib.Path(sys.argv[1])\n"
        "try:\n"
        " os.chmod(staged, 0o600); staged.write_bytes(b'corrupted')\n"
        "except OSError:\n"
        " pass\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\n')\n"
    )
    out = tmp_path / "scores.tsv"
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(snapshot, source, 1, out, isolation="bwrap")

    assert result["returncode"] == 0
    assert source.read_bytes() == VCF
    assert out.read_text() == "sample_id\tPC1\n"
    assert not list(tmp_path.glob("_sbx_*"))


@production_sandbox
def test_empty_proc_boundary_preserves_numeric_stack_threads_and_fork(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import multiprocessing, os, pathlib, sys\n"
        "import numpy as np\n"
        "from scipy.linalg import eigh\n"
        "from scipy.sparse import eye\n"
        "from scipy.sparse.linalg import eigsh\n"
        "def worker():\n"
        " a=np.arange(36,dtype=float).reshape(6,6)\n"
        " if np.linalg.svd(a,compute_uv=False)[0] <= 0: os._exit(31)\n"
        " if eigh(a@a.T,eigvals_only=True)[-1] <= 0: os._exit(32)\n"
        " if eigsh(eye(20),k=2,return_eigenvectors=False).shape != (2,): os._exit(33)\n"
        "if os.listdir('/proc'): raise SystemExit(40)\n"
        "for path in ('/proc/self/mem','/proc/thread-self/mem','/proc/1/mem'):\n"
        " if os.path.exists(path): raise SystemExit(41)\n"
        "process=multiprocessing.get_context('fork').Process(target=worker)\n"
        "process.start(); process.join()\n"
        "if process.exitcode != 0: raise SystemExit(42)\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\ns1\\t0\\ns2\\t0\\n')\n"
    )
    out = tmp_path / "scores.tsv"
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(snapshot, source, 1, out, isolation="bwrap")

    assert result["returncode"] == 0
    assert out.read_text() == "sample_id\tPC1\ns1\t0\ns2\t0\n"


@production_sandbox
def test_production_sandbox_preserves_fork_pool_and_queue(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import multiprocessing, pathlib, sys\n"
        "def send(queue): queue.put(23)\n"
        "context=multiprocessing.get_context('fork')\n"
        "queue=context.Queue()\n"
        "process=context.Process(target=send,args=(queue,))\n"
        "process.start()\n"
        "if queue.get(timeout=5) != 23: raise SystemExit(51)\n"
        "process.join()\n"
        "if process.exitcode != 0: raise SystemExit(52)\n"
        "queue.close(); queue.join_thread()\n"
        "with context.Pool(2) as pool:\n"
        " if pool.map(abs,[-4,-1,2]) != [4,1,2]: raise SystemExit(53)\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\ns1\\t0\\ns2\\t0\\n')\n"
    )
    out = tmp_path / "scores.tsv"
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(snapshot, source, 1, out, isolation="bwrap")

    assert result["returncode"] == 0, result
    assert out.read_text() == "sample_id\tPC1\ns1\t0\ns2\t0\n"


@production_sandbox
def test_guarded_solver_cannot_unshare_user_namespace_but_completes(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import os, pathlib, sys\n"
        "try:\n"
        " os.unshare(os.CLONE_NEWUSER)\n"
        "except OSError as error:\n"
        " if error.errno != 1: raise SystemExit(62)\n"
        "else:\n"
        " raise SystemExit(61)\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\ns1\\t0\\ns2\\t0\\n')\n"
    )
    out = tmp_path / "scores.tsv"
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(snapshot, source, 1, out, isolation="bwrap")

    assert result["returncode"] == 0, result
    assert out.read_text() == "sample_id\tPC1\ns1\t0\ns2\t0\n"


def test_trusted_input_staging_failure_is_infra_and_cleans_sandbox(tmp_path, monkeypatch):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("")

    def fail_stage(source_path, destination_path):
        raise OSError("synthetic staging failure")

    monkeypatch.setattr("grader.grade._stage_input", fail_stage)
    monkeypatch.setattr("grader.grade._bwrap_work_root", lambda: tmp_path)
    snapshot = immutable_snapshot(tmp_path, submission)

    with pytest.raises(OSError, match="synthetic staging failure"):
        run_submission(
            snapshot, source, 1, tmp_path / "scores.tsv", isolation="bwrap",
        )

    assert not list(tmp_path.glob("_sbx_*"))


def test_watchdog_report_schema_is_private_and_strict():
    payload = json.dumps({
        "watchdog_pid": 123,
        "child_returncode": 0,
        "timed_out": False,
        "violation": None,
        "monitor_error": None,
        "peak_pss_bytes": 4096,
        "peak_storage_bytes": 8192,
    }).encode()

    assert grade._decode_watchdog_report(payload, 123)["peak_pss_bytes"] == 4096
    with pytest.raises(RuntimeError, match="wrong process identity"):
        grade._decode_watchdog_report(payload, 124)
    with pytest.raises(RuntimeError, match="invalid schema"):
        grade._decode_watchdog_report(b'{"watchdog_pid":123}', 123)


def test_watchdog_monitor_failure_is_grader_infrastructure():
    report = {
        "child_returncode": 0,
        "timed_out": False,
        "violation": None,
        "monitor_error": "ResourceInspectionError: synthetic procfs loss",
        "peak_pss_bytes": 4096,
        "peak_storage_bytes": 8192,
    }

    with pytest.raises(RuntimeError, match="submission resource monitor failed"):
        grade._assert_resource_monitor_complete(report)

    report["monitor_error"] = None
    grade._assert_resource_monitor_complete(report)


def test_watchdog_storage_counts_only_submission_owned_physical_blocks(tmp_path, monkeypatch):
    root = tmp_path / "chroot"
    for relative in ("tmp", "var/tmp", "run/lock", "dev/shm", "dev/mqueue", "work"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    owned = root / "tmp" / "owned.bin"
    owned.write_bytes(b"x" * 8192)
    other = root / "tmp" / "other.bin"
    other.write_bytes(b"y" * 8192)
    # Distinguish the root-owned fixture without requiring chown privileges.
    monkeypatch.setattr(resource_watchdog.os, "lstat", lambda path: (
        type("Info", (), {
            "st_uid": os.getuid() + 1,
            "st_blocks": 16,
            "st_dev": 1,
            "st_ino": 2,
        })()
        if Path(path) == other else os.stat(path, follow_symlinks=False)
    ))

    blocks, entries = resource_watchdog._storage_usage(
        (root,), os.getuid(), {},
        storage_limit=grade._MAX_WRITABLE_STORAGE_BYTES,
        entry_limit=grade._MAX_WRITABLE_ENTRIES,
    )

    assert entries >= 1
    assert blocks >= owned.stat().st_blocks * 512


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux host procfs")
def test_watchdog_storage_counts_unlinked_open_files(tmp_path):
    root = tmp_path / "scratch"
    root.mkdir()
    hidden = root / "unlinked.bin"
    descriptor = os.open(hidden, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"x" * 8192)
        expected_blocks = os.fstat(descriptor).st_blocks * 512
        hidden.unlink()
        blocks, entries = resource_watchdog._storage_usage(
            (root,), os.getuid(), {os.getpid(): "R"},
            storage_limit=grade._MAX_WRITABLE_STORAGE_BYTES,
            entry_limit=grade._MAX_WRITABLE_ENTRIES,
        )

        assert entries == 0
        assert blocks >= expected_blocks > 0
    finally:
        os.close(descriptor)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux host procfs")
def test_watchdog_storage_counts_unlinked_mappings_after_descriptor_close(tmp_path):
    root = tmp_path / "scratch"
    root.mkdir()
    hidden = root / "mapped.bin"
    descriptor = os.open(hidden, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.ftruncate(descriptor, 8192)
    mapping = mmap.mmap(descriptor, 8192, access=mmap.ACCESS_WRITE)
    try:
        mapping[:] = b"x" * 8192
        mapping.flush()
        expected_blocks = os.fstat(descriptor).st_blocks * 512
        os.close(descriptor)
        descriptor = -1
        hidden.unlink()
        blocks, entries = resource_watchdog._storage_usage(
            (root,), os.getuid(), {os.getpid(): "R"},
            storage_limit=grade._MAX_WRITABLE_STORAGE_BYTES,
            entry_limit=grade._MAX_WRITABLE_ENTRIES,
        )

        assert entries == 0
        assert blocks >= expected_blocks > 0
    finally:
        mapping.close()
        if descriptor >= 0:
            os.close(descriptor)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux host procfs")
def test_watchdog_procfs_visibility_fails_closed(monkeypatch):
    original = resource_watchdog.Path.read_text

    def deny_smaps(path, *args, **kwargs):
        if str(path).endswith("/smaps_rollup"):
            raise PermissionError(errno.EACCES, "synthetic ptrace denial")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(resource_watchdog.Path, "read_text", deny_smaps)
    with pytest.raises(resource_watchdog.ResourceInspectionError, match="smaps_rollup"):
        resource_watchdog._aggregate_pss_bytes({os.getpid(): "R"})


def test_watchdog_empty_smaps_skips_a_process_that_became_zombie(monkeypatch):
    original = resource_watchdog.Path.read_text

    def exiting_process(path, *args, **kwargs):
        if str(path) == "/proc/123/smaps_rollup":
            return ""
        if str(path) == "/proc/123/status":
            return "Name:\tpython\nState:\tZ (zombie)\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(resource_watchdog.Path, "read_text", exiting_process)

    assert resource_watchdog._aggregate_pss_bytes({123: "R"}) == 0


def test_watchdog_empty_smaps_skips_a_process_that_disappeared(monkeypatch):
    original = resource_watchdog.Path.read_text

    def disappearing_process(path, *args, **kwargs):
        if str(path) == "/proc/123/smaps_rollup":
            return ""
        if str(path) == "/proc/123/status":
            raise FileNotFoundError(errno.ENOENT, "synthetic process exit")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(resource_watchdog.Path, "read_text", disappearing_process)

    assert resource_watchdog._aggregate_pss_bytes({123: "R"}) == 0


def test_watchdog_empty_smaps_retries_a_live_process(monkeypatch):
    original = resource_watchdog.Path.read_text
    reads = 0

    def transient_empty(path, *args, **kwargs):
        nonlocal reads
        if str(path) == "/proc/123/smaps_rollup":
            reads += 1
            return "" if reads == 1 else "Pss:                17 kB\n"
        if str(path) == "/proc/123/status":
            return "Name:\tpython\nState:\tR (running)\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(resource_watchdog.Path, "read_text", transient_empty)

    assert resource_watchdog._aggregate_pss_bytes({123: "R"}) == 17 * 1024
    assert reads == 2


def test_watchdog_empty_smaps_for_a_live_process_fails_closed(monkeypatch):
    original = resource_watchdog.Path.read_text

    def live_without_accounting(path, *args, **kwargs):
        if str(path) == "/proc/123/smaps_rollup":
            return ""
        if str(path) == "/proc/123/status":
            return "Name:\tpython\nState:\tS (sleeping)\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(resource_watchdog.Path, "read_text", live_without_accounting)

    with pytest.raises(resource_watchdog.ResourceInspectionError, match="live process"):
        resource_watchdog._aggregate_pss_bytes({123: "R"})


def test_watchdog_fd_scan_retries_a_transient_live_exec_denial(monkeypatch):
    class FakeScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            entry = type("Entry", (), {"name": "7"})()
            return iter((entry,))

    expected = Path(__file__).stat()
    original_stat = os.stat
    scans = 0
    sleeps = []

    def transitioning_scandir(path):
        nonlocal scans
        assert path == "/proc/123/fd"
        scans += 1
        if scans == 1:
            raise PermissionError(errno.EACCES, "synthetic userns transition")
        return FakeScandir()

    monkeypatch.setattr(resource_watchdog.os, "scandir", transitioning_scandir)
    monkeypatch.setattr(resource_watchdog.os, "stat", lambda path, **kwargs: (
        expected if path == "/proc/123/fd/7" else original_stat(path, **kwargs)
    ))
    monkeypatch.setattr(resource_watchdog, "_process_state", lambda _pid: "R")
    monkeypatch.setattr(resource_watchdog.time, "sleep", sleeps.append)

    assert resource_watchdog._submission_fd_stats(123) == [expected]
    assert scans == 2
    assert sleeps == [0.0]


def test_watchdog_fd_entry_stat_retries_a_transient_live_exec_denial(monkeypatch):
    class FakeScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            entry = type("Entry", (), {"name": "9"})()
            return iter((entry,))

    expected = Path(__file__).stat()
    stats = 0

    def transitioning_stat(path, **_kwargs):
        nonlocal stats
        assert path == "/proc/123/fd/9"
        stats += 1
        if stats == 1:
            raise PermissionError(errno.EACCES, "synthetic exec transition")
        return expected

    monkeypatch.setattr(resource_watchdog.os, "scandir", lambda _path: FakeScandir())
    monkeypatch.setattr(resource_watchdog.os, "stat", transitioning_stat)
    monkeypatch.setattr(resource_watchdog, "_process_state", lambda _pid: "S")
    monkeypatch.setattr(resource_watchdog.time, "sleep", lambda _delay: None)

    assert resource_watchdog._submission_fd_stats(123) == [expected]
    assert stats == 2


@pytest.mark.parametrize("final_state", [None, "Z"])
def test_watchdog_fd_denial_skips_only_an_exited_process(monkeypatch, final_state):
    scans = 0

    def denied_scandir(_path):
        nonlocal scans
        scans += 1
        raise PermissionError(errno.EACCES, "synthetic exiting transition")

    monkeypatch.setattr(resource_watchdog.os, "scandir", denied_scandir)
    monkeypatch.setattr(resource_watchdog, "_process_state", lambda _pid: final_state)

    assert resource_watchdog._submission_fd_stats(123) == []
    assert scans == 1


def test_watchdog_fd_denial_for_a_persistently_live_process_fails_closed(monkeypatch):
    scans = 0
    sleeps = []

    def denied_scandir(_path):
        nonlocal scans
        scans += 1
        raise PermissionError(errno.EACCES, "synthetic persistent denial")

    monkeypatch.setattr(resource_watchdog.os, "scandir", denied_scandir)
    monkeypatch.setattr(resource_watchdog, "_process_state", lambda _pid: "S")
    monkeypatch.setattr(resource_watchdog.time, "sleep", sleeps.append)

    with pytest.raises(resource_watchdog.ResourceInspectionError, match="live process 123"):
        resource_watchdog._submission_fd_stats(123)
    assert scans == 5
    assert sleeps == list(resource_watchdog._FD_PERMISSION_RETRY_DELAYS)


def test_watchdog_fd_retry_deadline_is_shared_across_many_entries(monkeypatch):
    class FakeScandir:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(type("Entry", (), {"name": str(index)})() for index in range(20))

    expected = Path(__file__).stat()
    clock = [100.0]
    sleeps = []
    stat_attempts = {}

    def transient_stat(path, **_kwargs):
        stat_attempts[path] = stat_attempts.get(path, 0) + 1
        if stat_attempts[path] == 1:
            raise PermissionError(errno.EACCES, "synthetic per-entry transition")
        return expected

    def advance_clock(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(resource_watchdog.os, "scandir", lambda _path: FakeScandir())
    monkeypatch.setattr(resource_watchdog.os, "stat", transient_stat)
    monkeypatch.setattr(resource_watchdog, "_process_state", lambda _pid: "S")
    monkeypatch.setattr(resource_watchdog.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(resource_watchdog.time, "sleep", advance_clock)

    with pytest.raises(resource_watchdog.ResourceInspectionError, match="live process 123"):
        resource_watchdog._submission_fd_stats(123)

    assert sum(sleeps) == pytest.approx(0.15)
    assert len(stat_attempts) == 5
    assert sum(stat_attempts.values()) == 9


def test_watchdog_deleted_mapping_fallback_deduplicates_extents(monkeypatch):
    maps = (
        "00001000-00002000 rw-s 00000000 08:01 77 /tmp/mapped (deleted)\n"
        "00003000-00004000 rw-s 00001000 08:01 77 /tmp/mapped (deleted)\n"
        "00005000-00007000 rw-p 00000000 00:00 0 [heap]\n"
        "00008000-00009000 r--p 00000000 08:01 88 /tmp/named\n"
    )
    original = resource_watchdog.Path.read_text

    def fake_maps(path, *args, **kwargs):
        if str(path) == "/proc/123/maps":
            return maps
        return original(path, *args, **kwargs)

    monkeypatch.setattr(resource_watchdog.Path, "read_text", fake_maps)
    identity = (os.makedev(0x08, 0x01), 77)

    assert resource_watchdog._deleted_mapped_extents({123}, set()) == {
        identity: 8192,
    }
    assert resource_watchdog._deleted_mapped_extents({123}, {identity}) == {}


@production_sandbox
def test_same_uid_watchdog_reports_real_pss_through_bwrap(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\ns1\\t0\\ns2\\t0\\n')\n"
    )
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(
        snapshot, source, 1, tmp_path / "scores.tsv", isolation="bwrap",
    )

    assert result["returncode"] == 0, result
    assert result["peak_pss_bytes"] > 0


@production_sandbox
def test_watchdog_repeated_short_lived_bwrap_jobs_do_not_false_fail(tmp_path):
    """Exercise the exit/procfs race repeatedly through the production launcher."""
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\ns1\\t0\\ns2\\t0\\n')\n"
    )
    snapshot = immutable_snapshot(tmp_path, submission)

    for repetition in range(12):
        result = run_submission(
            snapshot,
            source,
            1,
            tmp_path / f"scores-{repetition}.tsv",
            isolation="bwrap",
        )
        assert result["returncode"] == 0, (repetition, result)


@production_sandbox
def test_watchdog_repeated_numeric_bwrap_exec_transitions_do_not_false_fail(tmp_path):
    """Keep each numeric child alive across polls while repeatedly exercising bwrap exec."""
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import pathlib, sys, time\n"
        "import numpy as np\n"
        "from scipy.linalg import eigh\n"
        "x = np.arange(384 * 96, dtype=np.float64).reshape(384, 96)\n"
        "for shift in range(3):\n"
        " gram = (x + shift).T @ (x + shift)\n"
        " eigh(gram, subset_by_index=[92, 95], check_finite=False)\n"
        "time.sleep(0.45)\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\ns1\\t0\\ns2\\t0\\n')\n"
    )
    snapshot = immutable_snapshot(tmp_path, submission)

    for repetition in range(6):
        result = run_submission(
            snapshot,
            source,
            1,
            tmp_path / f"numeric-scores-{repetition}.tsv",
            isolation="bwrap",
        )
        assert result["returncode"] == 0, (repetition, result)
        assert result["peak_pss_bytes"] > 0


@production_sandbox
def test_same_uid_watchdog_enforces_memory_through_bwrap(tmp_path, monkeypatch):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import time\n"
        "payload = bytearray(64 * 1024 * 1024)\n"
        "time.sleep(10)\n"
    )
    snapshot = immutable_snapshot(tmp_path, submission)
    monkeypatch.setattr(grade, "_MAX_AGGREGATE_RSS_BYTES", 32 * 1024 * 1024)

    result = run_submission(
        snapshot, source, 1, tmp_path / "scores.tsv", isolation="bwrap", timeout=20,
    )

    assert result["returncode"] == 137, result
    assert "aggregate resident-memory limit exceeded" in result["stderr"]
    assert result["peak_pss_bytes"] > 32 * 1024 * 1024


@production_sandbox
def test_same_uid_watchdog_counts_open_unlinked_storage_through_bwrap(
    tmp_path, monkeypatch,
):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import os, time\n"
        "path = os.path.join(os.environ['TMPDIR'], 'hidden')\n"
        "handle = open(path, 'w+b')\n"
        "handle.write(b'x' * (2 * 1024 * 1024)); handle.flush()\n"
        "os.unlink(path)\n"
        "time.sleep(10)\n"
    )
    snapshot = immutable_snapshot(tmp_path, submission)
    monkeypatch.setattr(grade, "_MAX_WRITABLE_STORAGE_BYTES", 1024 * 1024)

    result = run_submission(
        snapshot, source, 1, tmp_path / "scores.tsv", isolation="bwrap", timeout=20,
    )

    assert result["returncode"] == 137, result
    assert "aggregate temporary-storage limit exceeded" in result["stderr"]
    assert result["peak_storage_bytes"] > 1024 * 1024


@production_sandbox
def test_same_uid_watchdog_counts_mmap_unlinked_storage_through_bwrap(
    tmp_path, monkeypatch,
):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import mmap, os, time\n"
        "path = os.path.join(os.environ['TMPDIR'], 'mapped')\n"
        "fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)\n"
        "os.ftruncate(fd, 2 * 1024 * 1024)\n"
        "view = mmap.mmap(fd, 2 * 1024 * 1024, access=mmap.ACCESS_WRITE)\n"
        "view[:] = b'x' * (2 * 1024 * 1024); view.flush()\n"
        "os.close(fd); os.unlink(path)\n"
        "for candidate in range(3, 256):\n"
        " try:\n"
        "  os.close(candidate)\n"
        " except OSError:\n"
        "  pass\n"
        "time.sleep(10)\n"
    )
    snapshot = immutable_snapshot(tmp_path, submission)
    monkeypatch.setattr(grade, "_MAX_WRITABLE_STORAGE_BYTES", 1024 * 1024)

    result = run_submission(
        snapshot, source, 1, tmp_path / "scores.tsv", isolation="bwrap", timeout=20,
    )

    assert result["returncode"] == 137, result
    assert "aggregate temporary-storage limit exceeded" in result["stderr"]
    assert result["peak_storage_bytes"] > 1024 * 1024


@production_sandbox
def test_symlink_output_is_rejected_without_following_it(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import os, sys\n"
        "os.symlink('/dev/zero', sys.argv[3])\n"
    )
    out = tmp_path / "scores.tsv"
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(snapshot, source, 1, out, isolation="bwrap")

    assert result["returncode"] != 0
    assert not out.exists()


@production_sandbox
def test_oversized_sparse_output_is_rejected_before_copy(tmp_path):
    """An oversized scores file never reaches the grader -- now via a graceful read-side rejection.

    This used to be enforced by RLIMIT_FSIZE killing the writer mid-call, which also capped every
    TEMP file at 64 MiB and contradicted the prompt's promised 4 GiB of temporary storage (it killed
    a legitimate memmap spill with a short write). The per-file ceiling is now the storage budget, so
    the submission is ALLOWED to produce the oversized file and exits 0 -- and the size check in
    ``_copy_bounded_regular`` rejects it before it is ever copied. The load-bearing property, that an
    oversized output cannot reach the grader, is unchanged; only the mechanism moved from a signal to
    a stated reason.
    """
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import sys\n"
        f"open(sys.argv[3], 'wb').truncate({_MAX_SCORES_BYTES + 1})\n"
    )
    out = tmp_path / "scores.tsv"
    snapshot = immutable_snapshot(tmp_path, submission)

    result = run_submission(snapshot, source, 1, out, isolation="bwrap")

    assert result["returncode"] == 0
    assert not out.exists()


def test_submission_symlink_is_rejected_before_execution(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("raise SystemExit(99)\n")
    (submission / "leak").symlink_to(tmp_path / "secret")

    try:
        _stage_submission(submission, tmp_path / "staged")
    except ValueError as error:
        assert "symlink" in str(error)
    else:
        raise AssertionError("submission symlink was accepted")


def test_missing_entrypoint_is_a_failed_submission_not_infra(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()

    with pytest.raises(InvalidSubmission, match="regular pca"):
        _stage_submission(submission, tmp_path / "staged")


def test_snapshot_io_failure_remains_an_infrastructure_error(tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("print('ok')\n")

    def fail_open(*args, **kwargs):
        raise OSError(errno.EIO, "I/O")

    monkeypatch.setattr(grade.os, "open", fail_open)

    with pytest.raises(OSError) as error:
        _stage_submission(submission, tmp_path / "staged")
    assert error.value.errno == errno.EIO


def test_invalid_submission_result_is_a_complete_zero_contract(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "case.vcf").write_bytes(VCF)
    (data / "case.vcf.truth.json").write_text(json.dumps({
        "spec": {"category": "continental", "weight": 1.2, "k": 5},
    }))

    result = _invalid_submission_result(
        data, "missing entrypoint", isolation="bwrap",
    )

    assert result["submission_status"] == "failed"
    assert result["submission_error"] == "missing entrypoint"
    assert result["reward"] == 0
    assert result["reward_detail"]["category_scores"] == {"continental": 0.0}
    assert result["reward_detail"]["per_dataset"] == [{
        "dataset": "case",
        "category": "continental",
        "k": 5,
        "weight": 1.2,
        "reward": 0.0,
    }]


def test_grade_suite_maps_missing_entrypoint_to_zero_not_exception(tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "case.vcf").write_bytes(VCF)
    (data / "case.vcf.truth.json").write_text(json.dumps({
        "spec": {"category": "continental", "weight": 1.0, "k": 1},
    }))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(grade, "_UNPRIV", (65534, 65534))
    monkeypatch.setattr(grade, "_BWRAP", sys.executable)
    monkeypatch.setattr(grade, "_RUNTIME_PYTHON", sys.executable)
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade, "_seal_grader_package", lambda: None)
    monkeypatch.setattr(grade, "_assert_grader_package_sealed", lambda: None)
    monkeypatch.setattr(grade, "_protect_grader_directory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        grade, "_assert_execution_boundary", lambda _workdir, _isolation: None,
    )
    # Calibration runs a no-op submission through the real sandbox; this test fakes that sandbox.
    monkeypatch.setattr(grade, "measure_execution_overhead", lambda *args, **kwargs: 0.0)

    result = grade_suite(
        submission, data, work, isolation="bwrap", math_key=MATH_KEY,
    )

    assert result["submission_status"] == "failed"
    assert result["reward"] == 0
    assert "regular pca" in result["submission_error"]


def test_openat_snapshot_rejects_symlink_swap_without_copying_secret(tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("print('ok')\n")
    helper = submission / "helper.py"
    helper.write_text("SAFE = True\n")
    secret = tmp_path / "secret"
    secret.write_text("PRIVATE GRADER DATA\n")
    original_open = grade.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "helper.py" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            helper.unlink()
            helper.symlink_to(secret)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(grade.os, "open", racing_open)
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="symlink|changed"):
        _stage_submission(submission, destination)
    assert not destination.exists()


def test_snapshot_is_immutable_after_live_submission_changes(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    entry = submission / "pca"
    entry.write_text("print('first')\n")
    snapshot = immutable_snapshot(tmp_path, submission)

    entry.write_text("print('second')\n")

    assert (snapshot / "pca").read_text() == "print('first')\n"
    assert snapshot.stat().st_mode & 0o222 == 0
    assert (snapshot / "pca").stat().st_mode & 0o222 == 0


def test_snapshot_strips_interpreter_bytecode_cache_but_keeps_source(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("#!/usr/bin/env python3\nimport helper\n")
    (submission / "helper.py").write_text("VALUE = 1\n")
    cache = submission / "__pycache__"
    cache.mkdir()
    (cache / "helper.cpython-312.pyc").write_bytes(b"opaque-bytecode")

    snapshot = tmp_path / "snapshot"
    _stage_submission(submission, snapshot)

    assert (snapshot / "helper.py").read_text() == "VALUE = 1\n"
    assert not (snapshot / "__pycache__").exists()


def test_snapshot_breaks_solver_controlled_hardlink_identity(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("#!/usr/bin/env python3\n")
    first = submission / "first.dat"
    second = submission / "second.dat"
    first.write_bytes(b"ordinary-data")
    os.link(first, second)
    assert first.stat().st_ino == second.stat().st_ino

    snapshot = tmp_path / "snapshot"
    _stage_submission(submission, snapshot)

    assert (snapshot / "first.dat").read_bytes() == b"ordinary-data"
    assert (snapshot / "second.dat").read_bytes() == b"ordinary-data"
    assert (snapshot / "first.dat").stat().st_ino != (snapshot / "second.dat").stat().st_ino


def test_snapshot_rejects_excessive_directory_depth_as_submission_error(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("print('ok')\n")
    current = submission
    for index in range(grade._MAX_SUBMISSION_DEPTH + 2):
        current = current / f"d{index}"
        current.mkdir()

    destination = tmp_path / "snapshot"
    with pytest.raises(InvalidSubmission, match="directory-depth"):
        _stage_submission(submission, destination)
    assert not destination.exists()


def test_detached_submission_processes_are_stopped_then_killed(monkeypatch):
    snapshots = iter((
        {701: "R"},
        {701: "T"},
        {701: "T"},
        {},
    ))
    signals = []
    monkeypatch.setattr(grade, "_UNPRIV", (1234, 1234))
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade, "_submission_processes", lambda: next(snapshots))
    monkeypatch.setattr(grade.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(grade.time, "sleep", lambda _seconds: None)

    grade._terminate_submission_processes()

    assert signals == [(701, signal.SIGSTOP), (701, signal.SIGKILL)]


def test_sysvipc_inventory_tracks_owner_and_creator_uid(tmp_path, monkeypatch):
    table = tmp_path / "msg"
    table.write_text(
        "key msqid perms cbytes qnum lspid lrpid uid gid cuid cgid stime rtime ctime\n"
        "0 41 600 0 0 0 0 1234 1234 1234 1234 0 0 0\n"
        "0 42 600 0 0 0 0 0 0 0 0 0 0 0\n"
    )
    monkeypatch.setattr(grade, "_UNPRIV", (1234, 1234))
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(
        grade, "_SYSVIPC_TABLES", {"msg": (table, "msqid", "msgctl")},
    )

    assert grade._submission_sysvipc() == {"msg": [41]}


@production_sandbox
def test_stderr_is_drained_but_memory_tail_is_bounded(tmp_path):
    source = tmp_path / "input.vcf"
    source.write_bytes(VCF)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text(
        "import pathlib, sys\n"
        "sys.stderr.write('x' * (5 * 1024 * 1024))\n"
        "pathlib.Path(sys.argv[3]).write_text('sample_id\\tPC1\\n')\n"
    )

    snapshot = immutable_snapshot(tmp_path, submission)
    result = run_submission(
        snapshot, source, 1, tmp_path / "scores.tsv", isolation="bwrap",
    )

    assert result["returncode"] == 0
    assert 0 < len(result["stderr"]) <= 2000
