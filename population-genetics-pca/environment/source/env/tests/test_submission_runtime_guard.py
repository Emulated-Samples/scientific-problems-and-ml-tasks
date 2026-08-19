from __future__ import annotations

import base64
import marshal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "grader" / "submission_runner.py"


def _run(tmp_path: Path, body: str):
    entry = tmp_path / "pca"
    entry.write_text(body)
    return subprocess.run(
        [sys.executable, "-I", str(RUNNER), str(entry), "argument"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )


def test_guard_runs_ordinary_pure_python(tmp_path):
    output = tmp_path / "result.txt"
    completed = _run(
        tmp_path,
        f"import pathlib,sys\npathlib.Path({str(output)!r}).write_text(sys.argv[1])\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "argument"


def test_guard_supports_sibling_modules_and_reviewed_native_stack(tmp_path):
    (tmp_path / "helper.py").write_text("VALUE = 7\n")
    output = tmp_path / "result.txt"
    completed = _run(
        tmp_path,
        textwrap.dedent(
            f"""
            import _struct
            import helper
            import math
            import mmap
            import pathlib
            import numpy as np
            from scipy.linalg import eigh
            from scipy.special import ndtr
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh

            dense = eigh(np.eye(4), subset_by_index=[2, 3])[0]
            singular = np.linalg.svd(np.eye(3), compute_uv=False)
            sparse = eigsh(csr_matrix(np.eye(5)), k=2)[0]
            special = ndtr(np.array([0.0]))
            data_map = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            data_map[0] = 9
            pathlib.Path({str(output)!r}).write_text(
                f"{{helper.VALUE}} {{math.sqrt(4)}} {{_struct.calcsize('i')}} "
                f"{{dense.size}} {{singular.size}} {{sparse.size}} {{special[0]}} {{data_map[0]}}"
            )
            """
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "7 2.0 4 2 3 2 0.5 9"


def test_guard_allows_lazy_reviewed_stdlib_source_and_native_dependencies(tmp_path):
    completed = _run(
        tmp_path,
        "import csv, email, statistics, fractions, decimal, sqlite3, urllib.parse\n"
        "assert csv.reader and email.message_from_string and statistics.mean\n"
        "assert fractions.Fraction(1, 2) and decimal.Decimal('1.5')\n"
        "assert sqlite3.connect and urllib.parse.urlsplit\n"
        "print('ok')\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_guard_allows_reviewed_dataclass_method_generation(tmp_path):
    completed = _run(
        tmp_path,
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Item:\n"
        "    value: int\n"
        "assert Item(7).value == 7\n"
        "print('ok')\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_guard_allows_valid_collections_namedtuple_generation(tmp_path):
    completed = _run(
        tmp_path,
        "from collections import namedtuple\n"
        "Point = namedtuple('Point', ['x', 'y'])\n"
        "assert Point(2, 3).x == 2\n"
        "print('ok')\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_guard_rejects_encoded_import_of_unapproved_pure_python_package(tmp_path):
    completed = _run(
        tmp_path,
        "import builtins\n"
        "getattr(builtins, '__im' + 'port__')('pytest')\n",
    )

    assert completed.returncode == 126
    assert "third-party import is disabled: pytest" in completed.stderr


def test_guard_rejects_str_subclass_import_path_without_dispatching_overrides(tmp_path):
    completed = _run(
        tmp_path,
        textwrap.dedent(
            """
            import sys

            class MisleadingPath(str):
                def lower(self):
                    return "harmless.py"

                def split(self, *args, **kwargs):
                    return ["trusted"]

                def startswith(self, *args, **kwargs):
                    return True

            sys.audit("import", "numpy.payload", MisleadingPath("/tmp/payload.so"))
            """
        ),
    )

    assert completed.returncode == 126
    assert "untrusted import-path type is disabled" in completed.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux extension-loader audit")
def test_guard_rejects_renamed_native_extension_via_create_dynamic(tmp_path):
    payload = tmp_path / "payload.bin"
    completed = _run(
        tmp_path,
        textwrap.dedent(
            f"""
            import _imp
            import importlib.machinery
            import shutil
            import numpy._core._multiarray_umath as extension

            shutil.copyfile(extension.__file__, {str(payload)!r})
            spec = importlib.machinery.ModuleSpec(
                "numpy._core._multiarray_umath", None, origin={str(payload)!r}
            )
            _imp.create_dynamic(spec)
            """
        ),
    )

    assert completed.returncode == 126
    assert "untrusted imported code path is disabled" in completed.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux executable-map filter")
def test_kernel_filter_rejects_encoded_native_extension_without_audit_hook(tmp_path):
    """The immutable kernel policy must hold even when no Python audit callback is installed."""
    root = Path(__file__).resolve().parents[1]
    payload = tmp_path / "encoded-payload.bin"
    code = textwrap.dedent(
        f"""
        import _imp
        import base64
        import importlib.machinery
        import pathlib
        import shutil
        import sys

        sys.path.insert(0, {str(root)!r})
        from grader.submission_runner import (
            _install_no_exec_filter,
            _install_no_new_executable_memory_filter,
            _preload_approved_native_stack,
        )

        _install_no_exec_filter()
        _preload_approved_native_stack()
        _install_no_new_executable_memory_filter()
        import numpy._core._multiarray_umath as approved_extension
        encoded = base64.b64encode(pathlib.Path(approved_extension.__file__).read_bytes())
        pathlib.Path({str(payload)!r}).write_bytes(base64.b64decode(encoded))
        spec = importlib.machinery.ModuleSpec(
            "numpy._core._multiarray_umath", None, origin={str(payload)!r}
        )
        try:
            _imp.create_dynamic(spec)
        except ImportError as error:
            assert "Operation not permitted" in str(error) or "failed to map" in str(error)
            print("blocked")
        else:
            raise SystemExit(9)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked"


def test_frame_chain_cannot_mutate_registered_audit_policy(tmp_path):
    completed = _run(
        tmp_path,
        textwrap.dedent(
            """
            import sys

            frame = sys._getframe()
            while frame is not None:
                for name in (
                    "_audit", "_AUDIT_BLOCKED_EVENTS", "_BLOCKED_EVENTS",
                    "_NATIVE_SUFFIXES", "_TRUSTED_NATIVE_ROOTS",
                ):
                    frame.f_globals.pop(name, None)
                frame = frame.f_back
            ctypes = __import__("ct" + "ypes")
            ctypes.CDLL(None)
            """
        ),
    )

    assert completed.returncode == 126
    assert "ctypes.dlopen" in completed.stderr


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires fork")
def test_guard_preserves_fork_only_multiprocessing(tmp_path):
    output = tmp_path / "result.txt"
    completed = _run(
        tmp_path,
        textwrap.dedent(
            f"""
            import multiprocessing
            import pathlib

            context = multiprocessing.get_context("fork")
            worker = context.Process(target=lambda: None)
            worker.start()
            worker.join()
            assert worker.exitcode == 0
            pathlib.Path({str(output)!r}).write_text("fork-ok")
            """
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "fork-ok"


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires fork")
def test_guard_names_ctypes_backed_multiprocessing_shared_objects(tmp_path):
    completed = _run(
        tmp_path,
        "import multiprocessing\n"
        "context = multiprocessing.get_context('fork')\n"
        "context.Array('q', 4)\n",
    )

    assert completed.returncode == 126
    assert completed.stderr == "ctypes-backed multiprocessing shared objects are disabled\n"


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires fork")
def test_guard_preserves_preloaded_numeric_stack_inside_fork_worker(tmp_path):
    output = tmp_path / "result.txt"
    completed = _run(
        tmp_path,
        textwrap.dedent(
            f"""
            import multiprocessing
            import pathlib

            def worker(path):
                import numpy as np
                from scipy.linalg import eigh
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh
                from scipy.special import ndtr

                dense = eigh(np.eye(4), subset_by_index=[2, 3])[0]
                sparse = eigsh(csr_matrix(np.eye(5)), k=2)[0]
                value = float(ndtr(np.array([0.0]))[0])
                pathlib.Path(path).write_text(f"{{dense.size}} {{sparse.size}} {{value}}")

            context = multiprocessing.get_context("fork")
            process = context.Process(target=worker, args=({str(output)!r},))
            process.start()
            process.join()
            assert process.exitcode == 0
            """
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "2 2 0.5"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux fork Pool/Queue guard")
def test_guard_preserves_fork_pool_and_queue_after_executable_map_closure(tmp_path):
    output = tmp_path / "result.txt"
    completed = _run(
        tmp_path,
        textwrap.dedent(
            f"""
            import multiprocessing
            import pathlib

            def send(queue):
                queue.put(17)

            context = multiprocessing.get_context("fork")
            queue = context.Queue()
            process = context.Process(target=send, args=(queue,))
            process.start()
            assert queue.get(timeout=5) == 17
            process.join()
            assert process.exitcode == 0
            queue.close()
            queue.join_thread()
            with context.Pool(2) as pool:
                assert pool.map(abs, [-3, -2, 1]) == [3, 2, 1]
            pathlib.Path({str(output)!r}).write_text("pool-queue-ok")
            """
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "pool-queue-ok"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="seccomp is Linux-only")
def test_kernel_filter_blocks_exec_but_preserves_fork(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(root)!r})
        from grader.submission_runner import _install_no_exec_filter

        _install_no_exec_filter()
        child = os.fork()
        if child == 0:
            os._exit(0)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        try:
            os.execv('/bin/true', ['/bin/true'])
        except PermissionError:
            print('blocked')
        else:
            raise SystemExit(9)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="seccomp is Linux-only")
def test_kernel_filter_blocks_observer_blinding_but_preserves_other_prctl(tmp_path):
    """Use ctypes only in this trusted harness; solver source never receives native-call access."""
    root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        f"""
        import ctypes
        import errno
        import sys
        sys.path.insert(0, {str(root)!r})
        from grader.submission_runner import _install_no_exec_filter

        libc = ctypes.CDLL(None, use_errno=True)
        libc.unshare.argtypes = [ctypes.c_int]
        libc.unshare.restype = ctypes.c_int
        libc.setns.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.setns.restype = ctypes.c_int
        libc.prctl.argtypes = [
            ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_ulong, ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int

        _install_no_exec_filter()

        ctypes.set_errno(0)
        assert libc.unshare(0) == -1
        assert ctypes.get_errno() == errno.EPERM
        ctypes.set_errno(0)
        assert libc.setns(-1, 0) == -1
        assert ctypes.get_errno() == errno.EPERM

        # PR_GET_DUMPABLE and PR_SET_DUMPABLE=1 remain legal; only the observer-blinding zero
        # value is rejected.
        assert libc.prctl(3, 0, 0, 0, 0) >= 0
        assert libc.prctl(4, 1, 0, 0, 0) == 0
        ctypes.set_errno(0)
        assert libc.prctl(4, 0, 0, 0, 0) == -1
        assert ctypes.get_errno() == errno.EPERM
        print("observer-protected")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "observer-protected"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="seccomp is Linux-only")
def test_kernel_filter_blocks_executable_mappings_but_allows_data_mappings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        f"""
        import ctypes
        import mmap
        import sys
        import threading
        sys.path.insert(0, {str(root)!r})
        from grader.submission_runner import (
            _install_no_exec_filter,
            _install_no_new_executable_memory_filter,
            _preload_approved_native_stack,
        )

        _install_no_exec_filter()
        _preload_approved_native_stack()

        release = threading.Event()
        result = []
        def existing_thread():
            release.wait()
            try:
                mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_EXEC)
            except PermissionError:
                result.append("thread-blocked")
            else:
                result.append("thread-escaped")

        thread = threading.Thread(target=existing_thread)
        thread.start()
        _install_no_new_executable_memory_filter()
        release.set()
        thread.join()
        assert result == ["thread-blocked"]

        data = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        data[0] = 7
        assert data[0] == 7
        address = ctypes.addressof(ctypes.c_char.from_buffer(data))
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.mprotect.restype = ctypes.c_int
        assert libc.mprotect(address, 4096, mmap.PROT_READ | mmap.PROT_EXEC) == -1
        assert ctypes.get_errno() == 1
        try:
            mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_EXEC)
        except PermissionError:
            print("blocked")
        else:
            raise SystemExit(9)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked"


@pytest.mark.parametrize(
    "body,event",
    [
        ("import subprocess\nsubprocess.run(['/bin/true'])\n", "subprocess.Popen"),
        ("import ctypes\nctypes.CDLL(None)\n", "ctypes.dlopen"),
        ("import os\nos.system('true')\n", "os.system"),
        ("import _posixsubprocess\n", "_posixsubprocess"),
    ],
)
def test_guard_blocks_external_or_dynamic_code(tmp_path, body, event):
    completed = _run(tmp_path, body)

    assert completed.returncode != 0
    assert event in completed.stderr
