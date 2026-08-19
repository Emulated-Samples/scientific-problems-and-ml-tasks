from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from grader.gates import library_scan


def _submission(tmp_path: Path) -> Path:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("#!/usr/bin/env python3\nimport numpy\n")
    return submission


def test_scan_rejects_shipped_bytecode_inside_pycache(tmp_path):
    submission = _submission(tmp_path)
    cache = submission / "__pycache__"
    cache.mkdir()
    (cache / "hidden.cpython-312.pyc").write_bytes(
        importlib.util.MAGIC_NUMBER + b"\x00" * 28,
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "bytecode-payload" for hit in result["hits"])


def test_scan_rejects_renamed_zipimport_archive_by_magic(tmp_path):
    submission = _submission(tmp_path)
    payload = submission / "weights.dat"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("hidden.py", "import forbidden_dependency\n")

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_allows_numeric_npy_and_npz_helper_data(tmp_path):
    submission = _submission(tmp_path)
    np.save(submission / "weights.npy", np.arange(24, dtype=np.float32).reshape(6, 4))
    np.savez(submission / "empty.npz")
    np.savez_compressed(
        submission / "tables.npz",
        offsets=np.arange(8, dtype=np.int64),
        weights=np.linspace(0.0, 1.0, 8),
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


def test_scan_rejects_pickle_backed_object_arrays(tmp_path):
    submission = _submission(tmp_path)
    np.savez(submission / "payload.npz", encoded=np.array([{"code": "hidden"}], dtype=object))

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_allows_data_only_zip_and_renamed_zip(tmp_path):
    submission = _submission(tmp_path)
    for name in ("metadata.zip", "metadata.cache"):
        with zipfile.ZipFile(submission / name, "w") as archive:
            archive.writestr("columns.csv", "sample,weight\na,1.0\n")
            archive.writestr("schema.json", '{"version": 1}\n')

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


def test_scan_rejects_native_member_in_renamed_zip(tmp_path):
    submission = _submission(tmp_path)
    with zipfile.ZipFile(submission / "weights.dat", "w") as archive:
        archive.writestr("matrix.bin", b"\x7fELF" + b"\x00" * 64)

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_allows_compressed_and_tarred_data(tmp_path):
    submission = _submission(tmp_path)
    with gzip.open(submission / "columns.csv.gz", "wb") as stream:
        stream.write(b"sample,weight\na,1.0\n")
    with tarfile.open(submission / "tables.tar.gz", "w:gz") as archive:
        payload = b"sample,weight\na,1.0\n"
        info = tarfile.TarInfo("columns.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


def test_scan_rejects_zip_member_decompression_bomb_before_open(tmp_path, monkeypatch):
    submission = _submission(tmp_path)
    monkeypatch.setattr(library_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 1024)
    with zipfile.ZipFile(submission / "oversized.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("table.csv", b"0," * 1024)
    monkeypatch.setattr(
        zipfile.ZipFile,
        "open",
        lambda *_args, **_kwargs: pytest.fail("oversized zip member was opened"),
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_rejects_zip_total_decompression_bomb(tmp_path, monkeypatch):
    submission = _submission(tmp_path)
    monkeypatch.setattr(library_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 4096)
    monkeypatch.setattr(library_scan, "_MAX_ARCHIVE_TOTAL_BYTES", 2048)
    with zipfile.ZipFile(submission / "many-tables.zip", "w") as archive:
        archive.writestr("a.csv", b"a" * 1536)
        archive.writestr("b.csv", b"b" * 1536)

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_rejects_zip_compression_ratio_bomb(tmp_path, monkeypatch):
    submission = _submission(tmp_path)
    monkeypatch.setattr(library_scan, "_MAX_ARCHIVE_COMPRESSION_RATIO", 4)
    with zipfile.ZipFile(submission / "ratio.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("repeated.csv", b"same,value\n" * 4096)

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_rejects_tar_member_decompression_bomb_before_extract(tmp_path, monkeypatch):
    submission = _submission(tmp_path)
    monkeypatch.setattr(library_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 1024)
    with tarfile.open(submission / "oversized.tar.gz", "w:gz") as archive:
        payload = b"x" * 2048
        info = tarfile.TarInfo("table.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(
        tarfile.TarFile,
        "extractfile",
        lambda *_args, **_kwargs: pytest.fail("oversized tar member was extracted"),
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_rejects_tar_total_compression_ratio_bomb(tmp_path, monkeypatch):
    submission = _submission(tmp_path)
    monkeypatch.setattr(library_scan, "_MAX_ARCHIVE_COMPRESSION_RATIO", 4)
    with tarfile.open(submission / "ratio.tar.gz", "w:gz") as archive:
        payload = b"same,value\n" * 4096
        info = tarfile.TarInfo("table.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(
        tarfile.TarFile,
        "extractfile",
        lambda *_args, **_kwargs: pytest.fail("ratio-bomb tar member was extracted"),
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


def test_scan_rejects_compressed_python_source(tmp_path):
    submission = _submission(tmp_path)
    with gzip.open(submission / "hidden.py.gz", "wb") as stream:
        stream.write(b"import forbidden_dependency\n")

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "archive-payload" for hit in result["hits"])


@pytest.mark.parametrize(
    "header",
    [
        b"MZ" + b"\x00" * 30,
        b"\x00asm" + b"\x01\x00\x00\x00",
        b"\xcf\xfa\xed\xfe" + b"\x00" * 28,
        b"!<arch>\n" + b"\x00" * 24,
    ],
)
def test_scan_rejects_renamed_native_payload_magic(tmp_path, header):
    submission = _submission(tmp_path)
    (submission / "matrix.cache").write_bytes(header)

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "native-payload" for hit in result["hits"])


def test_scan_still_allows_inspectable_source_helpers(tmp_path):
    submission = _submission(tmp_path)
    (submission / "helper.py").write_text("import scipy.linalg\nVALUE = 1\n")

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


@pytest.mark.parametrize(
    "module",
    ["_ctypes", "_imp", "ctypes", "importlib", "marshal", "runpy", "zipimport"],
)
def test_scan_rejects_opaque_code_loader_modules(tmp_path, module):
    submission = _submission(tmp_path)
    (submission / "pca").write_text(f"#!/usr/bin/env python3\nimport {module}\n")

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "import" and hit["module"] == module for hit in result["hits"])


@pytest.mark.parametrize("module", ["pickle", "_pickle", "shelve"])
def test_scan_accepts_pickle_family_data_serialization(tmp_path, module):
    """Object serialization is data, not a reconstructed executable payload.

    The prompt bans reconstructed *executable/native* payloads, not pickling; a hand-rolled fork
    worker legitimately pickles NumPy arrays over a pipe. The only integrity risk -- a hostile
    ``__reduce__`` -- cannot exceed the runtime audit hook (a forbidden import, ctypes, os/subprocess
    exec, or code/function construction from inside an unpickle is blocked exactly as from ordinary
    code), and a shipped pickle *blob* is still rejected at file level. Denying the module name only
    zeroed correct solves, the same static/runtime disagreement that retired the exec/eval hit.
    """
    submission = _submission(tmp_path)
    (submission / "pca").write_text(f"#!/usr/bin/env python3\nimport {module}\n")

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


def test_scan_still_rejects_a_shipped_pickle_blob_even_though_pickle_is_importable(tmp_path):
    """Permitting ``import pickle`` must not permit shipping an opaque pickled object *file*.

    The data-to-code vector is closed at file level, independently of the module allowlist: a
    ``.pkl`` (or object-dtype ``.npy``) blob is the container a hostile ``__reduce__`` would be
    smuggled in, and it stays rejected so the module can be allowed for runtime serialization only.
    """
    submission = _submission(tmp_path)
    (submission / "helper.py").write_text("import pickle\n")
    (submission / "state.pkl").write_bytes(b"\x80\x04K\x01.")

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "encoded-python-payload" for hit in result["hits"])


@pytest.mark.parametrize("evaluator", ["compile", "eval", "exec"])
def test_scan_accepts_a_submission_that_metaprograms_its_own_python(tmp_path, evaluator):
    """Dynamic Python is NOT an integrity violation, and this gate must agree with the runtime.

    `submission_runner` deliberately permits runtime codegen: the image ships only NumPy and SciPy
    and the import allowlist is enforced by an audit hook CPython cannot unregister, so generated
    code cannot reach a library that is not already allowed. While this scan still flagged every
    `exec`/`eval`/`compile`, the two gates disagreed and the static one was wrong -- measured through
    the real grader, a submission that metaprograms its own pure-Python helper RAN, scored accuracy
    1.000 on every dev fold, and still took reward 0.0 from this gate, with an empty stderr. That is
    a correct program scoring zero for a reason unrelated to its correctness.

    The prompt is the tiebreak: it forbids encoding or reconstructing executable/NATIVE payloads, and
    says nothing against ordinary metaprogramming. A gate may only enforce what the prompt states.
    """
    submission = _submission(tmp_path)
    (submission / "pca").write_text(
        f"#!/usr/bin/env python3\npayload = '40 + 2'\nprint({evaluator}(payload))\n",
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


def test_scan_still_rejects_a_computed_import_name(tmp_path):
    """The margin. Permitting codegen must not permit reaching a forbidden library.

    Computing an import name is the tell, and it stays a static hit. Runtime is the real boundary --
    `exec(compile("import " + "nu" + "mba", ...))` is refused on the box with
    `third-party import is disabled: numba` -- but a submission that never runs still must not pass
    this gate by assembling the name.
    """
    submission = _submission(tmp_path)
    (submission / "pca").write_text(
        "#!/usr/bin/env python3\n"
        "mod = __import__('num' + 'ba')\n",
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["module"] == "<computed-import>" for hit in result["hits"])


def test_scan_does_not_confuse_an_ordinary_eval_method_with_builtin_eval(tmp_path):
    submission = _submission(tmp_path)
    (submission / "helper.py").write_text(
        "class Parser:\n"
        "    def eval(self, value):\n"
        "        return float(value)\n"
        "\n"
        "VALUE = Parser().eval('1.5')\n",
    )

    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []


@pytest.mark.parametrize(("fixture", "module"), [("numba_jit", "numba"), ("sklearn_pca", "sklearn")])
def test_the_real_cheat_fixtures_still_fail_closed(fixture, module):
    """The margin, pinned against the SHIPPED fixtures rather than a synthetic string.

    Retiring the blanket dynamic-code hit removed a layer, so the layer that actually carries the
    dependency policy has to be shown still carrying it -- on the real arms, cheaply, on every run.
    The full ladder in `scripts/validate.py` proves this too, but only under root+bwrap in CI; a
    dependency-policy regression should not need a sandbox to notice.
    """
    submission = Path(__file__).resolve().parents[1] / "cheats" / fixture
    result = library_scan.scan(submission)

    assert result["factor"] == 0.0
    assert any(hit["kind"] == "import" and hit["module"] == module for hit in result["hits"])


@pytest.mark.parametrize("fixture", ["raw_covariance", "first_k_variants"])
def test_method_cheats_are_not_the_library_gate_s_job(fixture):
    """The other side of that margin: this gate must not claim credit it did not earn.

    A raw-covariance or first-k-variants shortcut imports nothing forbidden -- it is WRONG SCIENCE,
    and the hwe_norm/coverage probes are what catch it. If library_scan started failing these, a
    dependency regression would hide behind a gate that was firing for an unrelated reason.
    """
    submission = Path(__file__).resolve().parents[1] / "cheats" / fixture
    result = library_scan.scan(submission)

    assert result["factor"] == 1.0
    assert result["hits"] == []
