import os
from pathlib import Path

import pytest

from data.release_key import (
    RELEASE_ID,
    derive_bytes,
    derive_seed,
    key_commitment,
    read_math_key,
)


KEY = bytes(range(32))


def _write_private_key(path, key=KEY):
    path.write_text(key.hex() + "\n")
    path.chmod(0o600)
    return path


def test_private_key_round_trip_and_release_commitment(tmp_path):
    path = _write_private_key(tmp_path / "math.key")

    assert read_math_key(path) == KEY
    assert len(key_commitment(KEY)) == 64
    assert key_commitment(KEY) != key_commitment(b"z" * 32)
    assert RELEASE_ID.encode() not in KEY


def test_derivations_are_stable_domain_separated_and_keyed():
    first = derive_bytes(KEY, "synthetic/grade/example")

    assert first == derive_bytes(KEY, "synthetic/grade/example")
    assert first != derive_bytes(KEY, "synthetic/grade/other")
    assert first != derive_bytes(b"z" * 32, "synthetic/grade/example")
    assert 0 <= derive_seed(KEY, "probe/hwe") < 2**63
    assert derive_seed(KEY, "probe/hwe") != derive_seed(KEY, "probe/coverage")


@pytest.mark.parametrize(
    "payload",
    [
        "00" * 31,
        "00" * 33,
        "AA" * 32,
        "not-a-key",
    ],
)
def test_key_reader_rejects_malformed_payloads(tmp_path, payload):
    path = tmp_path / "math.key"
    path.write_text(payload + "\n")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        read_math_key(path)


def test_key_reader_rejects_group_or_world_access(tmp_path):
    path = _write_private_key(tmp_path / "math.key")
    path.chmod(0o640)

    with pytest.raises(PermissionError, match="group or other"):
        read_math_key(path)


def test_key_reader_rejects_hardlinks(tmp_path):
    path = _write_private_key(tmp_path / "math.key")
    alias = tmp_path / "alias.key"
    os.link(path, alias)

    with pytest.raises(PermissionError, match="singly linked"):
        read_math_key(path)


def test_key_reader_does_not_follow_symlinks(tmp_path):
    path = _write_private_key(tmp_path / "math.key")
    alias = tmp_path / "alias.key"
    alias.symlink_to(path)

    with pytest.raises(OSError):
        read_math_key(alias)


@pytest.mark.parametrize("length", [0, 33])
def test_derivation_rejects_invalid_key_lengths(length):
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        derive_bytes(b"x" * length, "domain")


@pytest.mark.parametrize("domain", ["", "bad\x00domain"])
def test_derivation_rejects_ambiguous_domains(domain):
    with pytest.raises(ValueError, match="nonempty plain string"):
        derive_bytes(KEY, domain)


def test_ci_invokes_every_keyed_entrypoint_with_the_release_key():
    workflow = Path(".github/workflows/validate.yml").read_text()

    assert "install -m 0600 grader/private/release-v3.math-key /tmp/pcabench-ci-math.key" in workflow
    assert "sudo chown root:root /tmp/pcabench-ci-math.key" in workflow
    assert "sudo chmod 0400 /tmp/pcabench-ci-math.key" in workflow
    assert "--math-key-file /tmp/pcabench-ci-math.key" in workflow
    assert "--representation-key-file /tmp/pcabench-ci-representation.key" in workflow
    assert workflow.count("--math-key-file /tmp/pcabench-ci-math.key") == 2
