from pathlib import Path
import subprocess
import sys

import numpy as np
import grader.gates.probes as probes
from grader.metrics.subspace import subspace_accuracy
from data.vcf_serialization import PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS
from reference.full_scan_pca import fit

REPRESENTATION_KEY = bytes(range(32))


def _structured_genotypes(seed: int = 19, samples: int = 100,
                          variants: int = 800) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = np.arange(samples) % 4
    groups = groups[rng.permutation(samples)]
    ancestral = rng.uniform(0.08, 0.92, size=variants)
    shifts = np.array([-0.16, -0.05, 0.05, 0.16])
    frequencies = np.clip(ancestral[:, None] + shifts[None, :], 0.01, 0.99)
    G = np.empty((variants, samples), dtype=np.int8)
    for group in range(4):
        columns = np.flatnonzero(groups == group)
        G[:, columns] = rng.binomial(
            2, frequencies[:, group, None], size=(variants, columns.size))
    return G


def _exact_run_one(vcf: Path, k: int, out: Path, sample_ids: list[str]):
    ids, scores, _, _ = fit(vcf, k)
    assert ids == sample_ids
    return True, scores


def test_pair_serializations_recover_same_exact_patterson_subspace(tmp_path):
    rng = np.random.default_rng(7)
    G = _structured_genotypes(samples=92, variants=500)
    missing = rng.random(G.shape) < 0.06
    G[missing] = -1
    sample_ids = [f"subject_{index}" for index in range(G.shape[1])]
    plain = tmp_path / "first.vcf"
    rich = tmp_path / "second.vcf"
    carrier = probes.ProbeCarrier(92, 500, 3, "synthetic_variable_width")
    order_a, order_b = probes._write_representation_pair(
        plain, rich, G, sample_ids, rng, carrier)

    ids_a, scores_a, kept_a, _ = fit(plain, 3)
    ids_b, scores_b, kept_b, _ = fit(rich, 3)
    ref, _ = probes._patterson_pca(G, 3)
    canonical_a = np.empty_like(scores_a)
    canonical_b = np.empty_like(scores_b)
    canonical_a[order_a] = scores_a
    canonical_b[order_b] = scores_b

    assert ids_a == [sample_ids[index] for index in order_a]
    assert ids_b == [sample_ids[index] for index in order_b]
    assert kept_a == kept_b
    assert subspace_accuracy(canonical_a, ref, np.ones(3))["accuracy"] > 0.999999
    assert subspace_accuracy(canonical_b, ref, np.ones(3))["accuracy"] > 0.999999

    rich_bytes = rich.read_bytes()
    plain_bytes = plain.read_bytes()
    for payload in (plain_bytes, rich_bytes):
        assert b"\tGT\t" in payload
        assert b"\tGT:DP\t" in payload
        assert b"\tGT:DP:GQ\t" in payload
        assert b"\tGT:AD:DP:GQ\t" in payload
        assert b"\tDP=" in payload and b";ZZ=" in payload
        assert b"\t.\t" in payload
        assert b"|" in payload
        assert b"\r\n" in payload
        assert not payload.endswith((b"\r", b"\n"))
    assert any(line.split(b"\t")[3].islower()
               for line in rich_bytes.splitlines() if line and not line.startswith(b"#"))
    plain_gt = {
        token.split(b":", 1)[0]
        for line in plain_bytes.splitlines() if line and not line.startswith(b"#")
        for token in line.split(b"\t")[9:]
    }
    rich_gt = {
        token.split(b":", 1)[0]
        for line in rich_bytes.splitlines() if line and not line.startswith(b"#")
        for token in line.split(b"\t")[9:]
    }
    assert {b".", b"./.", b".|."} <= plain_gt
    assert not ({b"0x1", b"1x0", b"00", b"11", b"10/0"} & plain_gt)
    assert {b"0", b"1", b"0x1", b"1x0", b"00", b"11", b"10/0"} <= rich_gt
    assert not ({b".", b"./.", b".|."} & rich_gt)

    # Malformed spellings are representation-only, but their first bytes form one private,
    # balanced sample contrast. A correct parser ignores the entire token. A prefix parser instead
    # invents a coherent 0-vs-2 axis, making the contract violation observable above the gate's
    # deliberately conservative dead-zone rather than merely adding isotropic noise.
    malformed = set(PREFIX_AMBIGUOUS_MISSING_GT_ENCODINGS)
    prefixes_by_sample = [set() for _ in range(G.shape[1])]
    for line in rich_bytes.splitlines():
        if not line or line.startswith(b"#"):
            continue
        for sample, token in enumerate(line.split(b"\t")[9:]):
            gt = token.split(b":", 1)[0]
            if gt in malformed:
                prefixes_by_sample[sample].add(gt[:1])
    assert all(prefixes in ({b"0"}, {b"1"}) for prefixes in prefixes_by_sample)
    assert sum(prefixes == {b"0"} for prefixes in prefixes_by_sample) == G.shape[1] // 2
    assert sum(prefixes == {b"1"} for prefixes in prefixes_by_sample) == G.shape[1] // 2


def test_representation_severity_has_dead_zones_and_catches_equal_failure():
    assert probes._representation_severity(0.97, 0.91) == 0.0
    partial = probes._representation_severity(0.98, 0.70)
    assert 0.0 < partial < 1.0
    assert probes._representation_severity(1.0, 0.0) == 1.0
    assert probes._representation_severity(0.1, 0.1) == 1.0


def test_exact_format_aware_fit_is_not_penalized(monkeypatch, tmp_path):
    G = _structured_genotypes()
    carrier = probes.ProbeCarrier(
        G.shape[1], G.shape[0], 3, "synthetic_variable_width")
    monkeypatch.setattr(probes, "_build_representation_probe", lambda rng, selected: G)
    result = probes.representation_equivalence_gate(
        _exact_run_one, tmp_path, [carrier], seed=11,
        representation_key=REPRESENTATION_KEY)

    assert result["severity"] == 0.0
    assert result["factor"] == 1.0
    assert result["reference_agreement_a"] > 0.999
    assert result["reference_agreement_b"] > 0.999


def test_malformed_prefixes_cannot_masquerade_as_haploid_calls(tmp_path):
    """A parser must validate the whole GT token, not only its first allele-like byte."""
    # Exercise the real k=5 probe builder rather than a hand-strengthened genotype fixture. The
    # production carrier is larger (360 x 50k) but uses this identical construction.
    carrier = probes.ProbeCarrier(120, 20_000, 5, "synthetic_variable_width")
    accepted = {
        b"0x1": b"0", b"1x0": b"1", b"00": b"0", b"11": b"1", b"10/0": b"1",
    }

    def prefix_accepting_run(vcf: Path, k: int, out: Path, sample_ids: list[str]):
        rewritten = vcf.with_name(vcf.stem + ".prefix-accepted.vcf")
        output: list[bytes] = []
        for line in vcf.read_bytes().splitlines():
            if not line or line.startswith(b"#"):
                output.append(line)
                continue
            fields = line.split(b"\t")
            for index in range(9, len(fields)):
                parts = fields[index].split(b":")
                parts[0] = accepted.get(parts[0], parts[0])
                fields[index] = b":".join(parts)
            output.append(b"\t".join(fields))
        rewritten.write_bytes(b"\n".join(output))
        ids, scores, _, _ = fit(rewritten, k)
        assert ids == sample_ids
        return True, scores

    result = probes.representation_equivalence_gate(
        prefix_accepting_run, tmp_path, [carrier, carrier, carrier], seed=17,
        representation_key=REPRESENTATION_KEY)

    # The private balanced contrast is structurally material, so this sits safely beyond the
    # 0.08 raw agreement dead-zone instead of depending on lowering a false-positive safeguard.
    diagnostic = (
        f"severity={result['severity']!r}, factor={result['factor']!r}, "
        f"a={result['reference_agreement_a']!r}, b={result['reference_agreement_b']!r}, "
        f"gap={result['agreement_gap']!r}, reps={result['severities']!r}"
    )
    # The parser must fail the representation-mastery bar (0.90), not merely receive a cosmetic
    # score dent while still mastering the parser-stress category.
    assert result["severity"] >= 0.20, diagnostic
    assert result["factor"] < 0.90, diagnostic


def test_format_sensitive_fit_receives_continuous_penalty(monkeypatch, tmp_path):
    G = _structured_genotypes()
    carrier = probes.ProbeCarrier(
        G.shape[1], G.shape[0], 3, "synthetic_variable_width")
    monkeypatch.setattr(probes, "_build_representation_probe", lambda rng, selected: G)

    calls = 0

    def format_sensitive_run(vcf: Path, k: int, out: Path, sample_ids: list[str]):
        nonlocal calls
        ids, scores, _, _ = fit(vcf, k)
        assert ids == sample_ids
        calls += 1
        if calls == 2:
            scores = np.random.default_rng(1234).normal(size=scores.shape)
        return True, scores

    result = probes.representation_equivalence_gate(
        format_sensitive_run, tmp_path, [carrier], seed=13,
        representation_key=REPRESENTATION_KEY)

    assert result["severity"] > 0.9
    assert 0.0 <= result["factor"] < 0.1
    assert result["reference_agreement_a"] > 0.999
    assert result["reference_agreement_b"] < 0.2


def test_known_fixed_width_shortcut_is_caught(monkeypatch, tmp_path):
    import grader.grade as grade

    G = _structured_genotypes(samples=100, variants=500)
    carrier = probes.ProbeCarrier(
        G.shape[1], G.shape[0], 3, "synthetic_variable_width")
    monkeypatch.setattr(probes, "_build_representation_probe", lambda rng, selected: G)
    shortcut = Path(__file__).resolve().parents[1] / "cheats/format_fragile/pca"

    def run_one(vcf: Path, k: int, out: Path, sample_ids: list[str]):
        completed = subprocess.run(
            [sys.executable, str(shortcut), str(vcf), str(k), str(out)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return False, None
        scores, _ = grade.load_scores(out, sample_ids, k)
        return scores is not None, scores

    result = probes.representation_equivalence_gate(
        run_one, tmp_path, [carrier], seed=41,
        representation_key=REPRESENTATION_KEY)
    assert result["severity"] == 1.0
    assert result["factor"] == 0.0


def test_run_probe_gates_wires_representation_gate(monkeypatch, tmp_path):
    import grader.grade as grade

    sentinel = object()
    monkeypatch.setattr(
        grade,
        "_make_run_one",
        lambda submission_dir, workdir, timings, *, isolation, budget=None: sentinel,
    )

    received = {}

    def fake_gate(name):
        def run(run_one, workdir, carriers, seed, *, representation_key):
            assert run_one is sentinel
            assert len(representation_key) == 32
            assert carriers
            received[name] = tuple(carriers)
            return {"name": name, "factor": 1.0, "severity": 0.0}
        return run

    monkeypatch.setattr(probes, "hwe_norm_gate", fake_gate("hwe_norm"))
    monkeypatch.setattr(probes, "coverage_gate", fake_gate("coverage"))
    monkeypatch.setattr(probes, "representation_equivalence_gate",
                        fake_gate("representation_equivalence"))

    active = [
        probes.ProbeCarrier(384, 64_000, 8, "synthetic_plain"),
        probes.ProbeCarrier(320, 60_000, 16, "synthetic_plain"),
        probes.ProbeCarrier(400, 250_000, 12, "synthetic_plain"),
        probes.ProbeCarrier(500, 120_000, 5, "synthetic_plain"),
        probes.ProbeCarrier(500, 120_000, 3, "synthetic_variable_width"),
    ]
    result, runtime = grade.run_probe_gates(
        tmp_path, active, tmp_path, bytes(range(32)), isolation="bwrap",
    )
    assert set(result) == {"hwe_norm", "coverage", "representation_equivalence"}
    assert runtime == {
        "submission_seconds": 0,
        "reference_seconds": 0,
        "invocations": 0,
        "deadline_exhausted": False,
    }
    selected = tuple(carrier for carriers in received.values() for carrier in carriers)
    assert set(selected) <= set(active)
    assert len({carrier.k for carrier in selected}) >= 3
