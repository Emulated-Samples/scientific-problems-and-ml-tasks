from pathlib import Path

from grader.grade import _validity, load_scores
from data.generate import dev_specs, grade_specs
from grader.gates.probes import ProbeCarrier, select_probe_carriers


MATH_KEY = bytes(range(32))


def test_score_loader_requires_exact_requested_width(tmp_path):
    out = tmp_path / "scores.tsv"
    out.write_text("sample_id\tPC1\tPC2\ns1\t0.1\ns2\t-0.1\n")

    scores, reason = load_scores(out, ["s1", "s2"], 2)

    assert scores is None
    assert reason == "expected exactly 2 PC columns, got 1"


def test_score_loader_requires_canonical_pc_header(tmp_path):
    out = tmp_path / "scores.tsv"
    out.write_text("sample_id\taxis1\taxis2\ns1\t0.1\t0.2\ns2\t-0.1\t-0.2\n")

    scores, reason = load_scores(out, ["s1", "s2"], 2)

    assert scores is None
    assert reason == "header must be sample_id/PC1/PC2"


def test_score_loader_stops_after_the_exact_sample_count(tmp_path):
    out = tmp_path / "scores.tsv"
    out.write_text("sample_id\tPC1\ns1\t0.1\ns2\t-0.1\nextra\t0.0\n")

    scores, reason = load_scores(out, ["s1", "s2"], 1)

    assert scores is None
    assert reason == "extra score rows"


def test_score_loader_accepts_every_finite_output_scale(tmp_path):
    out = tmp_path / "scores.tsv"
    out.write_text("sample_id\tPC1\ns1\t1e300\ns2\t-1e300\n")

    scores, reason = load_scores(out, ["s1", "s2"], 1)

    assert reason == "ok"
    assert scores[:, 0].tolist() == [1e300, -1e300]


def test_score_loader_aligns_by_position_and_id_rejecting_reordered_rows(tmp_path):
    """The alignment contract: row i MUST be VCF sample i AND carry sample i's id. A submission that
    emits a correct PCA but permutes the rows (self-consistent ids, wrong order) is a contract
    violation and must be REJECTED, never silently re-joined by id (an FP seam) and never positionally
    mis-scored against the wrong labels (an FN). This is what keeps S row j 1:1 with truth label j.
    """
    out = tmp_path / "scores.tsv"
    out.write_text("sample_id\tPC1\ns1\t-0.1\ns0\t0.1\n")

    scores, reason = load_scores(out, ["s0", "s1"], 1)

    assert scores is None
    assert reason == "samples not in VCF order"


def test_score_loader_rejects_duplicate_and_missing_samples(tmp_path):
    """Duplicate ids cannot silently overwrite a row, and a short file cannot be padded/aligned."""
    dup = tmp_path / "dup.tsv"
    dup.write_text("sample_id\tPC1\ns0\t0.1\ns0\t-0.1\n")
    assert load_scores(dup, ["s0", "s1"], 1) == (None, "samples not in VCF order")

    short = tmp_path / "short.tsv"
    short.write_text("sample_id\tPC1\ns0\t0.1\n")
    scores, reason = load_scores(short, ["s0", "s1"], 1)
    assert scores is None
    assert reason == "missing score row for 's1'"


def test_score_loader_rejects_nonfinite_values(tmp_path):
    """NaN/Inf/overflowing magnitudes parse as floats but are not a scorable subspace; they must be
    rejected here so a degenerate matrix never reaches the accuracy metric as if it were valid."""
    import numpy as np

    for token in ("nan", "inf", "-inf", "1e400"):
        out = tmp_path / f"nf_{token}.tsv"
        out.write_text(f"sample_id\tPC1\ns0\t{token}\ns1\t-1.0\n")
        scores, reason = load_scores(out, ["s0", "s1"], 1)
        assert scores is None, token
        assert reason == "non-finite scores", token

    # A genuinely finite (even extreme) file is still accepted -- the check rejects non-finite, not big.
    ok = tmp_path / "ok.tsv"
    ok.write_text("sample_id\tPC1\ns0\t1e300\ns1\t-1e300\n")
    scores, reason = load_scores(ok, ["s0", "s1"], 1)
    assert reason == "ok" and np.isfinite(scores).all()


def test_score_loader_accepts_crlf_and_a_missing_final_newline(tmp_path):
    """FN guard: line-ending conventions a correct submission might legitimately emit must NOT be
    rejected. Universal-newline reading handles CRLF, and the last row needs no trailing newline."""
    crlf = tmp_path / "crlf.tsv"
    crlf.write_bytes(b"sample_id\tPC1\r\ns0\t0.5\r\ns1\t-0.5\r\n")
    scores, reason = load_scores(crlf, ["s0", "s1"], 1)
    assert reason == "ok"
    assert scores[:, 0].tolist() == [0.5, -0.5]

    no_nl = tmp_path / "no_nl.tsv"
    no_nl.write_text("sample_id\tPC1\ns0\t0.5\ns1\t-0.5")
    scores, reason = load_scores(no_nl, ["s0", "s1"], 1)
    assert reason == "ok"
    assert scores[:, 0].tolist() == [0.5, -0.5]


def test_score_loader_rejects_non_numeric_and_invalid_utf8(tmp_path):
    """Neither a non-numeric cell nor undecodable bytes may be silently coerced into a scorable S."""
    words = tmp_path / "words.tsv"
    words.write_text("sample_id\tPC1\ns0\tabc\ns1\t-1.0\n")
    assert load_scores(words, ["s0", "s1"], 1) == (None, "non-numeric score for 's0'")

    binary = tmp_path / "binary.tsv"
    binary.write_bytes(b"sample_id\tPC1\ns0\t\xff\xfe\ns1\t-1.0\n")
    scores, reason = load_scores(binary, ["s0", "s1"], 1)
    assert scores is None
    assert reason.startswith("unreadable:")


def test_validity_requires_exact_requested_width():
    import numpy as np

    result = _validity(
        np.zeros((4, 2)), "ok", {"returncode": 0}, 3,
    )

    assert result["factor"] == 0.0
    assert result["reason"] == "expected exactly 3 PCs, got 2"


def test_hidden_catalog_varies_requested_rank_without_invalid_shapes():
    specs = grade_specs(MATH_KEY)
    ordinary_requested = {spec.k for spec in specs if spec.regime == "balding_nichols"}
    high_rank = [spec for spec in specs if spec.category == "high_rank"]
    very_high_rank = [spec for spec in specs if spec.category == "very_high_rank"]
    crossover = [spec for spec in specs if spec.category == "crossover"]

    assert ordinary_requested == {3, 5, 6, 8, 10, 12, 16}
    assert len(high_rank) == 1
    assert 24 <= high_rank[0].k <= 48
    assert high_rank[0].n_pops - 1 >= high_rank[0].k
    assert len(very_high_rank) == 1
    assert 64 <= very_high_rank[0].k <= 96
    assert very_high_rank[0].n_pops - 1 >= very_high_rank[0].k
    assert len(crossover) == 1
    assert crossover[0].n_samples == crossover[0].n_variants
    assert crossover[0].n_pops - 1 >= crossover[0].k
    assert all(1 <= spec.k < spec.n_samples for spec in specs)
    spectral_selection = [spec for spec in specs if spec.category == "spectral_selection"]
    assert len(spectral_selection) == 1
    assert spectral_selection[0].n_pops - 1 >= spectral_selection[0].k + 4
    assert len(specs) == 21
    assert {
        "crossover", "haploid", "io_wide", "sample_heavy", "spatial_ld", "rare_structure",
        "spectral_selection", "variable_width", "very_high_rank",
    } <= {
        spec.category for spec in specs
    }


def test_dev_catalog_can_run_the_real_probe_plan():
    specs = dev_specs(MATH_KEY)
    assert {"messy", "variable_width"} <= {spec.category for spec in specs}
    carriers = [
        ProbeCarrier(
            spec.n_samples,
            spec.n_variants,
            spec.k,
            "synthetic_variable_width" if spec.representation == "variable_width"
            else "synthetic_messy" if spec.messy_frac > 0
            else "synthetic_plain",
        )
        for spec in specs
    ]

    plan = select_probe_carriers(carriers, MATH_KEY)

    assert set(plan) == {"hwe_norm", "coverage", "representation_equivalence"}
    assert plan["hwe_norm"]


def test_harbor_entrypoint_accepts_the_canonical_suite_shape():
    """A stale shell preflight must not reject the suite before grading starts."""
    root = Path(__file__).resolve().parents[1]
    entrypoint = (root / "tasks" / "from_scratch_pca" / "tests" / "test.sh").read_text()

    assert "assert len(truth_paths) == 22" in entrypoint
    for category in ("crossover", "spectral_selection", "very_high_rank"):
        assert f'"{category}"' in entrypoint
