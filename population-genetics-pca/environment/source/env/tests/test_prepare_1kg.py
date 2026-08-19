import pytest

from data.prepare_1kg import (
    DERIVATION_VERSION,
    PanelLabel,
    _record_selected,
    build_truth,
    read_panel,
    select_balanced_samples,
    write_reviewed_subset,
)
from data.release_key import RELEASE_ID, key_commitment


SOURCE_SHA = "a" * 64
MATH_KEY = bytes(range(32))
COHORT_KEY = b"c" * 32
RECORD_KEY = b"r" * 32
REPRESENTATION_KEY = bytes(reversed(range(32)))
MATH_COMMITMENT = key_commitment(MATH_KEY)


def _write_fixture(tmp_path, *, omit_last_panel_sample=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    samples = ["s_af", "s_am", "s_ea", "s_eu", "s_sa"]
    vcf = tmp_path / "chr22.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\r\n"
        + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples)
        + "\r\n"
        + "22\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\t0|0\t0|1\r\n"
    )
    panel = tmp_path / "panel.txt"
    rows = [
        ("s_af", "YRI", "AFR"),
        ("s_am", "PEL", "AMR"),
        ("s_ea", "CHB", "EAS"),
        ("s_eu", "CEU", "EUR"),
        ("s_sa", "GIH", "SAS"),
    ]
    if omit_last_panel_sample:
        rows.pop()
    panel.write_text(
        "sample\tpop\tsuper_pop\tgender\r\n"
        + "".join(
            f"{sample}\t{population}\t{superpop}\t0\r\n"
            for sample, population, superpop in rows
        )
    )
    return vcf, panel, samples


def test_build_truth_requires_complete_reviewed_panel_and_preserves_vcf_order(tmp_path):
    vcf, panel, samples = _write_fixture(tmp_path)

    truth = build_truth(
        str(vcf), read_panel(str(panel)), name="observed", category="observed",
        weight=2.0, k=4, source_sha256=SOURCE_SHA,
        math_commitment=MATH_COMMITMENT,
    )

    assert truth["sample_ids"] == samples
    assert truth["sample_pop"] == [0, 1, 2, 3, 4]
    assert truth["sample_subpop"] == [4, 3, 1, 0, 2]
    assert truth["superpop_names"] == ["AFR", "AMR", "EAS", "EUR", "SAS"]
    assert truth["population_names"] == ["CEU", "CHB", "GIH", "PEL", "YRI"]
    assert truth["spec"] == {
        "name": "observed",
        "category": "observed",
        "weight": 2.0,
        "k": 4,
        "structure_weight": 0.30,
        "hierarchy_weight": 0.45,
        "source": "1000genomes_phase3_grch37",
        "source_sha256": SOURCE_SHA,
        "derivation_version": DERIVATION_VERSION,
        "math_release": RELEASE_ID,
        "math_key_commitment": MATH_COMMITMENT,
    }

    _, incomplete_panel, _ = _write_fixture(
        tmp_path / "incomplete", omit_last_panel_sample=True,
    )
    with pytest.raises(ValueError, match="absent from the reviewed panel"):
        build_truth(
            str(tmp_path / "incomplete/chr22.vcf"), read_panel(str(incomplete_panel)),
            name="observed", category="observed", weight=2.0, k=4,
            source_sha256=SOURCE_SHA,
            math_commitment=MATH_COMMITMENT,
        )


def test_build_truth_rejects_unreviewed_source_digest(tmp_path):
    vcf, panel, _ = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        build_truth(
            str(vcf), read_panel(str(panel)), name="observed", category="observed",
            weight=2.0, k=4, source_sha256="not-a-digest",
            math_commitment=MATH_COMMITMENT,
        )


def test_balanced_selection_is_release_stable_and_privately_permuted():
    superpops = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    samples = [f"{pop}_{index}" for index in range(3) for pop in reversed(superpops)]
    panel = {
        sample: PanelLabel(f"{sample.split('_')[0]}_population", sample.split("_")[0])
        for sample in samples
    }

    first = select_balanced_samples(
        samples, panel, per_superpop=2,
        cohort_key=COHORT_KEY, representation_key=REPRESENTATION_KEY,
    )
    second = select_balanced_samples(
        samples, panel, per_superpop=2,
        cohort_key=COHORT_KEY, representation_key=REPRESENTATION_KEY,
    )
    alternate = select_balanced_samples(
        samples, panel, per_superpop=2,
        cohort_key=COHORT_KEY, representation_key=b"z" * 32,
    )

    assert first == second
    assert first != [sample for sample in samples if sample in set(first)]
    assert alternate != first
    assert set(alternate) == set(first)
    assert {
        pop: sum(panel[sample].superpopulation == pop for sample in first)
        for pop in superpops
    } \
        == {pop: 2 for pop in superpops}


def test_streamed_subset_uses_opaque_keyed_columns_and_balances_labels(tmp_path):
    superpops = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    samples = [f"s_{pop}_{index}" for pop in superpops for index in range(3)]
    source = tmp_path / "source.vcf.gz"
    import gzip
    with gzip.open(source, "wt", newline="") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        fh.write("\t".join(samples) + "\n")
        for position in range(1, 301):
            calls = [f"{position % 2}|{(position + index) % 2}:{10 + index}"
                     for index in range(len(samples))]
            fh.write(
                f"22\t{position}\trs{position}\tA\tC\t.\tPASS\tX={position}\tGT:DP\t"
                + "\t".join(calls) + "\n"
            )
    panel = tmp_path / "panel.txt"
    panel.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        + "".join(
            f"{sample}\t{sample.split('_')[1]}_{int(sample.rsplit('_', 1)[1]) % 2}"
            f"\t{sample.split('_')[1]}\t0\n" for sample in samples
        )
    )
    output = tmp_path / "derived.vcf"

    result, alias_labels = write_reviewed_subset(
        str(source), str(panel), str(output),
        per_superpop=2, record_modulus=3,
        cohort_key=COHORT_KEY,
        record_key=RECORD_KEY,
        representation_key=REPRESENTATION_KEY,
        minimum_records=1,
    )

    lines = output.read_text().splitlines()
    header = next(line for line in lines if line.startswith("#CHROM")).split("\t")
    selected = header[9:]
    assert len(selected) == 10
    assert [selected.count(sample) for sample in selected] == [1] * 10
    assert all(sample.startswith("p") and len(sample) == 21 for sample in selected)
    assert not set(selected) & set(samples)
    assert {
        pop: sum(alias_labels[sample].superpopulation == pop for sample in selected)
        for pop in superpops
    } \
        == {pop: 2 for pop in superpops}
    selected_populations = {alias_labels[sample].population for sample in selected}
    assert len(selected_populations) == 10
    assert 60 <= result["records"] <= 140
    assert result["bytes"] == output.stat().st_size
    assert len(result["derived_sha256"]) == 64
    data = [line for line in lines if not line.startswith("#")]
    assert all(len(line.split("\t")) == 19 for line in data)
    assert all(line.split("\t")[0] == "chr22" for line in data)
    assert all(line.split("\t")[2] == "." for line in data)
    assert all(line.split("\t")[5:9] == [".", "PASS", ".", "GT"] for line in data)
    assert [int(line.split("\t")[1]) for line in data] == [
        10_000 + 10 * index for index in range(len(data))
    ]
    assert all(":" not in token for line in data for token in line.split("\t")[9:])
    assert not any("X=" in line or "rs" in line for line in data)


def test_release_selection_key_changes_record_sample():
    records = [f"22\t{position}\trs{position}\tA\tC\t.\tPASS\t.\tGT"
               for position in range(1, 301)]
    first = [_record_selected(record, 7, RECORD_KEY) for record in records]
    second = [_record_selected(record, 7, b"z" * 32) for record in records]

    assert first != second
    assert 25 <= sum(first) <= 65
    assert 25 <= sum(second) <= 65
