"""Probe inputs must not advertise themselves through trivial dimensions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.generate import grade_specs
from grader.gates import probes
from grader import grade


def _active_release():
    root = Path(__file__).resolve().parents[1]
    math_key = bytes.fromhex(
        (root / "grader/private/release-v3.math-key").read_text().strip()
    )
    carriers = tuple(
        probes.ProbeCarrier(
            spec.n_samples,
            spec.n_variants,
            spec.k,
            "synthetic_variable_width" if spec.representation == "variable_width"
            else "synthetic_haploid" if spec.representation == "haploid_gt"
            else "synthetic_messy" if spec.messy_frac > 0
            else "synthetic_plain",
        )
        for spec in grade_specs(math_key)
    )
    return math_key, carriers


def test_private_probe_plan_uses_complete_realized_release_tuples():
    math_key, active = _active_release()
    plan = probes.select_probe_carriers(active, math_key)
    selected = tuple(carrier for carriers in plan.values() for carrier in carriers)

    assert set(plan) == {"hwe_norm", "coverage", "representation_equivalence"}
    assert set(selected) <= set(active)
    assert len({carrier.k for carrier in selected}) >= 3
    assert len(selected) == sum(probes._PROBE_REPETITIONS.values())
    assert all(carrier.k == next(
        spec.k for spec in grade_specs(math_key)
        if (spec.n_samples, spec.n_variants, spec.k) == tuple(carrier[:3])
    ) for carrier in selected)
    assert all(
        carrier.layout == "synthetic_plain"
        for gate in ("hwe_norm", "coverage") for carrier in plan[gate]
    )
    assert all(
        carrier.layout == "synthetic_variable_width"
        for carrier in plan["representation_equivalence"]
    )


def test_hwe_carriers_are_active_complete_tuples_with_their_actual_rank():
    math_key, active = _active_release()
    carriers = probes.select_probe_carriers(active, math_key)["hwe_norm"]
    rng = np.random.default_rng(117)

    assert set(carriers) <= set(active)
    for carrier in carriers:
        assert probes._walsh_carrier_compatible(carrier)
        axes = probes._walsh_axes(carrier.n_samples, carrier.k + 1, rng)
        gram = axes.astype(np.int64) @ axes.astype(np.int64).T
        assert np.array_equal(
            gram, np.eye(carrier.k + 1, dtype=np.int64) * carrier.n_samples)
        assert np.all(axes.sum(axis=1) == 0)


def test_truth_contract_extracts_complete_synthetic_and_observed_carriers():
    math_key, _ = _active_release()
    synthetic = grade_specs(math_key)[0]
    synthetic_truth = {
        "n_samples": synthetic.n_samples,
        "n_variants": synthetic.n_variants,
        "spec": {"k": synthetic.k, "representation": synthetic.representation,
                 "messy_frac": synthetic.messy_frac},
    }
    observed_truth = {
        "n_samples": 500,
        "spec": {"n_records": 73_219, "k": 8,
                 "source": "1000genomes_phase3_grch37"},
    }

    assert grade._probe_carrier_from_truth(synthetic_truth) == probes.ProbeCarrier(
        synthetic.n_samples, synthetic.n_variants, synthetic.k, "synthetic_plain")
    assert grade._probe_carrier_from_truth(observed_truth) == probes.ProbeCarrier(
        500, 73_219, 8, "observed")

    variable_truth = {
        "n_samples": 360,
        "n_variants": 50_000,
        "spec": {"k": 5, "representation": "variable_width", "messy_frac": 0.0},
    }
    messy_truth = {
        "n_samples": 500,
        "n_variants": 120_000,
        "spec": {"k": 10, "representation": "plain_gt", "messy_frac": 0.12},
    }
    assert grade._probe_carrier_from_truth(variable_truth).layout == \
        "synthetic_variable_width"
    assert grade._probe_carrier_from_truth(messy_truth).layout == "synthetic_messy"
    haploid_truth = {
        "n_samples": 320,
        "n_variants": 30_000,
        "spec": {"k": 8, "representation": "haploid_gt", "messy_frac": 0.0},
    }
    assert grade._probe_carrier_from_truth(haploid_truth).layout == "synthetic_haploid"


def test_plain_probe_layout_has_no_probe_only_format_or_chromosome_tiling(tmp_path):
    rng = np.random.default_rng(812)
    carrier = probes.ProbeCarrier(24, 220, 3, "synthetic_plain")
    G = rng.integers(0, 3, size=(carrier.n_records, carrier.n_samples), dtype=np.int8)
    path = tmp_path / "opaque.vcf"
    probes._write_plain_probe_vcf(
        path, G, [f"p{index:020x}" for index in range(carrier.n_samples)], rng, carrier)

    records = [line.split(b"\t") for line in path.read_bytes().splitlines()
               if line and not line.startswith(b"#")]
    assert {record[8] for record in records} == {b"GT"}
    assert {record[2] for record in records} == {b"."}
    chromosome_counts = np.unique(
        [record[0] for record in records], return_counts=True)[1]
    assert np.unique(chromosome_counts).size > 1


def test_three_field_carrier_is_rejected_instead_of_guessing_a_layout():
    with pytest.raises(ValueError, match="must include"):
        probes.select_probe_carriers([(320, 40_000, 5)], bytes(range(32)))


def test_runtime_probe_rekey_preserves_exact_object_and_axis_capture():
    rng = np.random.default_rng(181)
    frequencies = rng.uniform(0.03, 0.97, size=(1200, 1))
    G = rng.binomial(2, frequencies, size=(1200, 96)).astype(np.int8)
    G[rng.random(G.shape) < 0.07] = -1
    axis = rng.normal(size=G.shape[1])

    first, order_a, ids_rng_a = probes._keyed_representation(
        G, bytes(range(32)), "unit",
    )
    second, order_b, ids_rng_b = probes._keyed_representation(
        G, bytes(reversed(range(32))), "unit",
    )
    scores_a, spectrum_a = probes._patterson_pca(first, 6)
    scores_b, spectrum_b = probes._patterson_pca(second, 6)

    assert not np.array_equal(first, second)
    np.testing.assert_allclose(spectrum_a, spectrum_b, rtol=1e-12, atol=1e-12)
    assert probes._axis_capture(scores_a, axis[order_a]) == pytest.approx(
        probes._axis_capture(scores_b, axis[order_b]), abs=1e-12,
    )
    ids_a = probes._rand_sample_ids(G.shape[1], ids_rng_a)
    ids_b = probes._rand_sample_ids(G.shape[1], ids_rng_b)
    assert ids_a != ids_b
    assert all(len(sample) == 21 and sample.startswith("p") for sample in [*ids_a, *ids_b])
