import os

import numpy as np

from reference import fast_pca
from reference.fast_pca import (_core_bytes, _density_bounds, _plan_cores,
                                _rotate_cores)


def _selected_bytes(cores):
    selected = set()
    for core in cores:
        for offset, length in core:
            selected.update(range(offset, offset + length))
    return selected


def test_circular_rotation_gives_every_byte_exact_inclusion_probability():
    body_start = 17
    body_len = 31
    starts = np.array([0, 10, 20])
    core_len = 3
    counts = np.zeros(body_len, dtype=int)

    for rotation in range(body_len):
        cores = _rotate_cores(body_start, body_len, starts, core_len, rotation)
        selected = _selected_bytes(cores)
        assert len(selected) == len(starts) * core_len
        for offset in selected:
            counts[offset - body_start] += 1

    assert np.all(counts == len(starts) * core_len)


def test_partial_plan_is_nonoverlapping_and_has_fixed_byte_budget():
    cores, full = _plan_cores(
        body_start=100, body_len=10_003, core_len=137, n_blocks=31,
        rng=np.random.default_rng(4),
    )
    selected = _selected_bytes(cores)

    assert not full
    assert _core_bytes(cores) == 31 * 137
    assert len(selected) == 31 * 137


def test_sufficient_budget_tiles_entire_body_including_short_tail():
    body_start = 23
    body_len = 199
    cores, full = _plan_cores(
        body_start, body_len, core_len=100, n_blocks=2, rng=np.random.default_rng(5),
    )

    assert full
    assert _core_bytes(cores) == body_len
    assert _selected_bytes(cores) == set(range(body_start, body_start + body_len))


def test_full_tiling_decodes_long_records_once_without_terminal_newline(tmp_path):
    path = tmp_path / "long.vcf"
    header = (
        b"##fileformat=VCFv4.2\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
    )
    records = [
        b"1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1",
        b"1\t2\t.\tA\tC\t.\tPASS\tX=" + (b"x" * 5000) + b"\tGT\t0/0\t0/1\t1/1",
        b"1\t3\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1",
    ]
    path.write_bytes(header + b"\n".join(records))

    fd = os.open(path, os.O_RDONLY)
    try:
        _, body_start = fast_pca._read_header(fd)
        file_size = path.stat().st_size
        cores, full = _plan_cores(
            body_start, file_size - body_start, 127, 100, np.random.default_rng(1),
        )
        decoded = [
            fast_pca._parse_logical_core(fd, spans, file_size, 3, 64)
            for spans in cores
        ]
    finally:
        os.close(fd)

    observed = np.vstack([block for block in decoded if block is not None])
    assert full
    np.testing.assert_array_equal(observed, np.tile([0, 1, 2], (3, 1)))


def test_core_wholly_inside_long_record_does_not_scan_to_newline(tmp_path):
    path = tmp_path / "record.txt"
    path.write_bytes(b"start\t" + (b"x" * (1 << 20)) + b"\nnext\n")
    fd = os.open(path, os.O_RDONLY)
    try:
        data, at_start = fast_pca._read_core(fd, 1000, 128, path.stat().st_size, 64)
    finally:
        os.close(fd)

    assert not at_start
    assert data == b""


def test_final_sample_is_accepted_without_marker_count_conditioning(tmp_path, monkeypatch):
    path = tmp_path / "sparse.vcf"
    header = (
        b"##fileformat=VCFv4.2\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.seek(len(header) + (16 << 20) - 1)
        fh.write(b"\n")

    monkeypatch.setattr(
        fast_pca, "_count_cores",
        lambda fd, cores, *args: np.full(len(cores), 10, dtype=np.int64),
    )
    worker_calls = 0

    def fake_worker(fd, cores, n_samples, file_size, tail_chunk):
        nonlocal worker_calls
        worker_calls += 1
        return np.eye(n_samples, dtype=np.float32) * 4_000, 4_000

    monkeypatch.setattr(fast_pca, "_worker_gram", fake_worker)

    _, scores, kept, meta = fast_pca.fit(
        path, 2, n_blocks=4, block_size=4096, n_workers=1, engine="gram",
    )

    assert scores.shape == (4, 2)
    assert kept == 4_000
    assert worker_calls == 1
    assert not meta["marker_target_met"]


def test_density_floor_is_conservative_across_probe_strata():
    cores = [((i * 100, 100),) for i in range(16)]
    uniform = np.full(16, 10)
    heterogeneous = np.asarray([0] * 8 + [20] * 8)

    uniform_bounds = _density_bounds(uniform, cores)
    heterogeneous_bounds = _density_bounds(heterogeneous, cores)
    assert uniform_bounds == (0.1, 0.1, 0.1)
    assert 0 <= heterogeneous_bounds[0] < heterogeneous_bounds[1] < heterogeneous_bounds[2]


def test_representative_probe_ignores_huge_first_record_for_auto_engine(tmp_path):
    n_samples = 128
    sample_ids = [f"s{i}".encode() for i in range(n_samples)]
    path = tmp_path / "first-record-outlier.vcf"
    header = (
        b"##fileformat=VCFv4.2\n"
        + b"\t".join([
            b"#CHROM", b"POS", b"ID", b"REF", b"ALT", b"QUAL", b"FILTER", b"INFO",
            b"FORMAT", *sample_ids,
        ])
        + b"\n"
    )
    calls = [b"0/0", b"0/1", b"1/1"]
    with open(path, "wb") as fh:
        fh.write(header)
        for variant in range(5_000):
            info = b"X=" + (b"x" * (1 << 20)) if variant == 0 else b"."
            genotypes = b"\t".join(calls[(sample + variant) % 3]
                                   for sample in range(n_samples))
            fh.write(b"\t".join([
                b"1", str(variant + 1).encode(), b".", b"A", b"C", b".", b"PASS", info,
                b"GT", genotypes,
            ]) + b"\n")

    _, scores, kept, meta = fast_pca.fit(
        path, 2, n_blocks=1, block_size=4096, n_workers=1, engine="auto",
    )

    assert scores.shape == (n_samples, 2)
    assert kept == 5_000
    assert meta["engine"] == "gram"
    assert meta["estimated_variants"] > n_samples
    assert meta["full_coverage"]


def test_default_block_setting_does_not_force_unneeded_full_coverage(tmp_path, monkeypatch):
    path = tmp_path / "large-body.vcf"
    header = (
        b"##fileformat=VCFv4.2\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.seek(len(header) + (16 << 20) - 1)
        fh.write(b"\n")

    monkeypatch.setattr(
        fast_pca, "_count_cores",
        lambda fd, cores, *args: np.full(len(cores), 100, dtype=np.int64),
    )

    def fake_worker(fd, cores, n_samples, file_size, tail_chunk):
        return np.eye(n_samples, dtype=np.float32) * 100_000, 100_000

    monkeypatch.setattr(fast_pca, "_worker_gram", fake_worker)
    _, _, _, meta = fast_pca.fit(path, 2, engine="gram")

    assert not meta["full_coverage"]
    # Automatic worker selection must not stand up the parser pipeline below its measured
    # break-even. This plan has hundreds of partial-read cores, so without the size guard it would
    # expose the host CPU count here rather than one worker.
    assert meta["n_workers"] == 1
    assert meta["n_blocks"] < fast_pca.DEFAULT_BLOCKS
    assert meta["n_blocks"] >= fast_pca.MIN_SAMPLING_STRATA
    assert meta["final_selected_bytes"] < path.stat().st_size - len(header)

    monkeypatch.setattr(
        fast_pca, "_pipeline_gram",
        lambda fd, cores, n_samples, *args: (
            np.eye(n_samples, dtype=np.float32) * 100_000,
            100_000,
        ),
    )
    _, _, _, constrained = fast_pca.fit(
        path, 2, n_blocks=32, n_workers=2, engine="gram",
    )
    assert constrained["n_blocks"] == 32
    # The small-file heuristic applies only to the automatic choice. An explicit resource request
    # remains a contract and must not be silently rewritten.
    assert constrained["n_workers"] == 2

    monkeypatch.setattr(
        fast_pca, "_count_cores",
        lambda fd, cores, *args: np.zeros(len(cores), dtype=np.int64),
    )
    _, _, _, complete = fast_pca.fit(path, 2, n_workers=1, engine="gram")
    assert complete["full_coverage"]
    # A total read is a sequential STREAM, not a scattered sample, and the two want opposite core
    # sizes. Spatial strata exist to spread a partial read across the genome; when the read is total
    # they spread nothing and only multiply parse calls -- tiling this 16 MiB body into block-sized
    # cores cost 64 separate decodes, whose per-call overhead is what made the sampler lose to a
    # plain full scan. Full coverage must therefore take a handful of long contiguous runs, sized
    # for throughput (one per worker, capped), not one core per block_size.
    assert complete["n_blocks"] <= max(1, complete["n_workers"])
    assert complete["n_blocks"] * fast_pca.SEQUENTIAL_CORE_BYTES >= complete["final_selected_bytes"]


def test_cli_reports_under_target_accuracy_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        fast_pca,
        "fit",
        lambda *args, **kwargs: (
            ["s1", "s2"], np.zeros((2, 1)), 4_000,
            {"marker_target_met": False, "required_variants": 8_192,
             "selected_bytes": 100, "file_size": 1_000, "n_workers": 1},
        ),
    )

    assert fast_pca.main(["unused.vcf", "1", str(tmp_path / "scores.tsv")]) == 0
    assert "retained 4000 informative variants; accuracy target is 8192" in capsys.readouterr().err
