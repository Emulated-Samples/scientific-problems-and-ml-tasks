import os

import numpy as np
import pytest

from reference import fast_pca, pca_core
from reference.full_scan_pca import read_samples


def _record(fmt: bytes, samples: bytes) -> bytes:
    return b"1\t7\t.\tA\tC\t.\tPASS\t.\t" + fmt + b"\t" + samples


@pytest.mark.parametrize("decoder", [fast_pca._decode_lines, pca_core._decode_lines])
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (_record(b"GT", b"0/0\t0/1\t1/1\r"), [0, 1, 2]),
        (_record(b"GT:DP", b"0/0:8\t.:9\t1/1:7"), [0, -1, 2]),
        (_record(b"GT", b"0/0\t.\t1/1"), [0, -1, 2]),
        (_record(b"GT", b"0/\t0/00\t0/1"), [-1, -1, 1]),
        (_record(b"GT", b"0/0\t2/2\t0x1"), [0, -1, -1]),
        (_record(b"GT", b"1x0\t00\t11"), [-1, -1, -1]),
        (_record(b"GT", b"10/0\t0/0\t0/1"), [-1, 0, 1]),
        (_record(b"GT:DP", b"0|1:4\t./.:0\t1|1:8"), [1, -1, 2]),
        (_record(b"GT", b"0\t1\t1"), [0, 2, 2]),
        (_record(b"GT:DP", b"0:4\t.:0\t1:8"), [0, -1, 2]),
        (b"1\t7\t.\ta\tc\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1", [0, 1, 2]),
    ],
)
def test_decoders_preserve_valid_calls_and_mark_only_bad_calls_missing(decoder, line, expected):
    observed = decoder([line], 3)

    assert observed is not None
    np.testing.assert_array_equal(observed, np.asarray([expected], dtype=np.int8))


@pytest.mark.parametrize("decoder", [fast_pca._decode_lines, pca_core._decode_lines])
def test_decoder_rejects_records_without_a_gt_format(decoder):
    assert decoder([_record(b"DP:GQ", b"0/0\t0/1\t1/1")], 3) is None


def test_crlf_header_and_record_decode_end_to_end(tmp_path):
    path = tmp_path / "crlf.vcf"
    path.write_bytes(
        b"##fileformat=VCFv4.2\r\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\r\n"
        + _record(b"GT", b"0/0\t0/1\t1/1")
        + b"\r\n"
    )

    assert read_samples(path) == ["s1", "s2", "s3"]
    fd = os.open(path, os.O_RDONLY)
    try:
        sample_ids, body_start = fast_pca._read_header(fd)
        observed = fast_pca._parse_core(
            fd, body_start, path.stat().st_size - body_start, path.stat().st_size, 3, 4096,
        )
    finally:
        os.close(fd)

    assert sample_ids == ["s1", "s2", "s3"]
    np.testing.assert_array_equal(observed, np.asarray([[0, 1, 2]], dtype=np.int8))
