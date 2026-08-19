#!/usr/bin/env python3
"""Benchmark the fast reference PCA against standard population-genetics / PCA tools.

The question this answers: *is our from-scratch, sampling-based PCA actually faster than the
established tools a geneticist would reach for* -- plink2, plink1.9, flashpca, scikit-allel --
on the same machine, on the same VCF, computing the same object (an HWE-normalised Patterson
PCA), and **is it still correct** while doing so?

The comparison is deliberately fair rather than flattering:

  * **end-to-end** is measured from *VCF on disk -> PC scores written*, which is our tool's
    actual contract. Tools that cannot read a VCF (flashpca wants plink ``.bed``) pay their
    real conversion cost, reported as a separate stage so nothing is hidden.
  * **cold cache** is the headline number. Our tool's whole thesis is that it reads only a
    *fraction* of the file (random byte-offset sampling), so it only wins once the file is too
    big to sit in RAM. We drop the kernel page cache before every cold run (needs root) and
    also record *bytes actually read from disk* (via ``/usr/bin/time -v`` "File system inputs"),
    which is the concrete evidence for the sampling advantage. Warm cache is reported too, as
    the pessimistic case for us (it hands every streaming tool the whole file for free).
  * **accuracy** is scored for every tool the same way the grader scores a submission:
    structure-weighted subspace agreement against the full-scan anchor. A fast wrong answer is
    not a win, and the plot puts speed and accuracy on the same picture so you can see it.

Each tool is an :class:`Adapter`. Adding a tool = adding one adapter; unavailable tools
(binary not installed) are skipped with a note, so the same script runs on a laptop with only
our tool and on a box with the full zoo.

Usage:
    python scripts/bench_vs_tools.py <vcf> [<vcf> ...] --k 10 --repeat 3 \
        --out results.json [--cold] [--panel data/real/panel.txt]

Then render plots with ``scripts/plot_bench.py results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference.full_scan_pca import fit as full_scan_fit, read_samples   # noqa: E402
from grader.metrics.subspace import subspace_accuracy, structure_weights  # noqa: E402
from grader.metrics.structure import population_separation               # noqa: E402


# ---------------------------------------------------------------------------
# Cold-cache control + timing with disk-read accounting
# ---------------------------------------------------------------------------

def drop_caches() -> bool:
    """Flush the kernel page cache so the next read hits disk. Root + Linux only.

    Returns True if the cache was actually dropped, False otherwise (e.g. macOS, non-root) so
    callers can label their numbers honestly rather than silently reporting warm times as cold.
    """
    p = Path("/proc/sys/vm/drop_caches")
    if not p.exists() or os.geteuid() != 0:
        return False
    try:
        subprocess.run(["sync"], check=True)
        p.write_text("3\n")
        return True
    except (OSError, PermissionError, subprocess.CalledProcessError):
        return False


@dataclass
class RunResult:
    seconds: float
    ok: bool
    fs_input_bytes: int | None = None      # bytes read from disk (cold), via /usr/bin/time -v
    stages: dict = field(default_factory=dict)   # sub-stage wall-clock, e.g. convert vs solve
    note: str = ""


_HAVE_GNU_TIME = shutil.which("/usr/bin/time") is not None


def _time_cmd(cmd: list[str], measure_io: bool, timeout: float = 3600) -> tuple[float, int, int | None]:
    """Run cmd, return (wall_seconds, returncode, fs_input_bytes|None).

    When ``measure_io`` and GNU time is available we wrap in ``/usr/bin/time -v`` and parse
    "File system inputs" (reported in 512-byte sectors) -> bytes read from disk. This is the
    number that makes the sampling story concrete: on a cold cache a streaming tool's fs-inputs
    ~ file size, ours is a fraction of it.
    """
    fs_bytes = None
    if measure_io and _HAVE_GNU_TIME:
        with tempfile.NamedTemporaryFile("r+", suffix=".time") as tf:
            wrapped = ["/usr/bin/time", "-v", "-o", tf.name] + cmd
            t0 = time.perf_counter()
            proc = subprocess.run(wrapped, capture_output=True, timeout=timeout)
            dt = time.perf_counter() - t0
            tf.seek(0)
            for line in tf:
                if "File system inputs" in line:
                    try:
                        fs_bytes = int(line.split(":")[1].strip()) * 512
                    except (ValueError, IndexError):
                        pass
        return dt, proc.returncode, fs_bytes
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return time.perf_counter() - t0, proc.returncode, fs_bytes


# ---------------------------------------------------------------------------
# Tool adapters
# ---------------------------------------------------------------------------

class Adapter:
    """One PCA tool. Subclasses implement availability, a run, and score loading."""

    name = "base"
    kind = "external"          # "ours" | "anchor" | "external" -- only for plotting/coloring

    def available(self) -> bool:
        raise NotImplementedError

    def run(self, vcf: Path, k: int, workdir: Path, cold: bool) -> RunResult:
        """Execute VCF -> PC scores end-to-end, writing scores where ``load_scores`` finds them."""
        raise NotImplementedError

    def load_scores(self, workdir: Path, sample_ids: list[str]) -> np.ndarray | None:
        """Return scores aligned to ``sample_ids`` order, or None if unreadable."""
        raise NotImplementedError


def _align(rows: dict[str, list[float]], sample_ids: list[str]) -> np.ndarray | None:
    """Order a {sample_id: pcs} map into the anchor's sample order; None if any sample missing."""
    if any(s not in rows for s in sample_ids):
        return None
    return np.asarray([rows[s] for s in sample_ids], dtype=float)


class OursAdapter(Adapter):
    name = "ours (fast_pca)"
    kind = "ours"

    def __init__(self, pca_path: Path, interpreter: str):
        self.pca_path = pca_path
        self.interp = interpreter

    def available(self) -> bool:
        return self.pca_path.exists()

    def run(self, vcf, k, workdir, cold):
        out = workdir / "ours.tsv"
        if out.exists():
            out.unlink()
        if cold:
            drop_caches()
        dt, rc, io = _time_cmd([self.interp, str(self.pca_path), str(vcf), str(k), str(out)],
                               measure_io=cold)
        return RunResult(seconds=dt, ok=(rc == 0 and out.exists()), fs_input_bytes=io)

    def load_scores(self, workdir, sample_ids):
        return _load_tsv(workdir / "ours.tsv", sample_ids)


class FullScanAdapter(Adapter):
    """Our own full-scan numpy PCA: doubles as the accuracy anchor AND a 'careful pure-Python,
    no tricks' speed baseline -- the thing every external tool and our fast path are beating."""
    name = "full-scan (numpy anchor)"
    kind = "anchor"

    def __init__(self, interpreter: str):
        self.interp = interpreter

    def available(self) -> bool:
        return True

    def run(self, vcf, k, workdir, cold):
        out = workdir / "fullscan.tsv"
        if cold:
            drop_caches()
        dt, rc, io = _time_cmd([self.interp, "-m", "reference.full_scan_pca",
                                str(vcf), str(k), str(out)], measure_io=cold)
        return RunResult(seconds=dt, ok=(rc == 0 and out.exists()), fs_input_bytes=io)

    def load_scores(self, workdir, sample_ids):
        return _load_tsv(workdir / "fullscan.tsv", sample_ids)


class Plink2Adapter(Adapter):
    """plink2 --pca, reading the VCF directly. ``approx=True`` uses plink2's randomized solver
    (its recommended mode at scale). plink2's default --pca is variance-standardised, i.e. the
    same HWE/Patterson normalisation this task requires."""

    def __init__(self, approx: bool):
        self.approx = approx
        self.name = "plink2 --pca approx" if approx else "plink2 --pca"

    def available(self):
        return shutil.which("plink2") is not None

    def run(self, vcf, k, workdir, cold):
        stem = workdir / ("plink2a" if self.approx else "plink2")
        cmd = ["plink2", "--vcf", str(vcf), "--pca", str(k)]
        if self.approx:
            cmd += ["approx"]
        cmd += ["--threads", str(os.cpu_count() or 4), "--out", str(stem)]
        if cold:
            drop_caches()
        dt, rc, io = _time_cmd(cmd, measure_io=cold)
        return RunResult(seconds=dt, ok=(rc == 0 and Path(f"{stem}.eigenvec").exists()),
                         fs_input_bytes=io)

    def load_scores(self, workdir, sample_ids):
        stem = workdir / ("plink2a" if self.approx else "plink2")
        return _load_plink_eigenvec(Path(f"{stem}.eigenvec"), sample_ids, has_header=True)


class Plink19Adapter(Adapter):
    name = "plink1.9 --pca"

    def available(self):
        return shutil.which("plink") is not None

    def run(self, vcf, k, workdir, cold):
        stem = workdir / "plink19"
        cmd = ["plink", "--vcf", str(vcf), "--double-id", "--allow-extra-chr",
               "--pca", str(k), "--out", str(stem)]
        if cold:
            drop_caches()
        dt, rc, io = _time_cmd(cmd, measure_io=cold)
        return RunResult(seconds=dt, ok=(rc == 0 and Path(f"{stem}.eigenvec").exists()),
                         fs_input_bytes=io)

    def load_scores(self, workdir, sample_ids):
        return _load_plink_eigenvec(workdir / "plink19.eigenvec", sample_ids, has_header=False)


class FlashpcaAdapter(Adapter):
    """flashpca cannot read a VCF; it needs plink ``.bed``. So end-to-end = plink1.9 convert
    (VCF -> bed) + flashpca solve, and we report the two stages separately so the conversion
    tax is explicit. flashpca standardises with the binomial (Patterson) scaler by default."""
    name = "flashpca (+plink convert)"

    def available(self):
        return shutil.which("flashpca") is not None and shutil.which("plink") is not None

    def run(self, vcf, k, workdir, cold):
        bed = workdir / "fp_in"
        if cold:
            drop_caches()
        # stage 1: VCF -> bed (plink1.9)
        t_conv, rc1, io1 = _time_cmd(
            ["plink", "--vcf", str(vcf), "--double-id", "--allow-extra-chr",
             "--make-bed", "--out", str(bed)], measure_io=cold)
        if rc1 != 0 or not Path(f"{bed}.bed").exists():
            return RunResult(seconds=t_conv, ok=False, note="plink convert failed")
        # stage 2: flashpca solve (warm -- bed just written, this is solve-time proper)
        t_solve, rc2, _ = _time_cmd(
            ["flashpca", "--bfile", str(bed), "--ndim", str(k),
             "--outpc", str(workdir / "fp_pcs.txt"), "--outval", str(workdir / "fp_val.txt"),
             "--outvec", str(workdir / "fp_vec.txt"), "--outload", str(workdir / "fp_load.txt")],
            measure_io=False)
        ok = rc2 == 0 and (workdir / "fp_pcs.txt").exists()
        return RunResult(seconds=t_conv + t_solve, ok=ok,
                         fs_input_bytes=io1,
                         stages={"convert": round(t_conv, 3), "solve": round(t_solve, 3)})

    def load_scores(self, workdir, sample_ids):
        f = workdir / "fp_pcs.txt"
        if not f.exists():
            return None
        rows = {}
        for i, line in enumerate(f.read_text().splitlines()):
            parts = line.split()
            if i == 0 and parts and parts[0].upper() in ("FID", "#FID"):
                continue                              # header
            if len(parts) < 3:
                continue
            rows[parts[1]] = [float(x) for x in parts[2:]]     # FID IID PC1..
        return _align(rows, sample_ids)


class ScikitAllelAdapter(Adapter):
    """scikit-allel: a mainstream Python genomics library that reads the VCF and runs a Patterson
    PCA (``allel.pca(..., scaler='patterson')``). Notably this is one of the libraries this task
    *bans* for a submission -- here we run it purely as a speed/accuracy reference point. Uses an
    isolated interpreter so the task's numpy/scipy-only env stays clean."""
    name = "scikit-allel"

    def __init__(self, interpreter: str | None):
        self.interp = interpreter

    def available(self):
        if not self.interp or not Path(self.interp).exists():
            return False
        r = subprocess.run([self.interp, "-c", "import allel"], capture_output=True)
        return r.returncode == 0

    def run(self, vcf, k, workdir, cold):
        out = workdir / "allel.tsv"
        script = workdir / "_allel_run.py"
        script.write_text(_ALLEL_SCRIPT)
        if cold:
            drop_caches()
        dt, rc, io = _time_cmd([self.interp, str(script), str(vcf), str(k), str(out)],
                               measure_io=cold)
        return RunResult(seconds=dt, ok=(rc == 0 and out.exists()), fs_input_bytes=io)

    def load_scores(self, workdir, sample_ids):
        return _load_tsv(workdir / "allel.tsv", sample_ids)


# scikit-allel driver: VCF -> Patterson PCA -> TSV in our score format.
_ALLEL_SCRIPT = r'''
import sys, numpy as np, allel
vcf, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cb = allel.read_vcf(vcf, fields=["samples", "calldata/GT", "variants/ALT", "variants/REF"])
samples = list(cb["samples"])
gt = allel.GenotypeArray(cb["calldata/GT"])
ref, alt = cb["variants/REF"], cb["variants/ALT"]
# biallelic SNV mask: single-base REF, single-base first ALT, no 2nd ALT
alt1 = alt[:, 0].astype(str); alt2 = alt[:, 1].astype(str) if alt.shape[1] > 1 else np.array([""]*len(alt1))
snv = (np.char.str_len(ref.astype(str)) == 1) & (np.char.str_len(alt1) == 1) & (alt2 == "")
gt = gt.compress(snv, axis=0)
ac = gt.count_alleles()
# keep segregating (non-monomorphic) biallelic
seg = ac.is_segregating() & (ac.shape[1] == 2 if ac.shape[1] <= 2 else False)
gn = gt.to_n_alt(fill=-1)[seg]      # dosage 0/1/2, -1 missing (int8)
del gt, cb, ac                       # free the diploid GT array before the float blow-up
gn = np.where(gn < 0, np.nan, gn).astype(np.float32)   # float32: half the RAM of float64
# mean-impute missing per variant
mean = np.nanmean(gn, axis=1, keepdims=True)
gn = np.where(np.isnan(gn), mean, gn).astype(np.float32)
coords, model = allel.pca(gn, n_components=k, scaler="patterson")
with open(out, "w") as fh:
    fh.write("sample_id\t" + "\t".join(f"PC{i+1}" for i in range(coords.shape[1])) + "\n")
    for sid, row in zip(samples, coords):
        fh.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")
'''


# ---------------------------------------------------------------------------
# Score file loaders
# ---------------------------------------------------------------------------

def _load_tsv(path: Path, sample_ids: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return None
    rows = {}
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        try:
            rows[parts[0]] = [float(x) for x in parts[1:]]
        except ValueError:
            return None
    return _align(rows, sample_ids)


def _load_plink_eigenvec(path: Path, sample_ids: list[str], has_header: bool) -> np.ndarray | None:
    if not path.exists():
        return None
    rows = {}
    for i, line in enumerate(path.read_text().splitlines()):
        parts = line.split()
        if has_header and i == 0:
            continue
        if len(parts) < 3:
            continue
        try:
            rows[parts[1]] = [float(x) for x in parts[2:]]     # FID IID PC1..
        except ValueError:
            return None
    return _align(rows, sample_ids)


# ---------------------------------------------------------------------------
# Anchor (cached per vcf,k): defines truth subspace + weights + speed baseline
# ---------------------------------------------------------------------------

def compute_anchor(vcf: Path, k: int) -> dict:
    t0 = time.perf_counter()
    sample_ids, scores, kept, spectrum = full_scan_fit(vcf, k)
    dt = time.perf_counter() - t0
    weights = structure_weights(spectrum, scores.shape[1])[:scores.shape[1]]
    return {"sample_ids": sample_ids, "scores": scores, "weights": weights,
            "seconds": dt, "kept": kept}


# ---------------------------------------------------------------------------
# Benchmark one (vcf, tool)
# ---------------------------------------------------------------------------

def bench_tool(adapter: Adapter, vcf: Path, k: int, anchor: dict, labels,
               repeat: int, cold: bool, workdir: Path) -> dict:
    sample_ids = anchor["sample_ids"]
    # The full-scan anchor is minutes-long and *identical* every run, so re-timing it ``repeat``
    # times is pure waste; one warm + one cold is plenty. Fast external tools still get ``repeat``.
    reps = 1 if adapter.kind == "anchor" else repeat
    # Warm timing: min of ``reps`` runs (robust to a loaded box). Correctness from last run.
    warm_times, last_ok = [], False
    for _ in range(reps):
        r = adapter.run(vcf, k, workdir, cold=False)
        warm_times.append(r.seconds)
        last_ok = r.ok
    warm = min(warm_times) if warm_times else float("nan")

    # Accuracy from the (warm) output.
    acc, ncc_ratio = 0.0, None
    S = adapter.load_scores(workdir, sample_ids) if last_ok else None
    if S is not None and np.isfinite(S).all():
        acc = subspace_accuracy(S, anchor["scores"], anchor["weights"])["accuracy"]
        if labels is not None and len(set(x for x in labels if x >= 0)) >= 2:
            lab = np.asarray(labels)
            sub = population_separation(S, lab)["ncc_accuracy"]
            ref = population_separation(anchor["scores"], lab)["ncc_accuracy"]
            ncc_ratio = round(sub / ref, 4) if ref > 1e-6 else None

    # Cold timing + disk-read accounting: a single run per re-drop (the drop is the expensive
    # part), taken as the min of a few so a stray page fault does not dominate.
    cold_time, cold_bytes = None, None
    if cold:
        cts, cbs = [], []
        for _ in range(reps):
            r = adapter.run(vcf, k, workdir, cold=True)
            if r.ok:
                cts.append(r.seconds)
                if r.fs_input_bytes:
                    cbs.append(r.fs_input_bytes)
        cold_time = min(cts) if cts else None
        cold_bytes = int(np.median(cbs)) if cbs else None

    return {"tool": adapter.name, "kind": adapter.kind, "ok": bool(last_ok),
            "warm_seconds": round(warm, 3), "cold_seconds": round(cold_time, 3) if cold_time else None,
            "cold_read_bytes": cold_bytes, "accuracy": round(float(acc), 4),
            "ncc_ratio_vs_anchor": ncc_ratio, "last_stages": getattr(S, "stages", None)}


def build_adapters(repo: Path, interpreter: str, allel_interp: str | None) -> list[Adapter]:
    ours = repo / "reference" / "fast_submission" / "pca"
    return [
        OursAdapter(ours, interpreter),
        Plink2Adapter(approx=False),
        Plink2Adapter(approx=True),
        Plink19Adapter(),
        FlashpcaAdapter(),
        ScikitAllelAdapter(allel_interp),
        FullScanAdapter(interpreter),
    ]


def load_labels(vcf: Path):
    """Ground-truth population labels, if a ``<vcf>.truth.json`` sidecar exists (synthetic) --
    real-data labels are joined separately by the caller."""
    truth = Path(str(vcf) + ".truth.json")
    if truth.exists():
        return json.loads(truth.read_text()).get("sample_pop")
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vcfs", nargs="+")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cold", action="store_true", help="also measure cold-cache (needs root+Linux)")
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--interpreter", default=sys.executable)
    ap.add_argument("--allel-interpreter", default=None,
                    help="python with scikit-allel installed (isolated env)")
    ap.add_argument("--panel", default=None, help="real-data panel: sample<TAB>pop, for labels")
    a = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    adapters = [ad for ad in build_adapters(repo, a.interpreter, a.allel_interpreter)]
    workdir = Path(tempfile.mkdtemp(prefix="benchvs_"))
    cold_ok = drop_caches() if a.cold else False
    if a.cold and not cold_ok:
        print("WARNING: --cold requested but page cache could not be dropped "
              "(need root on Linux); cold numbers will equal warm.", file=sys.stderr)

    results = {"meta": {"k": a.k, "repeat": a.repeat, "cold": a.cold and cold_ok,
                        "cpu_count": os.cpu_count()}, "datasets": []}
    for vpath in a.vcfs:
        vcf = Path(vpath)
        print(f"\n=== {vcf.name} ({vcf.stat().st_size/1e9:.2f} GB) ===", file=sys.stderr)
        print("  computing full-scan anchor (one-time)...", file=sys.stderr)
        anchor = compute_anchor(vcf, a.k)
        labels = load_labels(vcf)
        if a.panel:
            labels = _labels_from_panel(a.panel, anchor["sample_ids"]) or labels
        ds = {"vcf": vcf.name, "size_bytes": vcf.stat().st_size,
              "n_samples": len(anchor["sample_ids"]), "kept_variants": anchor["kept"],
              "anchor_seconds": round(anchor["seconds"], 3), "tools": []}
        for ad in adapters:
            if not ad.available():
                print(f"  - {ad.name:28s} SKIP (not installed)", file=sys.stderr)
                continue
            res = bench_tool(ad, vcf, a.k, anchor, labels, a.repeat, a.cold and cold_ok, workdir)
            ds["tools"].append(res)
            ct = f" cold={res['cold_seconds']}s" if res["cold_seconds"] else ""
            print(f"  - {ad.name:28s} warm={res['warm_seconds']:.2f}s{ct} "
                  f"acc={res['accuracy']:.3f} ok={res['ok']}", file=sys.stderr)
        results["datasets"].append(ds)

    Path(a.out).write_text(json.dumps(results, indent=2))
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


def _labels_from_panel(panel: str, sample_ids: list[str]):
    """Join a 1000G-style panel (sample<TAB>pop<TAB>super_pop...) into integer labels ordered
    like ``sample_ids``. Missing samples get -1 (ignored by the separation metric)."""
    mp = {}
    with open(panel) as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3 or (i == 0 and parts[1].lower() in ("pop", "population")):
                continue
            mp[parts[0]] = parts[2]        # super-population
    uniq = sorted(set(mp.values()))
    idx = {p: i for i, p in enumerate(uniq)}
    return [idx.get(mp.get(s, None), -1) for s in sample_ids]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
