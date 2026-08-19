"""Experiment: does a low-discrepancy (Sobol / van der Corput) or finer-stratified
byte-offset sequence beat the reference's RANDOM-PHASE SYSTEMATIC offsets for
sampling variants from a position-sorted, LD-structured VCF?

The reference (reference/fast_pca.py) picks N nonoverlapping "cores" of contiguous
variants and standardises them into a sample-by-sample Patterson Gram, then takes the
top-k eigen-subspace. The claim we test: because a position-sorted VCF has within-region
LD (neighbouring variants correlated), spreading the cores more EVENLY across the genome
(Sobol / finer stratification) reduces the chance two cores land in the same LD block,
lowering the variance of the estimated top-k subspace at a fixed byte/variant budget.

We build a position-sorted genotype "file" (n_variants x n_samples int8) with:
  * Balding-Nichols population structure (the SIGNAL a correct PCA must recover), and
  * explicit LD blocks: contiguous runs of variants driven by the SAME per-population
    frequency vector (so neighbours are correlated -> a contiguous core reads redundant
    copies of one block's structure), with independent binomial noise per variant.

A "core" at start index s reads a contiguous run of c variants [s, s+c). The sampling
DESIGN chooses the core start offsets in [0, n_variants - c]. We compare, at a MATCHED
budget (same total variants read):
  (a) IID uniform random offsets
  (b) random-phase systematic (the reference's current design)
  (c) Sobol low-discrepancy (scrambled) offsets
  (d) van der Corput (base-2 radical inverse) offsets
  (e) finer stratification: 2x as many cores, half the core length (same budget)

Metric: min canonical correlation between the top-k SAMPLE score subspace of the sampled
Gram and the FULL-data Gram, on the structured PCs. Averaged over many seeds so we compare
MEAN and VARIANCE (std). Pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


# --------------------------------------------------------------------------- data
def make_ld_file(n_samples=300, n_pops=4, fst=0.02, n_blocks=2500, block_len=16,
                 seed=0):
    """Position-sorted (n_variants x n_samples) int8 genotype matrix with LD blocks.

    Each LD block b has ONE per-population frequency vector (Balding-Nichols draw over the
    K pops); all `block_len` contiguous variants in the block are Binomial(2, freq of that
    sample's pop) draws from THAT block's frequencies -> within-block variants are strongly
    correlated (they carry the same structure axis), across-block independent. Blocks are
    laid out contiguously => position-sorted with realistic within-region LD.
    """
    rng = np.random.default_rng(seed)
    m = n_samples
    sample_pop = rng.integers(0, n_pops, size=m).astype(np.int32)

    n_variants = n_blocks * block_len
    G = np.empty((n_variants, m), dtype=np.int8)

    # ancestral freqs, one per BLOCK (the LD "tag")
    p_anc = np.clip(rng.beta(0.7, 0.7, size=n_blocks), 0.02, 0.98)
    a = p_anc * (1.0 - fst) / fst
    b = (1.0 - p_anc) * (1.0 - fst) / fst
    # per-block per-pop freqs (n_blocks, n_pops)
    pop_freqs = np.empty((n_blocks, n_pops))
    for k in range(n_pops):
        pop_freqs[:, k] = np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)

    # per-sample freq for each block = its pop's freq
    for bi in range(n_blocks):
        f_by_sample = pop_freqs[bi][sample_pop]              # (m,)
        # block_len correlated variants: same freq vector, independent binomial noise
        block = rng.binomial(2, f_by_sample[None, :], size=(block_len, m))
        G[bi * block_len:(bi + 1) * block_len] = block.astype(np.int8)

    return G, sample_pop, n_pops


# --------------------------------------------------------------------------- PCA
def patterson_gram(V):
    """Patterson-standardise (n_variants x n_samples) dosages and return sample Gram Z^T Z
    and the count of kept variants. Mirrors reference standardize_into_gram (no missing)."""
    X = V.astype(np.float64)
    p = X.mean(axis=1) / 2.0
    maf = np.minimum(p, 1 - p)
    scale = np.sqrt(2.0 * p * (1.0 - p))
    keep = (maf > 1e-7) & (scale > 1e-7)
    if not keep.any():
        return None, 0
    Xk = X[keep]
    pk = (2.0 * p[keep])[:, None]
    sk = scale[keep][:, None]
    Z = (Xk - pk) / sk
    return Z.T @ Z, int(keep.sum())


def topk_scores(gram, n_kept, k):
    G = gram / max(n_kept, 1)
    ev, V = np.linalg.eigh(G)
    ev = ev[::-1][:k]
    V = V[:, ::-1][:, :k]
    ev = np.clip(ev, 0, None)
    return V * np.sqrt(ev)[None, :]


def subspace_cc(S1, S2, ncmp):
    """Min canonical correlation between two top-ncmp score subspaces (sign/rotation inv)."""
    ncmp = min(ncmp, S1.shape[1], S2.shape[1])
    if ncmp < 1:
        return 1.0
    Q1, _ = np.linalg.qr(S1[:, :ncmp] - S1[:, :ncmp].mean(0))
    Q2, _ = np.linalg.qr(S2[:, :ncmp] - S2[:, :ncmp].mean(0))
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(s.min()) if s.size else 1.0


# --------------------------------------------------------------------------- designs
def offsets_iid(n_variants, c, N, rng):
    hi = n_variants - c
    return np.sort(rng.integers(0, hi + 1, size=N))


def offsets_systematic(n_variants, c, N, rng):
    """Reference design: nonoverlapping random-phase systematic."""
    body = n_variants
    gap = body / N
    phase = float(rng.random()) * max(gap - c, 0.0)
    offs = (phase + gap * np.arange(N)).astype(np.int64)
    return np.clip(offs, 0, n_variants - c)


def offsets_sobol(n_variants, c, N, rng):
    """Scrambled Sobol low-discrepancy 1D sequence scaled to valid start range."""
    eng = qmc.Sobol(d=1, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
    u = eng.random(N).ravel()                                # N points in [0,1)
    offs = (u * (n_variants - c)).astype(np.int64)
    return np.sort(np.clip(offs, 0, n_variants - c))


def offsets_vdc(n_variants, c, N, rng):
    """Van der Corput base-2 radical inverse, random digit-scrambled (Owen-lite)."""
    idx = np.arange(N) + int(rng.integers(1, 2**20))
    # radical inverse base 2
    u = np.zeros(N)
    denom = 2.0
    x = idx.copy()
    while x.any():
        u += (x & 1) / denom
        x >>= 1
        denom *= 2.0
    # random shift (Cranley-Patterson rotation) to decorrelate from a fixed grid
    u = (u + rng.random()) % 1.0
    offs = (u * (n_variants - c)).astype(np.int64)
    return np.sort(np.clip(offs, 0, n_variants - c))


def gather_gram(V, offs, c):
    """Read contiguous cores [s, s+c) at each start, stack, and form the sample Gram.
    Cores may overlap for IID/Sobol/VdC (that is inherent to those designs); systematic
    is nonoverlapping by construction."""
    rows = []
    for s in offs:
        rows.append(V[s:s + c])
    stacked = np.vstack(rows)
    return patterson_gram(stacked)


# --------------------------------------------------------------------------- run
def run(seeds=40, n_samples=300, n_pops=4, fst=0.02, n_blocks=2500, block_len=16,
        core_len=8, n_cores=160, verbose=True):
    """For a fixed budget (n_cores * core_len variants), compare designs across seeds.
    Uses a DIFFERENT file per seed (so we average over data realisations too) plus a fresh
    sampling draw; the truth subspace is recomputed per file from the FULL Gram."""
    k = n_pops - 1
    budget = n_cores * core_len

    designs = {
        "iid": lambda nv, rng: offsets_iid(nv, core_len, n_cores, rng),
        "systematic": lambda nv, rng: offsets_systematic(nv, core_len, n_cores, rng),
        "sobol": lambda nv, rng: offsets_sobol(nv, core_len, n_cores, rng),
        "vdc": lambda nv, rng: offsets_vdc(nv, core_len, n_cores, rng),
        # finer stratification: 2x cores, half core length -> SAME budget
        "finer2x": lambda nv, rng: (offsets_systematic(nv, core_len // 2, n_cores * 2, rng),
                                    core_len // 2),
        "finer4x": lambda nv, rng: (offsets_systematic(nv, core_len // 4, n_cores * 4, rng),
                                    core_len // 4),
    }

    acc = {name: [] for name in designs}
    for si in range(seeds):
        V, sample_pop, npops = make_ld_file(n_samples, n_pops, fst, n_blocks, block_len,
                                            seed=1000 + si)
        nv = V.shape[0]
        full_gram, full_kept = patterson_gram(V)
        truth = topk_scores(full_gram, full_kept, k)
        rng = np.random.default_rng(9000 + si)
        for name, fn in designs.items():
            out = fn(nv, rng)
            if isinstance(out, tuple):
                offs, c = out
            else:
                offs, c = out, core_len
            g, kept = gather_gram(V, offs, c)
            sc = topk_scores(g, kept, k)
            acc[name].append(subspace_cc(sc, truth, k))

    if verbose:
        print(f"\n=== fst={fst}  block_len={block_len}  core_len={core_len}  "
              f"n_cores={n_cores}  budget={budget} variants "
              f"({100*budget/(n_blocks*block_len):.1f}% of file)  seeds={seeds} ===")
        print(f"{'design':12s} {'mean_cc':>10s} {'std_cc':>10s} {'min_cc':>10s} "
              f"{'1-mean':>10s}")
        base = np.mean(acc["systematic"])
        for name in designs:
            a = np.array(acc[name])
            tag = ""
            if name != "systematic":
                d = a.mean() - base
                tag = f"  (Δ vs sys {d:+.4f})"
            print(f"{name:12s} {a.mean():10.5f} {a.std():10.5f} {a.min():10.5f} "
                  f"{1-a.mean():10.5f}{tag}")
    return acc


REGIMES = [
    # (fst, block_len, core_len, n_cores, n_blocks)
    (0.02, 16, 8, 160, 2500),    # moderate structure, moderate LD
    (0.006, 24, 12, 160, 2000),  # subtle structure, strong LD (should stress the design)
    (0.006, 24, 6, 160, 2000),   # subtle structure, strong LD, smaller cores
    (0.02, 1, 8, 160, 2500),     # NO within-block LD control (block_len=1)
]

if __name__ == "__main__":
    import sys, gc
    # Optional single-regime mode (run each in its own process to release memory between).
    sel = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else range(len(REGIMES))
    for i in sel:
        fst, block_len, core_len, n_cores, n_blocks = REGIMES[i]
        run(seeds=40, fst=fst, block_len=block_len, core_len=core_len, n_cores=n_cores,
            n_blocks=n_blocks)
        gc.collect()
