"""Empirical test of ONE lever: incremental / online PCA that never forms the n x n Gram,
vs the streaming-Gram + randomized-eig baseline, at n=2504.

We build a Patterson-standardized design Z (n_variants x n_samples) with planted low-rank
population structure, then compare, on the SAME streamed variant-chunks:

  * TRUTH        : full-data Gram G = Z^T Z, exact top-k eigh.
  * BASELINE     : stream variant-chunks, accumulate G += Zc^T Zc (sgemm), randomized top-k eig.
  * FrequentDir  : maintain an (ell x n) sketch (Liberty), never form G; top-k right sing vecs.
  * Oja          : stochastic power iteration, update an (n x k) basis per chunk.

Metric: min canonical correlation between each method's top-k subspace and TRUTH's, plus
wall-time. Verdict = does any incremental method beat BASELINE wall-time while holding
min-canoncorr >= 0.99?
"""
from __future__ import annotations

import os
_BLAS = os.environ.get("EXP_BLAS_THREADS", "")
if _BLAS:
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _BLAS

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.generate import DatasetSpec, simulate_genotypes


# --------------------------------------------------------------------------- data
def build_Z(n_samples=2504, n_variants=24000, n_pops=5, fst=0.01, seed=0):
    """Patterson-standardize simulated genotypes -> Z (n_variants x n_samples) float32.
    fst small => subtle structure needing many markers (the realistic regime).
    Memory-frugal: free intermediates as we go (this box has a ~1GB cap)."""
    spec = DatasetSpec(name="exp", category="subtle", n_samples=n_samples,
                       n_variants=n_variants, n_pops=n_pops, fst=fst, seed=seed)
    G, sample_pop, _, _, _ = simulate_genotypes(spec)          # (n_variants x n_samples) int8
    p = G.mean(axis=1, dtype=np.float64) / 2.0
    scale = np.sqrt(2.0 * p * (1.0 - p))
    keep = (scale > 1e-7) & (np.minimum(p, 1 - p) > 1e-7)
    Z = np.empty((int(keep.sum()), G.shape[1]), dtype=np.float32)
    Z[:] = G[keep]
    del G
    Z -= (2.0 * p[keep]).astype(np.float32)[:, None]
    Z /= scale[keep].astype(np.float32)[:, None]
    return Z, sample_pop


def chunks(M, cs):
    i = 0
    while i < M:
        yield i, min(i + cs, M)
        i += cs


# --------------------------------------------------------------------------- methods
def truth_topk(Z, k):
    """Exact top-k eigenpairs of G=Z^T Z, matrix-free (never forms n x n float64 G):
    Lanczos (eigsh) on G x = Z^T (Z x). Memory-light and exact for structured spectra."""
    n = Z.shape[1]
    def mv(x):
        xf = np.asarray(x, dtype=np.float32)
        return (Z.T @ (Z @ xf)).astype(np.float64)
    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    ev, V = eigsh(op, k=k, which="LA")
    order = np.argsort(ev)[::-1]
    return V[:, order], ev[order]


def baseline_stream_gram(Z, k, chunk, oversample=12, power_iters=2, seed=0):
    n = Z.shape[1]
    G = np.zeros((n, n), dtype=np.float32)
    for a, b in chunks(Z.shape[0], chunk):
        Zc = Z[a:b]
        G += Zc.T @ Zc
    # randomized top-k eig on G
    ell = min(n, k + oversample)
    rng = np.random.default_rng(seed)
    Om = rng.standard_normal((n, ell)).astype(np.float32)
    Y = G @ Om
    Q, _ = np.linalg.qr(Y)
    for _ in range(power_iters):
        Q, _ = np.linalg.qr(G @ Q)
    B = Q.T @ (G @ Q); B = 0.5 * (B + B.T)
    ev, Vv = np.linalg.eigh(B.astype(np.float64))
    V = Q @ Vv[:, ::-1][:, :k]
    return V.astype(np.float64), ev[::-1][:k]


def frequent_directions(Z, k, chunk, ell=None, seed=0):
    """Liberty's Frequent Directions on rows of Z (each row = a variant, length n).
    Maintain a (2*ell x n) buffer; when full, SVD and shrink. Final sketch top-k right
    singular vectors approximate the top-k eigenvectors of Z^T Z."""
    n = Z.shape[1]
    if ell is None:
        ell = 2 * k + 8
    ell = min(ell, n - 1)
    B = np.zeros((2 * ell, n), dtype=np.float32)
    row = 0
    def shrink(B):
        # SVD of (2ell x n); keep ell, subtract the (ell)th sing val^2
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        s2 = s ** 2
        cut = s2[ell - 1] if s2.size >= ell else 0.0
        s_new = np.sqrt(np.maximum(s2 - cut, 0.0))
        Bn = np.zeros_like(B)
        m = min(ell, Vt.shape[0])
        Bn[:m] = (s_new[:m, None] * Vt[:m])
        return Bn, m
    for a, b in chunks(Z.shape[0], chunk):
        Zc = Z[a:b]
        for r in range(Zc.shape[0]):
            if row >= 2 * ell:
                B, row = shrink(B)
            B[row] = Zc[r]
            row += 1
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt[:k].T
    return V.astype(np.float64), (s[:k] ** 2)


def frequent_directions_blocked(Z, k, chunk, ell=None, seed=0):
    """Block Frequent Directions: append a whole chunk then SVD-shrink. Far fewer SVD calls
    than row-at-a-time, so a fairer 'incremental' contender speed-wise."""
    n = Z.shape[1]
    if ell is None:
        ell = 2 * k + 8
    ell = min(ell, n - 1)
    B = np.zeros((ell, n), dtype=np.float32)
    have = 0
    for a, b in chunks(Z.shape[0], chunk):
        Zc = Z[a:b]
        stack = np.vstack([B[:have], Zc]) if have else Zc
        U, s, Vt = np.linalg.svd(stack, full_matrices=False)
        m = min(ell, s.size)
        s2 = s[:m] ** 2
        cut = s2[-1] if m == ell else 0.0
        s_new = np.sqrt(np.maximum(s2 - cut, 0.0))
        B = np.zeros((ell, n), dtype=np.float32)
        B[:m] = (s_new[:, None] * Vt[:m])
        have = m
    U, s, Vt = np.linalg.svd(B[:have], full_matrices=False)
    V = Vt[:k].T
    return V.astype(np.float64), (s[:k] ** 2)


def oja(Z, k, chunk, lr=None, seed=0, passes=1):
    """Oja's stochastic power iteration on an (n x k) basis. Each chunk: Y = W^T Zc^T (k x b),
    update W += lr * Zc^T @ (Zc @ W) then re-orthonormalize. Uses a decaying step."""
    n = Z.shape[1]
    rng = np.random.default_rng(seed)
    W, _ = np.linalg.qr(rng.standard_normal((n, k)).astype(np.float32))
    t = 0
    for _p in range(passes):
        for a, b in chunks(Z.shape[0], chunk):
            Zc = Z[a:b]
            step = lr if lr is not None else 1.0 / (Zc.shape[0])
            # gradient of Rayleigh quotient ~ Zc^T (Zc W)
            grad = Zc.T @ (Zc @ W)
            W = W + step * grad
            W, _ = np.linalg.qr(W)
            t += 1
    return W.astype(np.float64), None


# --------------------------------------------------------------------------- metric
def min_canoncorr(A, B):
    """Min canonical correlation between column spaces of A, B (both n x k, ~orthonormal)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.clip(s, -1, 1).min())


def timeit(fn, *a, reps=3, **kw):
    best = np.inf; out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def main():
    import gc
    k = 4
    chunk = 2000
    print(f"BLAS threads env: {_BLAS or '(default/all)'}   k={k} chunk={chunk}")

    # NB: this box has a ~0.5GB working-set cap and macOS does not return freed pages to the
    # OS, so we run the small head-to-head FIRST (low RSS, all methods live incl FD's SVDs) and
    # the 24000-variant full-regime baseline LAST (baseline-only; FD there would OOM).

    # Head-to-head: all methods on the same streamed chunks; subspace vs exact truth.
    for (n_samp, n_var, fst) in [(2504, 8000, 0.01), (2504, 8000, 0.05)]:
        Z, _ = build_Z(n_samples=n_samp, n_variants=n_var, fst=fst)
        M, n = Z.shape
        print(f"\n=== HEAD-TO-HEAD  n_samples={n}  M={M}  fst={fst}  Gram={n*n*4/1e6:.0f}MB ===")
        Vt_true, ev_true = truth_topk(Z, k)
        print(f"  truth top-4 eigenvalues: {np.round(ev_true,1)}  (bulk edge ~ M+n = {M+n})")

        t_base, (Vb, _) = timeit(baseline_stream_gram, Z, k, chunk, reps=3)
        cc_base = min_canoncorr(Vb, Vt_true)
        print(f"  BASELINE stream-Gram+randeig : {t_base*1000:8.1f} ms   cc={cc_base:.4f}   (x1.00)")

        t_fdb, (Vf, _) = timeit(frequent_directions_blocked, Z, k, chunk, reps=1)
        cc_fdb = min_canoncorr(Vf, Vt_true)
        print(f"  FrequentDirections (blocked) : {t_fdb*1000:8.1f} ms   cc={cc_fdb:.4f}   speedup x{t_base/t_fdb:.3f}")

        t_oja, (Vo, _) = timeit(oja, Z, k, chunk, reps=3)
        cc_oja = min_canoncorr(Vo, Vt_true)
        print(f"  Oja (1 pass)                 : {t_oja*1000:8.1f} ms   cc={cc_oja:.4f}   speedup x{t_base/t_oja:.3f}")

        t_oja5, (Vo5, _) = timeit(oja, Z, k, chunk, passes=8, reps=2)
        cc_oja5 = min_canoncorr(Vo5, Vt_true)
        print(f"  Oja (8 passes)               : {t_oja5*1000:8.1f} ms   cc={cc_oja5:.4f}   speedup x{t_base/t_oja5:.3f}")
        del Z, Vt_true, Vb, Vf, Vo, Vo5; gc.collect()

    # Scaling probe: does the picture flip at large n? Baseline Gram is O(M n^2) time (and
    # O(n^2) memory); Oja's update is O(M n k) time (O(n k) memory). So Oja gets relatively
    # cheaper as n grows -- we show baseline time exploding quadratically while Oja stays linear.
    print("\n=== SCALING PROBE: baseline (O(n^2)) vs Oja (O(n)) as n grows (4000 variants) ===")
    for n_samp in [2504, 5000, 7500, 10000]:
        Z, _ = build_Z(n_samples=n_samp, n_variants=4000, fst=0.02)
        M, n = Z.shape
        Vt_true, _ = truth_topk(Z, k)
        t_base, (Vb, _) = timeit(baseline_stream_gram, Z, k, chunk, reps=2)
        t_oja, (Vo, _) = timeit(oja, Z, k, chunk, passes=3, reps=2)
        cc_b = min_canoncorr(Vb, Vt_true); cc_o = min_canoncorr(Vo, Vt_true)
        print(f"  n={n:6d} Gram={n*n*4/1e6:6.0f}MB  base={t_base*1000:8.1f}ms(cc{cc_b:.3f})  "
              f"Oja3={t_oja*1000:7.1f}ms(cc{cc_o:.3f})  base/Oja=x{t_base/t_oja:.2f}")
        del Z, Vt_true, Vb, Vo; gc.collect()

    # Full-regime baseline GEMM cost (prompt's 24000-variant point), baseline only.
    Z, _ = build_Z(2504, 24000, fst=0.01)
    Vt, _ = truth_topk(Z, k)
    t_base24, (Vb, _) = timeit(baseline_stream_gram, Z, k, chunk, reps=3)
    def gram_only(Z):
        n = Z.shape[1]; G = np.zeros((n, n), np.float32)
        for a, b in chunks(Z.shape[0], chunk):
            Zc = Z[a:b]; G += Zc.T @ Zc
        return G
    t_g, G = timeit(gram_only, Z, reps=3)
    print(f"\n[full regime] n=2504 M=24000 : BASELINE {t_base24*1000:.1f} ms  "
          f"cc={min_canoncorr(Vb, Vt):.4f}   (Gram GEMM {t_g*1000:.1f} ms = "
          f"{100*t_g/t_base24:.0f}%, randomized-eig {(t_base24-t_g)*1000:.1f} ms)")
    del Z, G, Vb, Vt; gc.collect()


if __name__ == "__main__":
    main()
