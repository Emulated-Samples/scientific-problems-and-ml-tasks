"""Empirical test: leverage-score / Horvitz-Thompson importance sampling of variants
vs uniform sampling for Patterson PCA.

Question: does sampling variant j with prob pi_j ~ ||z_j||^2 and reweighting by 1/pi_j
(unbiased HT estimator of the full Gram) reach the same top-k subspace accuracy as uniform
sampling with FEWER variants? Account for the pilot pass needed to estimate leverage.

Pure numpy/scipy. Does NOT touch reference/ or grader/ (only imports read-only helpers).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.generate import DatasetSpec, simulate_genotypes
from grader.metrics.subspace import subspace_accuracy, structure_weights


def standardize_all(G: np.ndarray) -> np.ndarray:
    """Patterson-standardise the full (n_variants x n_samples) dosage matrix -> Z (float64).
    Drops monomorphic/degenerate variants exactly as the reference does."""
    X = G.astype(np.float64)
    n_samp = X.shape[1]
    p = X.sum(axis=1) / (2.0 * n_samp)
    maf = np.minimum(p, 1 - p)
    scale = np.sqrt(2.0 * p * (1.0 - p))
    keep = (maf > 1e-7) & (scale > 1e-7)
    Xk = X[keep]
    pk = (2.0 * p[keep])[:, None]
    sk = scale[keep][:, None]
    Z = (Xk - pk) / sk
    return Z.astype(np.float32)  # (n_kept_variants x n_samples), float32 to bound memory


def topk_subspace_from_gram(G: np.ndarray, k: int):
    """Return (scores n x k, evals k) from a sample-by-sample Gram via dense eigh."""
    G = 0.5 * (G + G.T)
    ev, V = np.linalg.eigh(G)
    idx = np.argsort(ev)[::-1][:k]
    evk = ev[idx]
    scores = V[:, idx] * np.sqrt(np.clip(evk, 0, None))[None, :]
    return scores, evk


def ortho(M: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(M)
    return Q


def min_canon_corr(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Min canonical correlation between the two top-k score subspaces (over samples)."""
    Qa = ortho(scores_a - scores_a.mean(0, keepdims=True))
    Qb = ortho(scores_b - scores_b.mean(0, keepdims=True))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.clip(s.min(), 0, 1))


def gram_uniform(Z: np.ndarray, M: int, rng) -> np.ndarray:
    """Unbiased full-Gram estimate from a uniform sample of M variants (with replacement).
    G_hat = (N/M) sum_{sampled} z z^T."""
    N = Z.shape[0]
    idx = rng.integers(0, N, size=M)
    Zs = Z[idx].astype(np.float64)
    return (N / M) * (Zs.T @ Zs)


def gram_leverage_ht(Z: np.ndarray, M: int, pi: np.ndarray, rng) -> np.ndarray:
    """Horvitz-Thompson estimate with with-replacement sampling prob pi_j.
    G_hat = (1/M) sum_draws (1/pi_j) z_j z_j^T  -> unbiased for sum_j z_j z_j^T."""
    N = Z.shape[0]
    idx = rng.choice(N, size=M, replace=True, p=pi)
    w = 1.0 / (M * pi[idx])          # HT weights
    Zs = Z[idx].astype(np.float64) * np.sqrt(w)[:, None]
    return Zs.T @ Zs


def main():
    t0 = time.time()
    spec = DatasetSpec("lev_subtle", "subtle", n_samples=800, n_variants=200000,
                       n_pops=4, fst=0.01, seed=42)
    print(f"Simulating {spec.n_variants} variants x {spec.n_samples} samples, "
          f"K={spec.n_pops} pops, Fst={spec.fst} ...", flush=True)
    G, sample_pop, *_ = simulate_genotypes(spec)
    Z = standardize_all(G)
    N, n = Z.shape
    print(f"  standardized: {N} kept variants x {n} samples  ({time.time()-t0:.1f}s)", flush=True)

    k = spec.n_pops - 1               # number of structured axes
    # Ground truth: full Gram over ALL variants.
    tG = time.time()
    G_full = np.zeros((n, n), dtype=np.float64)   # chunked float64 accumulation (memory-safe)
    for s in range(0, N, 20000):
        blk = Z[s:s + 20000].astype(np.float64)
        G_full += blk.T @ blk
    full_scores, full_evals = topk_subspace_from_gram(G_full, k)
    print(f"  full Gram top-{k} computed ({time.time()-tG:.1f}s)", flush=True)
    # Structure weights from the complete spectrum (for the grader metric).
    ev_all, _ = np.linalg.eigh(0.5 * (G_full + G_full.T))
    w_struct = structure_weights(np.sort(ev_all)[::-1], k)[:k]

    # Leverage scores = squared row norms of Z (this is the exact per-variant leverage
    # for the sum-of-outer-products; requires decoding+standardizing every variant).
    lev = np.einsum('ij,ij->i', Z, Z)
    pi = lev / lev.sum()
    print(f"  leverage stats: mean {lev.mean():.1f}  cv {lev.std()/lev.mean():.3f}  "
          f"max/mean {lev.max()/lev.mean():.2f}", flush=True)

    budgets = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    n_rep = 12
    print(f"\n{'M':>7} | {'uniform mcc':>22} | {'leverage mcc':>22} | "
          f"{'uniform acc':>14} | {'leverage acc':>14}")
    print("-" * 92)
    results = {}
    for M in budgets:
        u_mcc, l_mcc, u_acc, l_acc = [], [], [], []
        for r in range(n_rep):
            rng = np.random.default_rng(1000 + r)
            Gu = gram_uniform(Z, M, rng)
            Gl = gram_leverage_ht(Z, M, pi, rng)
            su, _ = topk_subspace_from_gram(Gu, k)
            sl, _ = topk_subspace_from_gram(Gl, k)
            u_mcc.append(min_canon_corr(su, full_scores))
            l_mcc.append(min_canon_corr(sl, full_scores))
            u_acc.append(subspace_accuracy(su, full_scores, w_struct)["accuracy"])
            l_acc.append(subspace_accuracy(sl, full_scores, w_struct)["accuracy"])
        results[M] = dict(u_mcc=np.mean(u_mcc), l_mcc=np.mean(l_mcc),
                          u_acc=np.mean(u_acc), l_acc=np.mean(l_acc))
        print(f"{M:>7} | {np.mean(u_mcc):.4f} +/- {np.std(u_mcc):.4f}      | "
              f"{np.mean(l_mcc):.4f} +/- {np.std(l_mcc):.4f}      | "
              f"{np.mean(u_acc):>14.4f} | {np.mean(l_acc):>14.4f}", flush=True)

    # How many uniform variants to match a given leverage budget's min-canon-corr?
    print("\n-- variance-reduction factor (min-canon-corr basis) --")
    for M in budgets:
        target = results[M]['l_mcc']
        # find smallest uniform budget whose mcc >= target
        need = None
        for Mu in budgets:
            if results[Mu]['u_mcc'] >= target - 1e-9:
                need = Mu
                break
        ratio = (need / M) if need else float('nan')
        print(f"  leverage M={M:>6} (mcc {target:.4f}) ~= uniform "
              f"{'>50000' if need is None else need:>7}  (factor {ratio if need else float('nan'):.2f}x)")

    print("\n-- pilot cost accounting --")
    print("  Leverage pi_j = ||z_j||^2 needs the standardized z_j for EVERY variant, i.e. a")
    print("  full decode+standardize pass over all N variants. In this VCF PCA the decode+")
    print("  standardize is the dominant per-variant cost; the Gram gemm (z z^T, the only")
    print("  thing sampling reduces) is O(n^2) on top. So an exact-leverage pilot reads 100%")
    print("  of variants -> defeats the entire point of sampling (which is to NOT read them).")
    print(f"  total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
