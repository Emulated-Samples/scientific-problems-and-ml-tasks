# Algorithmic lever experiments

Each script here isolates ONE proposed optimization for the fast reference PCA, benchmarks it
against the streaming-Gram baseline on representative synthetic genotype data, and reports a
numeric verdict. They exist so the reference's design choices are backed by measurement, not
assertion. None of them modifies `reference/` or `grader/`.

The reference regime under test: n≈2504 samples, ~200k sampled variants, Patterson-standardised
`z=(x-2p)/√(2p(1-p))`, sample Gram `G=Z^T Z` (n×n) accumulated by a float32 symmetric rank-k
update (`ssyrk`), top-k via an exact range eigensolver (LAPACK `dsyevr`). Runs on real 1000G
chr22 (11 GB), accuracy ~0.978 vs a full-scan (all variants, no filtering) anchor.

## Verdicts

| Lever | Script | Verdict | Why |
|---|---|---|---|
| **Leverage / Horvitz-Thompson importance sampling** | `leverage.py` | ❌ Reject | Patterson standardisation forces E‖z_j‖²=n_samples for *every* variant → leverage is flat (CV 0.037, max/mean 1.32). Importance sampling needs a heavy tail; the HWE denominator is exactly what removes it. Identical accuracy-vs-budget to uniform; and the leverage pilot must decode 100% of variants, defeating the point of sampling. |
| **Integer / low-precision Gram** | `int_gram.py` | ❌ Reject | numpy has BLAS GEMM only for float32/float64. int8/int16/int32 matmul falls to a scalar loop **400–900× slower** and OOMs at real size; int8 also overflows to garbage; float16 has no kernel. The honest "weighted-float + rank-1 centering" lever is correct but **1.5× slower** because the per-variant weight `1/s_j²` sits on the contraction axis and forces a float matrix. The reference's float32 `ssyrk` Gram is already optimal. |
| **Incremental / online PCA** (Oja, Frequent-Directions) | `incremental.py` | ❌ Reject at n=2504 | The 25 MB Gram's `Z^T Z` is a cache-friendly float32 rank-k update (`ssyrk`, ~85% of a 292 ms run) that BLAS does faster than any BLAS-unfriendly online update. Frequent-Directions is accurate but **165× slower**; Oja is fast but plateaus at cc≈0.96 on subtle structure (fails the 0.99 bar). A crossover exists at large n (Oja 24× faster by n=10000) — but that regime is handled by the reference's **residual-converged matrix-free Lanczos** engine switch, which triggers when n > n_variants. |
| **Sobol / low-discrepancy offsets** | `sobol.py` | ❌ Reject (but found the real lever) | Random-phase *systematic* offsets (a regular lattice + one random phase) retain periodic aliasing. **The real accuracy lever is finer independently-jittered stratification** — more, smaller cores at the same byte budget: +0.6% (moderate LD) to **+1.8% (subtle+strong LD)** subspace accuracy and ~40% lower variance, because smaller cores grab fewer redundant *within-LD-block* neighbours (each byte buys more independent structure). The live sampler combines independently jittered strata with a uniform circular rotation for exact equal marginal byte inclusion. |

## The one integration

Only Sobol's *secondary* finding — **finer cores** — survived as an accuracy-per-byte win, and
it is exactly the "exploit VCF sortedness to decorrelate the LD-redundant sample" idea. It is
benchmark-faithful (still uniform over the genome, unbiased for the full Gram), just a
lower-variance estimator at the same budget.

## The honest conclusion

For the benchmark's regime (n≈2504, a few gigabytes, k≈10) the reference is **at or near the
pure-numpy optimum**: streaming float32 Gram (`ssyrk`) + exact top-k eig (LAPACK `dsyevr`), uniform systematic
sampling, matrix-free only when n exceeds the variant count. The "clever" algorithmic levers
either do nothing (HWE flattens leverage; systematic is already optimal), collapse to the same
float GEMM (int8), or only help a large-n regime the engine switch already covers. That is a
useful thing to have *proven* rather than assumed.

> **Follow-up on finer cores (measured on the benchmark's own data):** the +1.8% held only on the Sobol experiment's *explicit-LD-block* synthetic. On the benchmark's Balding-Nichols data (variants independent, no within-block LD) finer cores gave **no accuracy gain** (big_subtle 0.9873 flat at 1024x256, 2048x128, 4096x64). Real chromosomes have LD, but the current sampler already reaches 0.996 on chr22. So finer cores was **not integrated** -- it is a real lever only where strong LD makes contiguous cores redundant, which the benchmark's synthetic data does not exercise.
