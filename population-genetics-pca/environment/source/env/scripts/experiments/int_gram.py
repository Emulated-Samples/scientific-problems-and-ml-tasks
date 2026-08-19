"""Empirical test: is an integer / low-precision Gram faster than float32 sgemm?

Shape: V=24000 variants x N=2504 samples. Gram G = Z^T Z is (2504 x 2504).
Reference (reference/fast_pca.py) accumulates a float32 Gram  Z^T Z  via a symmetric
rank-k update (ssyrk, half a GEMM's FLOPs). This experiment times a plain float32 GEMM as the
float baseline, which upper-bounds that cost -- the int/low-precision question is unchanged.

Run one mode per subprocess so a SIGKILL (OOM) in one path does not lose the others:
    uv run python -u scripts/experiments/int_gram.py <mode>
    modes: f32 f16 int16 int32 int8 weighted gt

Pure numpy only. gt writes float64 ground-truth top-k eigvals/vecs to a cache the
correctness modes load.
"""
import sys, time, os
import numpy as np

V, N, K = 24000, 2504, 10
CACHE = "/private/tmp/claude-501/-Users-user/dc0c892a-97aa-4c40-8090-ad12b6fe71b3/scratchpad/gt_cache.npz"

def make_X():
    rng = np.random.default_rng(0)          # fixed seed -> identical X every subprocess
    p = rng.uniform(0.05, 0.5, size=V).astype(np.float64)
    X = np.empty((V, N), dtype=np.int8)
    # generate row-blocks to bound peak memory
    for i in range(0, V, 2000):
        j = min(i + 2000, V)
        X[i:j] = rng.binomial(2, p[i:j, None], size=(j - i, N)).astype(np.int8)
    s = np.sqrt(2.0 * p * (1.0 - p))
    w = 1.0 / s**2
    return X, p, s, w

def timeit(fn, reps=5, warm=True):
    if warm:
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts), np.median(ts)

def topk(G, ev64, U64):
    ev = np.linalg.eigvalsh(G)[::-1][:K]
    val_rel = float(np.max(np.abs(ev - ev64) / np.abs(ev64)))
    _, Vv = np.linalg.eigh(G)
    U = Vv[:, ::-1][:, :K]
    sv = np.linalg.svd(U64.T @ U, compute_uv=False)
    return val_rel, float(np.max(np.abs(1.0 - sv)))

def report(name, mn, md, ve, se, ref=223.46e-3):
    spd = ref / mn if mn and mn > 0 else float('nan')
    ve_s = f"{ve:.2e}" if ve == ve else "   n/a"
    se_s = f"{se:.2e}" if se == se else "   n/a"
    print(f"{name:40s} min={mn*1e3:9.2f}ms med={md*1e3:9.2f}ms val_rel={ve_s} sub={se_s} ({spd:.2f}x vs f32)", flush=True)

mode = sys.argv[1]
X, p, s, w = make_X()

if mode == "gt":
    Z64 = (X.astype(np.float64) - 2.0 * p[:, None]) / s[:, None]
    G64 = Z64.T @ Z64
    ev = np.linalg.eigvalsh(G64)[::-1][:K]
    _, Vv = np.linalg.eigh(G64)
    U = Vv[:, ::-1][:, :K]
    np.savez(CACHE, ev64=ev, U64=U)
    print("ground truth cached. top eigval:", ev[0], flush=True)
    sys.exit(0)

d = np.load(CACHE)
ev64, U64 = d["ev64"], d["U64"]

if mode == "f32":
    Z = ((X.astype(np.float32) - np.float32(2.0) * p[:, None].astype(np.float32))
         / s[:, None].astype(np.float32))
    f = lambda: Z.T @ Z
    mn, md = timeit(f)
    ve, se = topk(f().astype(np.float64), ev64, U64)
    report("float32 sgemm (reference)", mn, md, ve, se)

elif mode == "f16":
    Z = ((X.astype(np.float32) - np.float32(2.0) * p[:, None].astype(np.float32))
         / s[:, None].astype(np.float32)).astype(np.float16)
    f = lambda: Z.T @ Z
    mn, md = timeit(f, reps=1, warm=False)
    ve, se = topk(f().astype(np.float64), ev64, U64)
    report("float16 Z^T Z", mn, md, ve, se)

elif mode == "int16":
    Xi = X.astype(np.int16)
    f = lambda: Xi.T @ Xi
    mn, md = timeit(f, reps=1, warm=False)
    report("int16 X^T X (raw)", mn, md, float('nan'), float('nan'))

elif mode == "int32":
    Xi = X.astype(np.int32)
    f = lambda: Xi.T @ Xi
    mn, md = timeit(f, reps=1, warm=False)
    report("int32 X^T X (raw)", mn, md, float('nan'), float('nan'))

elif mode == "int8":
    f = lambda: np.dot(X.T, X)
    print("int8 np.dot result dtype:", np.dot(X.T[:2, :2], X[:2, :2]).dtype, flush=True)
    mn, md = timeit(f, reps=1, warm=False)
    report("int8 np.dot X^T X", mn, md, float('nan'), float('nan'))

elif mode == "weighted":
    # THE LEVER: per-variant weight 1/s_j^2 sits BETWEEN the two X's (X^T diag(w) X).
    # It cannot factor out of an integer GEMM. Folding sqrt(w_j) into X makes X FLOAT.
    sw = np.sqrt(w).astype(np.float32)
    def build_and_gemm():
        Xw = X.astype(np.float32) * sw[:, None]     # per-variant float scale -> float matrix
        return Xw.T @ Xw
    mn, md = timeit(build_and_gemm)
    Xw = X.astype(np.float32) * sw[:, None]
    XtWX = (Xw.T @ Xw).astype(np.float64)
    c = (w * 2.0 * p)
    a = X.astype(np.float64).T @ c                  # (N,)
    const = float(np.sum(w * 4.0 * p * p))
    G = XtWX - a[:, None] - a[None, :] + const
    ve, se = topk(G, ev64, U64)
    report("weighted-f32 GEMM + rank1 corr", mn, md, ve, se)
