# Reward-hacking protections

Accuracy alone is gameable: a submission can post a high subspace score without ever being a
genuine, fast, HWE-normalised PCA. The bounded reward therefore combines quality-dominant
accuracy, a demanding systems term, and continuous object-identity factors. The design rests on one
invariant:

> **A genuine PCA is never misidentified as a shortcut.** Every identity gate is
> reference-relative, severity-scaled, and dead-zoned — a correct Patterson fit reads severity 0
> and factor 1. `fn_resistance/independent_python` is the standing proof: its accuracy and method
> factors must remain ≥ 0.9 even though its unoptimised full-scan runtime receives only partial
> systems credit.

Each known hack and the guard that catches it:

| Hack | Why accuracy alone misses it | Guard | Result |
|---|---|---|---|
| **plain covariance PCA** (skips `√(2p(1-p))`) | on strong structure the covariance and Patterson subspaces nearly coincide, so accuracy is high | `hwe_norm` gate: a probe with orthogonal common-variant (raw-cov-dominant) and rare-variant (Patterson-dominant) axes — genuine puts the rare axis on PC1, the impostor puts the common axis there | heavily discounted, floor 0.15 |
| **undercoverage** (fit on ~200 SNPs) | 200 markers resolve *strong* continental structure fine, so accuracy on easy data is high | `coverage` gate: a subtle-structure probe that needs thousands of markers; a hard-capped sampler cannot resolve it | heavily discounted, floor 0.15 |
| **representation-sensitive parser** | a brittle fit can look correct on one convenient GT layout | paired-equivalence probe: independently permuted samples/records plus allele flips, case, phase, and valid FORMAT variation | continuous penalty |
| **delegate to a library** (sklearn PCA, scikit-allel, pysam, plink…) | the numbers can be perfect | `library_scan`: an AST import allowlist — only stdlib + numpy/scipy + the submission's own local modules pass; every genomics-I/O and whole-PCA package is rejected. Shelling out to a genomics *binary* (plink/bcftools/…) is not statically scanned, but the image ships none, so there is nothing to delegate to; numpy/scipy primitives stay allowed | 0.00 |
| **random / constant / first-k-variants output** | — | subspace accuracy floor (naive = 0) + validity | 0.00 |

## Why the identity gates are hard to fool

- **Accuracy is the ceiling.** Runtime can strongly distinguish accurate implementations, but
  multiplying by scientific agreement means a fast nearby object cannot outrun its accuracy.
- **Anti-fingerprinting.** Probe mathematics is fixed by a private release key while every run
  receives fresh aliases, sample order, and allele polarity. Carrier selection reuses an active
  scored fold's complete dimensions, requested rank, and reviewed serialization family. Plain
  carriers remain plain GT; representation carriers use the same mixed-width FORMAT, INFO-width,
  chromosome-count, ID, and newline distributions as the active variable-width fold. This removes
  deterministic metadata and byte-layout classifiers; it does not rely on an impossible claim
  that arbitrary scientific content can never be classified.
- **Anchored to per-probe references, not a fixed threshold.** The `hwe_norm` gate measures the
  fraction of a fit's score-variance on the rare/ancestry axis and compares it to two references
  recomputed **on the same probe** — a genuine Patterson fit and a raw-covariance fit — scoring
  the submission by where it falls between them. Both a Patterson and a covariance PCA span the
  same {rare, common} subspace; only the *position* between the anchors separates them, and it
  does so on every draw. An earlier fixed-threshold rule ("does the common axis lead the rare
  axis by a set margin") was **not** seed-robust: raw-covariance's absolute rare-fraction wanders
  ~0.35–0.58 across draws (there are far more rare than common variants, so their combined raw
  variance sometimes rivals the common axis), landing inside the dead-zone on ~1/4 of seeds and
  leaking the whole reward. Anchoring to the freshly-computed Patterson/raw references — whose
  separation is a stable ~0.20–0.24 — crushes raw-covariance on every seed while a genuine fit
  (which sits at the Patterson anchor) reads severity 0.
- **Severity is seed-robust.** HWE and coverage severity use multiple private release draws and a
  median. Representation equivalence compares each arm with the exact common reference as well as
  with the other arm, so two equally broken parses cannot cancel into a passing gap.

## Standing checks

- `scripts/validate.py` runs `fn_resistance/` (accuracy and method factors must remain ≥ 0.9) and every
  `cheats/*` program through the **exact production grading path**. Wrong-object and undercoverage
  shortcuts must score ≤ 0.15 in every category. The format-fragile fixture is a genuine Patterson
  PCA on clean GT, so its legitimate clean-category credit is not treated as leakage; instead it
  must score ≤ 0.40 overall **and** ≤ 0.05 in each of `messy` and `variable_width`. Missing either
  parser category fails validation closed. A shortcut that exceeded its reviewed, failure-specific
  ceiling here would do so in the real benchmark — precisely the false positive the guards exist
  to catch.
- `scripts/probe_diag.py` prints the raw alignment/agreement numbers a submission produces on
  each probe, so gate thresholds are tuned against real fits rather than guessed.
