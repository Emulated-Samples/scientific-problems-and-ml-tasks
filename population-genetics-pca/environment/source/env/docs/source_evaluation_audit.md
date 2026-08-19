# External PCA evaluation audit

This note records which evaluation ideas were reviewed from two SauersML repositories and which
ones were deliberately incorporated into pcabench.  It is maintainer context, not solver guidance:
none of these source repositories, APIs, test names, or architectural suggestions are exposed in
the task prompt.

## Sources reviewed

The review on 2026-07-13 pinned the repositories at:

* [`SauersML/efficient_pca`](https://github.com/SauersML/efficient_pca), commit
  `e1d6d3cbed9f4638ce32f8cee83b0fd206f14d23`;
* [`SauersML/genomic_pca`](https://github.com/SauersML/genomic_pca), commit
  `1d5b910a8d54181664fd22dd1499fb24e31cf342`.

The files inspected included the Rust CI configurations; `efficient_pca`'s `pca_tests.rs`,
`eigensnp_tests.rs`, diagnostics tests, Python bootstrap, and Criterion benchmark/analysis
programs; and `genomic_pca`'s metric, disk, local-PCA, sweep, Hail, subset, and data scripts. The
purpose was to find additional failure modes and independent validation ideas, not to transplant
either library's implementation or public API.

## Incorporated ideas

### Shape-regime coverage

`efficient_pca/benches/benchmarks.rs` explicitly exercises square, feature-heavy, and sample-heavy
matrices.  That is a useful systems distinction for this benchmark: an implementation that always
materializes an `n_samples x n_samples` matrix can be strong in the usual genomic shape yet fail
badly when samples dominate informative variants.

Pcabench therefore includes a private sample-heavy scoring regime, alongside its conventional
variant-heavy and dense-compute regimes.  It is generated as biologically structured Patterson-PCA
data and receives smooth resource/quality partial credit; it is not a verbatim generic matrix
benchmark.

The upstream `Wide-k10`/`Wide-k50`/`Wide-k200` sweep also motivated varying the requested rank
rather than treating ten components as part of the API. Pcabench's private ranks span 3 through 64,
including distinct structured high-rank and very-high-rank arms. A square crossover arm complements
the feature-heavy and sample-heavy cases so a brittle regime switch is measurable without requiring
one particular decomposition. A separate spectral-selection arm plants additional identifiable
axes below a reviewed rank boundary, with descending signal and a finite eigengap at the requested
`k`; this distinguishes the leading subspace from an arbitrary set of population-structured axes
without requiring signed components or one eigensolver. The source's sparse and poorly conditioned
cases are covered independently by rare-structure, duplicate/near-duplicate, near-fixed,
monomorphic, and ill-conditioned Patterson fixtures. Those fixtures preserve genomic dosage
semantics instead of copying generic dense or iid sparse matrices.

### Reference-only PCA invariants

The upstream `efficient_pca` tests check dimensions, finiteness, component orthogonality,
eigenvalues, and agreement with independent decompositions.  Pcabench adapts the implementation-
neutral portion of that idea in `tests/test_reference_invariants.py`:

* score arrays and spectra have the expected rank and shape;
* reference scores are finite and centered;
* score columns are orthogonal and their squared norms equal the reported variances;
* the reference span agrees with an independent SVD in both shape regimes;
* a controlled spectrum retains identifiable tail structure while the returned score covariance
  equals the leading `k` eigenvalues across a clear boundary gap;
* the complete VCF decoder-to-PCA path preserves the same algebra.

These are tests of the sealed reference only.  Submission grading continues to compare subspaces,
so arbitrary PC signs and rotations inside unresolved eigenspaces remain valid.

### Independent scientific cross-checks

`genomic_pca/scripts/hail_pca.py` is a useful model for an occasional maintainer-only Hail
cross-check.  Such a check can compare Patterson/HWE-normalized score subspaces on generated data
and on the reviewed chromosome-22 slice.  Hail is intentionally not a runtime evaluator dependency:
the check belongs in a scheduled validation job, where a dependency outage cannot invalidate an
agent rollout.

The population-label diagnostics in `genomic_pca/tests/metrics.py` also reinforced the value of
checking real population hierarchy.  Pcabench's real chromosome-22 arm already measures hierarchy
through rotation-invariant structure diagnostics rather than a particular clustering package or
classifier.

The `disk.py` and local/sweep scripts are useful evidence that sample count, marker count, memory,
and population composition must be varied independently. Pcabench uses those as calibration axes,
not as solver-facing API requirements: the private suite includes variant-heavy I/O, sample-heavy
linear algebra, subtle and hierarchical structure, and the checksum-pinned observed chromosome-22
cohort under one output contract.

### Maintainer performance analysis

The benchmark tables and `benches/analyze_benchmarks.py` motivate maintainer-side Pareto sweeps over
wall time, memory, shape, and accuracy.  Those sweeps are calibration tools, not agent-visible
threshold recipes.  A release should be calibrated against multiple implementation tiers and retain
low repeat noise while separating their performance.

The grader therefore reports sampled aggregate proportional RSS and writable temporary-storage peaks
for each scored primary invocation. The watchdog polls, so these are sampled rather than continuous
maxima; probe invocation peaks are not included. Unexecuted deadline rows use `null`, not a fictitious
zero. These remain diagnostics, not reward terms by design. A 200 ms polling maximum is gameable by
short-lived allocations, probe peaks are excluded, and the fastest reference is not proven to be the
lowest-memory reference on every shape; turning it into reward would add noise and double-charge
legitimate memory-for-speed tradeoffs. Memory discipline is instead measured by the hard aggregate
cap and the independent 24k/50k/64k sample-count ladder, which provides stable graded credit for
f64-sample-Gram, f32-sample-Gram, and marker-space strategies.

## Deliberately excluded

The following source behaviors do not define this task and must not become submission requirements:

* exact signed principal components or exact column-by-column equality;
* loadings, `fit`/`transform`, persistence, PLS, refinement, block APIs, or any Rust-specific type;
* a specific eigensolver, matrix orientation, backend, thread topology, or randomized algorithm;
* PLINK-style MAF, HWE, LD, or missingness filtering, because those change the population of
  eligible markers defined by pcabench;
* performance assertions copied from another machine or library;
* source test fixtures as recognizable hidden-evaluation fixtures.

In particular, loadings and signed axes are valid library API concerns but would be false negatives
here: the solver contract asks for a sample score subspace, for which sign and tied-axis rotation are
not identifiable.

## Licensing and data provenance

`efficient_pca` has a root MIT license.  Pcabench nevertheless reimplemented only general testing
ideas; no source or test code was copied.

`genomic_pca` declares `license = "MIT"` in `Cargo.toml`, but the audited commit has no root license
file and its bundled population tables and chromosome-22 archives do not carry sufficiently clear
per-file redistribution provenance.  Pcabench therefore copied none of its code or data.  The real
chromosome-22 evaluation continues to be derived through pcabench's own reviewed pipeline from its
documented upstream 1000 Genomes source, with checksums and cohort-selection provenance maintained
in this repository.

This distinction matters operationally: useful scientific test ideas can be independently
reimplemented, while third-party data artifacts should not be redistributed on an inferred license.
