# Population-genetics PCA from scratch

Build a fast HWE-normalised PCA library over large, unindexed, plain-text VCFs, from scratch: NumPy and SciPy only, no genomics, dataframe, JIT, or ML packages.

Models get the mathematics right and lose on the systems work. The score is accuracy times speed, and across the shipped runs accuracy held between 0.93 and 1.0 while final scores ran 0.06 to 0.41 against the 0.89 reference bar — the spread is almost entirely the speed term. Two runs with accuracy 1.0 on every dataset finished at 0.15 and 0.24.

## The task

The agent ships a single executable, `pca <vcf_path> <k> <out_path>`, that must parse raw multi-sample VCF text the way real cohort files arrive — extra FORMAT fields, missing calls, indels, multiallelic and symbolic records, CRLF line endings, variable-width rows — select the polymorphic biallelic SNVs, compute a Patterson HWE-standardised PCA with missing calls imputed to the marker mean, and emit per-sample principal components in input order.

The prompt is the message an engineer would send: I need a fast PCA over large, plain-text, unindexed VCFs, here is the exact estimator, you have 24 hours. The estimator is pinned exactly, and shortcuts are ruled out up front: no selective marker filtering, and no approximation that biases marker inclusion by frequency, linkage, missingness, record width, or file position. The from-scratch constraint is enforced rather than requested: submissions are scanned at the AST level against an allowlist of the standard library, NumPy, and SciPy, which catches renamed imports, dynamic imports, and compiled artifacts.

Grading runs 22 datasets: 21 synthetic cohorts generated fresh at evaluation time under a Balding-Nichols model, spanning 96 samples x 1.5M variants to 64,000 samples x 2,000 variants, plus one cohort derived from real 1000 Genomes chromosome-22 data under opaque sample aliases so public-panel joins fail.

## Verifier design

Each dataset is scored as accuracy times speed: exactness must clear a mastery bar before speed earns anything, and speed is wall-clock-relative to reference implementations timed through the same execution path the submission runs in.

| What we check | How |
| --- | --- |
| The right mathematical object comes back | Recovered subspaces are compared per principal axis against ground truth, aggregated so that dropping any one structured axis collapses the score |
| Accuracy before speed | Speed earns nothing until accuracy clears a mastery bar, so wrong-and-fast is strictly worse than right-and-slow at every speed |
| Speed against an achievable ceiling | A full-scan reference and a fast reference are timed through the identical sandboxed execution path on every dataset; matching the fast reference scores full marks, and a crash's short wall time counts as zero, not fast |
| Sampling cannot cheat | A coverage probe defeats fitting on a small prefix of the file, and an equivalence probe re-serialises the same cohort (permuted, re-polarised, re-spelled) and requires the same recovered structure |
| It is the agent's own code | AST-level allowlist scan of every submitted file, plus rejection of compiled binaries and native extensions |
| The grader is out of reach | The grading tree is unreadable to the account the agent works in, and submissions execute under a separate isolated account |
| Slow is not an escape hatch | Grading runs on one global clock; datasets never started score zero without erasing the ones that finished |

## Trace walkthrough

Every shipped run gets the estimator essentially right. What separates the set is what each run does about the clock.

### A strong run

1. **Design for the file, not the matrix.** The winning run committed to streaming from its first message — never hold the genotype matrix in RAM — and kept two linear-algebra orientations, an exact sample-space Gram solve where that is cheap and a deterministic randomized range finder where the sample count grows, with the switch depending only on shape, never on which markers to read.
2. **Buy speed with parallel parsing.** It forked pure-Python workers across independent byte ranges of the VCF, exchanging NumPy scratch files through the temporary directory, then proved the forked and single-stream accumulations agree exactly — including byte boundaries that land in the middle of a record.
3. **Collect the verdict.** Under seven minutes and 13 tool calls end to end. Accuracy 0.93 or better on all 22 datasets, faster than the fast reference outright on the largest biobank cohort (14.3 seconds against 23.7), 228 seconds total against the reference's 127, final score 0.41 — the best in the set, with the remaining gap still wall clock.

### A failed run

1. **Perfect the mathematics.** The most instructive failure ran the deepest correctness campaign in the set: differential tests against a dense SVD matching to millionths of a degree, a forced partial-cache run agreeing with the full-cache result to 1e-7 in projector norm, even a rehearsal of writing the output in place inside a read-only directory. It landed accuracy 1.000 on all 22 graded datasets.
2. **Leave the parse on one lane.** It parsed each marker once into a dosage block on a single streaming pass, with a bounded dosage cache in temporary storage and a re-stream for whatever did not fit — and never parallelized any of it. Its one performance check before submitting was a 500-sample, 20,000-marker synthetic; the graded cohorts run to 1.5 million variants and 64,000 samples.
3. **Watch the clock take the score.** 471 seconds where the fast reference takes 135. On a continental-structure cohort it ran 57 seconds against the reference's 8, on the wide-IO cohort 36 against 7 — zero speed credit on both, with perfect accuracy worth about 0.1 apiece there. Final score 0.24.

The 0.24 run made 23 tool calls to the winner's 13, and every extra one was a correctness check. The winner banked the same correctness in half the moves and spent the difference making the parse parallel.

## Failure modes

These are the failure modes we saw across the evaluated runs.

| Failure mode | What goes wrong |
| --- | --- |
| Treating the per-invocation time limit as the target | Scoring is relative to the fast reference, not the cap; runs with accuracy 1.0 on all 22 datasets have finished at 0.15 and 0.24, the entire shortfall in wall clock |
| The parse left on one lane | The graded cohorts are parse-dominated, and submissions that kept a single-stream reader ran 2.5x to 36x the fast reference's wall clock; one spent its full one-hour invocation cap on a single cohort and scored zero on it |
| One fixed linear-algebra orientation | A sample-by-sample Gram matrix costs 4.6 to 32.8 GB on the largest cohorts against a 14 GiB cap, where the marker-space solve peaks at 1.6 GB; a submission that cannot switch orientations pays for it on one end of the size range or the other |
| Biased marker sampling | Sampling is the intended lever, but sampling correlated with allele frequency, missingness, or record width trips the coverage and equivalence probes and drags accuracy below the unlock, at which point every optimisation bought nothing |

Every shipped rollout was audited for reward hacking and is clean.

## Running

Replay the reference solution with the same verifier the agent is scored against:

```bash
harbor run -p delivery/scientific-problems-and-ml-tasks/population-genetics-pca --agent oracle -k 1 -o jobs/
```

The reference solution replays at 0.9999 against a 0.89 floor. Grading is wall-clock-relative, so run on non-burstable hardware (the task grades on 8 CPUs and 64 GB, CPU-only).
