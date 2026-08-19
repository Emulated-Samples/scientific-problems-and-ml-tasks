# pcabench Hyperfocal wrapper

The single problem id is `from_scratch_pca`. Setup provisions pinned Python 3.12,
NumPy, SciPy, bubblewrap, and the dedicated `pcasub` account. It exposes that
exact interpreter as `python`, `python3`, and `python3.12` on the solver PATH,
then resets the problem workspace. The solver writes `submission/pca` and
optional regular helper files beside it. Tests generate a fresh hidden synthetic
suite through the pinned interpreter, stages the pinned real chromosome-22 arm,
then reports one continuous score per scientific challenge category. Reviewed
category weights preserve the importance of harder and observed-data evidence. One private
category separates leading-subspace selection from merely recovering some genuine structured
axes by retaining identifiable structure on both sides of the requested rank.

The solver uses kernel-enforced `linux-user` workspace isolation. Test execution
adds a second boundary: root stages the submission into a private invocation
snapshot through no-follow descriptors, seals the grader/reference tree, and
uses that same immutable snapshot for scanning, probes, and scoring. A dedicated
uid inside bubblewrap receives only system libraries, the pinned runtime,
read-only submission/input mounts, isolated temporary storage, and one writable
output file. The random grading root is searchable but not listable so the
dropped uid can reach exact bind sources, while datasets stay owner-only and the
owner-only reward is created and read without following links. Network and host
processes are absent, resource limits fail closed, and a preflight verifies
`/root`, an outside sentinel, and host networking are unavailable before grading.

Category `score` is the bounded continuous performance reward. Category
`status` is independent of that scalar: integrity, category-specific mean and minimum accuracy,
relevant method factors, and any calibrated systems requirement must all hold. Weaker scientific
or systems work keeps its continuous partial credit without counting as a full pass. Duration is
the sum of submission and reference grader seconds for the category, converted to milliseconds,
and `output` contains per-dataset accuracy, reward, speed, gates, exit, timings, sampled
primary-invocation peak PSS and temporary-storage use, and stderr tail. The resource fields are not
continuous maxima and do not include probe peaks. An invocation skipped after the global deadline
reports those fields as `null`.

This separation follows Opus 4.8 rollout
`run_019f5a30-9830-711b-8f9d-9f14915c5d7f`: every original synthetic subspace was
accurate, but the implementation remained near the full-scan systems anchor and
had untested parser and sampling weaknesses. The revised suite makes those gaps
independent score dimensions instead of treating contract-valid execution as
scientific mastery.
