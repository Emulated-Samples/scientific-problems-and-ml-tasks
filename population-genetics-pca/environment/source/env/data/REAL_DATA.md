# Private observed-cohort input

The deployed evaluator derives one reviewed arm from the official 1000 Genomes
Project phase-3 GRCh37 chromosome-22 genotype callset and sample panel. The
source and derived data stay outside the repository and solver workspace; only
a root-owned hardlink enters a private grading directory.

Sources:

- `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz`
- `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel`

Reviewed artifact contract:

- compressed VCF bytes: `205612353`
- compressed VCF SHA-256:
  `a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5`
- panel SHA-256:
  `b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858`
- derived cohort: exactly 100 samples from each reviewed super-population,
  selected by a versioned content-key ranking
- records: a deterministic one-in-seven sample based on the five identifying
  VCF columns, spanning the full chromosome without conditioning on record
  width, FORMAT, allele frequency, or genomic region
- derived samples: `500`
- reviewed super-populations: `AFR`, `AMR`, `EAS`, `EUR`, `SAS`

The selected cohort and records are stable within a benchmark release. Every newly
derived bundle receives a private representation key used only for opaque sample aliases
and column order; a valid cached bundle retains its key and bytes. Those transformations
preserve the score subspace up to its allowed row permutation, so they resist public-panel
joins without adding evaluation noise.

`environment/provision.sh` performs that derivation for Hyperfocal. The packaged
Harbor task uses the same mirrored derivation script inside a separate private
verifier image (`tasks/from_scratch_pca/tests/Dockerfile`); the agent image never
contains the source, derived cohort, grader, or truth. Both paths record and
revalidate the derived byte count and SHA-256 before grading. A missing, stale,
or malformed bundle is an infrastructure error; the evaluator never silently
omits this arm.

The solver prompt intentionally does not identify this cohort. The submission
receives the same opaque sandbox path shape used by every other invocation.
