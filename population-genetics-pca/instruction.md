I need help building a fast population-genetics PCA over large, plain-text, unindexed multi-sample VCF files. Work in the provided workspace and create one regular Python entry file at `submission/pca` (no extension), beginning with a Python shebang. Regular helper files may live beside it under `submission/`.

The program is invoked as:

    pca <vcf_path> <k> <out_path>

The input is VCF text: metadata lines begin `##`, the `#CHROM` header names samples, and each later row is one variant. Genotypes live in sample columns; `GT` is the first FORMAT subfield when present, uses `/` or `|`, and may be missing. Valid inputs can include extra FORMAT fields, missing calls, indels, multiallelic records, symbolic alleles, CRLF line endings, variable-width records, and no final newline.

Eligible markers are polymorphic biallelic SNVs whose REF and ALT are each a single A/C/G/T base, case-insensitively. Skip indels, multiallelic records, symbolic/breakend alleles, and monomorphic sites. A diploid genotype has two alleles from `{0,1}` separated by `/` or `|`; haploid `0` and `1` are pseudo-diploid dosages 0 and 2. Missing, partial, out-of-range, or malformed calls are missing.

Write a TSV whose exact header is `sample_id<TAB>PC1<TAB>...<TAB>PCk`, followed by one row per VCF sample in `#CHROM` order. Preserve the sample IDs, emit exactly `k` finite numeric columns, and exit zero. The caller supplies `1 <= k < number_of_samples`. Inputs contain at least `k` identifiable, positive-variance sample directions.

For every eligible marker, compute alternate-allele frequency `p` from called genotypes and standardize dosage `x` as

    z = (x - 2p) / sqrt(2 p (1-p))

with missing calls imputed to the marker mean. Return the leading sample PC score subspace of this all-eligible-marker standardized genotype matrix. Do not selectively filter by MAF, LD, missingness, or HWE departure. Exact or approximate numerical methods are acceptable, but an approximation must remain faithful to this defined object and must not systematically exclude markers by allele frequency, linkage, missingness, record width, chromosome, or genomic region.

This is a from-scratch Python 3.12 task. The only permitted third-party packages are NumPy and SciPy generic numerical primitives. Do not use a VCF, genomics, PCA, machine-learning, JIT/compiler, native-extension, dataframe, or distributed-compute package; do not call external genomics tools, compilers, or another language runtime. Work offline and install nothing. Do not encode or reconstruct executable/native payloads and do not use child exec/spawn calls. Pure-Python worker processes may use Linux `fork`, but the submission must not start another executable. Foreign-function interfaces are unavailable: `ctypes` is rejected, and so is any standard-library API that is implemented through it, including the ctypes-backed multiprocessing shared objects (`Value`/`Array`/`sharedctypes`). Fork workers must stay on pure-Python/NumPy/SciPy paths. NumPy and SciPy are already installed in the Python 3.12 interpreter that runs your `pca`, so import them directly; do not build a virtualenv, install anything, or re-exec into a different interpreter. Symbolic links and special files are rejected.

Inputs may exceed memory, and end-to-end runtime includes reading the VCF, computing the scores, and writing the result. Each invocation is limited to 16 GiB of address space per process, 14 GiB aggregate resident memory, 4 GiB aggregate temporary storage, 128 processes, 256 open files, 64 MiB of output, and a one-hour wall-clock limit per invocation; calls on smaller inputs may use a 15-minute limit. Only the named input, output, isolated temporary filesystem, and regular submission files are accessible.

Write the output file IN PLACE. It already exists and is the only writable entry in its directory, so the usual write-a-temp-file-then-rename idiom fails there. While your program runs, the only writable paths are that output file and your temporary directory (`TMPDIR`); your program's own directory is read-only at that point, so put runtime scratch in `TMPDIR` -- not beside your program, and not beside the output. Ship whatever helper modules or data files you like alongside your program; the whole submitted tree is readable when it runs.

You may inspect the supplied VCF's metadata and size plus ordinary process CPU-count/affinity information. Do not inspect sibling paths, Git history, or unrelated filesystem contents. Keep a valid, runnable implementation in place.

You have 24 hours; get something in before then.
