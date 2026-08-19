Build a fast, from-scratch **population-genetics PCA** over raw VCFs.

You are given multi-sample VCF files on disk and must compute the top principal components of
population structure -- an HWE-normalised (Patterson-Price-Reich 2006) PCA of genotype dosages
-- and you must do it **fast**. The files are large, plain-text, and unindexed.

## What the input looks like

A VCF is plain text: metadata lines starting with `##`, one `#CHROM` header line naming the
samples, then one line per variant. Records are sorted by genomic position within each
chromosome. Genotypes are in the per-sample columns; the `GT` sub-field (always first when
present) is the genotype, `|` phased or `/` unphased, `.` missing. A realistic head:

```
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1>
#CHROM  POS     ID   REF  ALT   QUAL  FILTER  INFO   FORMAT   sample_A  sample_B  sample_C  sample_D
chr1    10177   .    A    AC    100   PASS    .      GT       0|0      0|1      1|1      0|0
chr1    11008   .    C    G     100   PASS    .      GT       0|0      ./.      0|1      1|1
chr1    13110   .    G    A     100   PASS    AC=3   GT:DP    0/0:31   0/1:22   1/1:18   0/1:27
chr1    16145   .    T    G,C   100   PASS    .      GT       0|0      1|2      0|1      2|2
```

Real data is messy: some records are indels (`A`->`AC`), symbolic/structural (`<DEL>`),
multiallelic (`T`->`G,C`), or have missing calls (`./.`); the FORMAT may carry extra
sub-fields (`GT:DP:GQ`). Your program must read the genotypes out of this correctly and not
crash on the messiness.

Inputs may be much larger than available memory. End-to-end runtime includes reading the VCF,
computing the scores, and writing the result.

## Program contract

Provide one program:

- `/app/submission/pca`

Invoked as:

    pca <vcf_path> <k> <out_path>

- `<vcf_path>`: a multi-sample VCF (plain `.vcf`; may be a large, unindexed text file).
- `<k>`: number of principal components to output.
- `<out_path>`: where to write the scores TSV.

`pca` is run as a Python 3.12 script under the runtime interpreter with `numpy` and `scipy`;
there is no build/compile step. Because the entry file is named `pca` (no `.py` extension), start
it with a shebang line containing `python` (e.g. `#!/usr/bin/env python3`) so it is recognised as
a Python program.

`pca` writes `<out_path>`, a TSV with a header row:

    sample_id<TAB>PC1<TAB>PC2<TAB>...<TAB>PCk

exactly one row per sample, **in the order the samples appear in the VCF `#CHROM` header
line**, with the sample IDs from that line in the `sample_id` column. Always emit exactly `k`
finite PC columns; any other shape is invalid.

Write `<out_path>` **in place**. That exact file already exists and is the only writable entry in
its directory: you cannot create a sibling next to it, so the usual write-a-temp-file-then-rename
idiom fails there. Use your own scratch space if you want to stage the bytes, then write them into
`<out_path>` itself.

The program must run non-interactively, read no network, and exit 0 on success. The caller
supplies `1 <= k < number_of_samples`.

Inputs contain at least `k` identifiable, positive-variance sample directions.

## What the PCA must be

For each biallelic SNV with alt-allele frequency `p`, a genotype dosage `x in {0,1,2}` is
standardised as

    z = (x - 2p) / sqrt(2 p (1 - p))

with missing calls imputed to the column mean. The requested output is the leading **sample score
subspace** of the standardised genotype matrix. The `sqrt(2p(1-p))` denominator is required --
it is what makes this a population-genetics PCA rather than a plain covariance PCA.

Consider every biallelic SNV whose REF and ALT are each exactly one of A/C/G/T,
case-insensitively, eligible. Skip
multiallelic records (comma in ALT), symbolic/structural ALTs (`<DEL>`, breakends), indels, and
monomorphic sites. A diploid call contains two alleles from `{0,1}` separated by `/` or `|`.
A haploid `0` or `1` call is pseudo-diploid dosage `0` or `2`. Missing, partial, out-of-range,
or malformed calls are missing and are mean-imputed for an otherwise eligible site.

**Do not apply a *selective* filter on allele frequency or linkage.** Do **not** apply a
minor-allele-frequency (MAF) cutoff, LD pruning, a missingness threshold, or an HWE-departure
test -- those are common in ancestry pipelines but they bias *which* variants enter the fit and
change the result into a *different* object, which is not what this tool computes. Rare variants
belong in the fit: the `sqrt(2p(1-p))` standardisation already weights every site correctly.

Exact and approximate numerical methods are permitted. Any approximation must remain faithful to
the defined all-eligible-marker decomposition and must not systematically exclude markers by
allele frequency, linkage, missingness, record width, chromosome, or genomic region.

## Speed

These VCFs are large. The tool is run repeatedly on many multi-gigabyte files, so fit speed
(VCF on disk -> scores written) matters a great deal -- a correct result that is also fast is
the goal. How you achieve that is up to you.

## From-scratch rule

This is a **pure-Python** task (Python 3.12). Implement the PCA yourself using only the general
numeric stack: **`numpy`** and **`scipy`** (`scipy.linalg`, `scipy.sparse`, `scipy.special`, and
`numpy.linalg.eigh`/`svd` for the eigendecomposition). That is the whole allowed dependency set.

What is **banned**:

- **any other third-party package** -- in particular no JIT/compiler or native-extension
  toolkits (`numba`, `cython`, `cffi`, `ctypes`-loaded native code, `pybind11`, `torch`, `jax`,
  ...), and no shelling out to a compiler or to another language runtime. Write the hot loops in
  numpy/Python;
- encoded or reconstructed executable/native payloads and child `exec`/`spawn` calls. Parallel
  pure-Python workers may use the Linux `fork` start method, but the submitted program must not
  start another executable. Fork workers must remain on pure-Python/NumPy/SciPy paths;
- foreign-function interfaces: `ctypes` is rejected outright -- not merely ctypes-loaded native
  code -- and so is any standard-library API implemented through it, including the ctypes-backed
  multiprocessing shared objects (`Value`/`Array`/`sharedctypes`);
- genotype/VCF ingestion via a genomics library (`pysam`, `cyvcf2`, `hail`, `sgkit`,
  `scikit-allel`, `pandas-plink`, `bed-reader`, `plinkio`, `pgenlib`, ...), or shelling out to
  `plink`/`plink2`/`bcftools`/`gcta`/`flashpca`/`vcftools`;
- the PCA/decomposition itself via a stats/ML library (`sklearn`'s `PCA`/`TruncatedSVD`,
  `statsmodels`, `scikit-allel`'s `pca`, `dask-ml`, ...).

The speed has to come from the algorithm and from numpy/BLAS, not from dropping to another language.

Keep the submission tree ordinary and self-contained: `pca` and any helper/data
files must be regular files under `/app/submission`. Symbolic links, device/FIFO
entries, more than 4096 files, or more than 64 MiB of submitted files are
rejected before execution.

The runtime copies that tree once into an immutable private snapshot. Each
invocation is limited to 16 GiB of address space per process, 14 GiB aggregate
resident memory, 4 GiB of aggregate temporary storage, 128 processes,
256 open files, and 64 MiB per output file. The wall-clock limit is one hour per
invocation; calls on smaller inputs may use a 15-minute limit. Only the named
output and an isolated temporary filesystem are writable.

## What "correct" means

The result must recover the decomposition defined above, up to the usual PCA
rotation, sign, and scale freedoms, across varied valid VCFs. Both numerical
fidelity and end-to-end runtime matter.

## Notes

- Work fully offline: no web, no installing packages; use the pre-installed toolchain.
- Confine file reads and writes to the paths you are given. You may inspect the supplied VCF's
  metadata and size, and ordinary process CPU-count/affinity information. Do not inspect private
  runtime files, sibling paths, Git state, or unrelated filesystem contents.
- Ship whatever helper modules or data files you like under `/app/submission`; that whole tree is
  available to your program. **While `pca` runs, the only paths it can WRITE are `<out_path>` and
  its temporary directory (`TMPDIR`, an isolated scratch filesystem).** Its own directory is
  read-only at that point, so put runtime scratch in `TMPDIR` -- not beside your program, and not
  beside the output.
- Keep a valid, runnable `pca` in place.
- You have 24 hours; get something in before then.
