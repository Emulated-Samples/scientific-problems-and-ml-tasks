# Workspace

Build your population-genetics PCA under `submission/`, as a single executable
Python entry file `submission/pca` (no extension, Python shebang), invoked as:

    pca <vcf_path> <k> <out_path>

See the task prompt for the full program contract (VCF parsing, the required
`z = (x - 2p) / sqrt(2 p (1-p))` HWE standardization, output TSV format, the
NumPy/SciPy-only from-scratch rule, and the speed target). Regular helper files
next to `pca` are allowed; symlinks and special files are rejected.
