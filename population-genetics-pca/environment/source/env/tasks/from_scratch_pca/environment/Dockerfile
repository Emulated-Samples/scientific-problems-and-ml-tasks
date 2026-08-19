# Environment for the from-scratch population-genetics PCA task.
#
# Pure-Python task: the agent parses the VCF and computes the PCA itself. We provide the
# generic numeric stack (numpy / scipy) and nothing that would let it delegate the core --
# no genomics I/O libraries (pysam/cyvcf2/hail/sgkit/scikit-allel/...) and no PCA/decomposition
# libraries (scikit-learn/statsmodels/...). The library-scan gate enforces this at grade time;
# their absence here is defence-in-depth.
#
# SLIM base: the full `python:3.12-bookworm` (buildpack-deps) ships gcc/g++/make, exactly the
# compiler toolchain the task forbids ("no shelling out to a compiler", "no native-extension
# toolkits"). The slim variant carries no compiler, so a submission cannot build a native
# extension at runtime -- the from-scratch rule is enforced by the environment, not only by the
# static AST scan. numpy/scipy install from prebuilt manylinux wheels, so no compiler is needed.
# Pinned to an exact interpreter patch + exact package versions so the numerical behaviour and the
# speed baseline are reproducible across rebuilds (a `3.12`/`numpy==2.*` moving tag is not).
FROM python:3.12.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Generic numeric primitives only, exact versions, in the same fixed runtime path used by the
# private verifier. The solver container intentionally contains neither verifier tooling nor data.
RUN python -m venv /opt/hyperfocal/pcabench \
    && /opt/hyperfocal/pcabench/bin/pip install --no-cache-dir \
        "numpy==2.2.6" \
        "scipy==1.15.3"

# Harbor copies /app/submission into the separate verifier after the agent finishes.
RUN mkdir -p /app/submission
WORKDIR /app/submission

CMD ["/bin/bash"]
