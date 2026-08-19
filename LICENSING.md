# Licensing

This file records the licence position for third-party material redistributed in this
repository. It is a companion to `THIRD_PARTY_NOTICES.md`, which carries the per-component
attribution, and to the per-task `<task>/THIRD_PARTY_NOTICES.md` files.

## Apache-2.0 upstream: mooncake-oplog

`mooncake-oplog` is derived from kvcache-ai/Mooncake (Apache-2.0). It redistributes upstream
sources, tests, documentation and specification text, and clones the upstream tree at image
build time. The attribution and the statement of changes required by Apache-2.0 section 4(b)
are carried in the task's notice; the full licence text is at `LICENSES/Apache-2.0.txt`.
Meta's CacheLib (Apache-2.0), vendored inside Mooncake, travels with that material and is
attributed there as well.

## MIT upstream: attribution-speedup

`attribution-speedup` ships the `lat` library, MIT, Copyright (c) 2026 Ryan "RyanIRL" Peters.
No other third-party code is redistributed in that task: circuit-tracer (MIT), TransformerLens
(MIT) and the model checkpoint it exercises are fetched at image build time. The full licence
text is at `LICENSES/MIT-lat.txt`.

## SauersML authorship: population-genetics-pca

`population-genetics-pca` records SauersML as the benchmark author. Emulated holds a written
licence grant from the author covering redistribution as part of this dataset and use in
commercial AI model training, fine-tuning and evaluation. No SauersML source code or data is
redistributed in this repository.

## Genomic reference data: population-genetics-pca

The grading path derives its cohort from the 1000 Genomes Project phase-3 GRCh37
chromosome-22 callset and sample panel, which the 1000 Genomes Project publishes without
redistribution restriction. The genotype callset is fetched and hash-checked at image build
time and is not redistributed here. The task notice records which panel-derived files appear
in the documentation tree.

## Everything else

All other third-party material is redistributed under its own licence. See
`THIRD_PARTY_NOTICES.md` for the component-level attribution and `LICENSES/` for full texts.
