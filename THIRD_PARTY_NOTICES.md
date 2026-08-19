# Third-party notices

This repository redistributes third-party open-source material. This file provides the
attribution and notices required by those licences. Full licence texts are in `LICENSES/`.
Per-task detail is in `<task>/THIRD_PARTY_NOTICES.md`.

These notices are deliberately placed **outside** each task's workspace tree (for
`attribution-speedup` and `population-genetics-pca`, outside `environment/source/env/`; for
`mooncake-oplog`, outside the source tree the image builds). The workspaces are decontaminated of
upstream identifiers so that the model under evaluation cannot retrieve an upstream solution;
keeping the notices outside that tree preserves both the integrity of the evaluation and the
attribution the licences require.

## Upstream projects under Apache-2.0

| Component | Licence | Task | Modified? |
|---|---|---|---|
| Mooncake Store (kvcache-ai) | Apache-2.0 | `mooncake-oplog` | **Yes** — implementation withheld and stubbed, interface contract overlaid, test registrations trimmed; statement of changes in the task notice |
| CacheLib (Meta), vendored inside Mooncake as `cachelib_memory_allocator` | Apache-2.0 | `mooncake-oplog` | **Yes** — allocator import APIs added upstream of this delivery |

`mooncake-oplog` redistributes roughly 1.2 MB of upstream material directly (golden test sources,
the interface-contract patch, the gold implementation patch and the specification documents); the
rest of the tree is cloned from upstream at image-build time. The statement of changes required by
Apache-2.0 s4(b) is carried in `mooncake-oplog/THIRD_PARTY_NOTICES.md`. Full text at
`LICENSES/Apache-2.0.txt`.

## Permissively licensed components requiring attribution

| Component | Licence | Task | Modified? |
|---|---|---|---|
| `lat` (c) 2026 Ryan "RyanIRL" Peters | MIT | `attribution-speedup` | **Yes** — the fast backward-sweep path is withheld for the task |

`circuit-tracer` (MIT), TransformerLens (MIT), PyTorch, NumPy, SciPy and the rest of the Python
stack are **declared dependencies installed at build time** on the buyer's machine, not
redistributed here. The same is true of `mooncake-oplog`'s yalantinglibs (Apache-2.0), pybind11
(BSD-3-Clause), etcd (Apache-2.0), the Go toolchain (BSD-3-Clause), GoogleTest (BSD-3-Clause) and
its Ubuntu apt packages.

## Data components

| Component | Terms | Task | Modified? |
|---|---|---|---|
| 1000 Genomes phase-3 chr22 callset + sample panel | published without redistribution restriction (IGSR) | `population-genetics-pca` | callset fetched at build time; the sample panel and derived principal components appear in the docs tree |
| PLINK 2.0 output (`plink2.tsv`) | program output; GPL-3.0 does not attach | `population-genetics-pca` | No |
| Attribution-graph fixture `capital_of_france.json` | generated with circuit-tracer over Qwen3-4B (Apache-2.0) | `attribution-speedup` | generated |
| `gelu-2l` weights and GPT-NeoX tokenizer | fetched into the image at build time | `attribution-speedup` | not redistributed |

All other datasets in this repository are synthetic and generated in-repo.

## SauersML components

`population-genetics-pca` names SauersML as its benchmark author; two SauersML repositories were
reviewed and explicitly **not copied**. See `LICENSING.md`.

## Sample trajectories

`trajectory/` carries sample rollouts recorded from Claude Opus 5, GPT-5.6 Sol and Gemini 3.7
Flash. See the repository README for the layout.

