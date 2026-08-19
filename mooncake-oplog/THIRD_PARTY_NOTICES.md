# Third-party notices for this task

Attribution and licence notices for third-party material redistributed in this task.
This file is intentionally placed beside `task.toml`, outside the trees the agent sees at grade
time, so that it travels with the distribution without becoming visible to the model under
evaluation.

## Mooncake (Apache-2.0)

This task is derived from **Mooncake Store** (https://github.com/kvcache-ai/Mooncake),
licensed **Apache-2.0**. Full text at `LICENSES/Apache-2.0.txt`.

The upstream tree is cloned at build time from `environment/Dockerfile`, pinned to parent commit
`d06806d19aba7d67712245fa399f4c33191060c4`. In addition, upstream material is **redistributed
directly in this repository**:

| Path | What it carries |
|---|---|
| `tests/golden/*.cpp`, `*.h`, `tests-CMakeLists.txt` | Mooncake gtest suites, restored over the agent's tree at grade time |
| `environment/contract.patch` | a diff against upstream sources; the interface-contract headers, build wiring, config surface and deployment-guide HA documentation land verbatim |
| `solution/gold.patch` | the upstream implementation the task withholds |
| `environment/specs/*.md` | the design RFC and derived normative specs |
| `environment/tests-CMakeLists-agent.txt` | the upstream test CMakeLists with the withheld registrations removed |

### Statement of changes (Apache-2.0 s4(b))

Emulated modified the upstream work for this evaluation environment:

- the batch-record HA OpLog implementation was **removed** and replaced with one-line `#include`
  stubs (`environment/stub-sources.txt`), and the legacy implementation files were deleted
  (`environment/removed-files.txt`);
- the interface contract was **overlaid** onto the parent tree with newer mtimes
  (`environment/contract.patch`), so the tree does not build until the agent completes it;
- the test suite was **trimmed**: registrations for the withheld tests were deleted from the
  agent-visible CMakeLists, and the corresponding sources are held in `tests/golden/`;
- upstream repository furniture (`.github`, `.claude`, `AGENTS.md`, `CLAUDE.md`, `mooncake-rl`)
  was deleted from the built tree;
- `mooncake-common/etcd/etcd_wrapper.go` was extended with transaction support.

The verifier (`tests/grader.py`, `tests/test.sh`, `tests/grading.json`,
`tests/expected_tests.json`) is Emulated-authored and contains no upstream code.

> **Note carried, not resolved:** the upstream child commit (the PR the gold patch corresponds to)
> is not pinned anywhere in the tree — only the parent commit is. The provenance of
> `tests/golden/` and `solution/gold.patch` rests on the comment in `solution/solve.sh`. Pinning
> that SHA would make this statement of changes verifiable rather than asserted.

## CacheLib (Apache-2.0)

Mooncake vendors Meta's **CacheLib** memory allocator
(https://github.com/facebook/CacheLib, Apache-2.0) as `cachelib_memory_allocator`. That code is
reached two layers down — Emulated → Mooncake → CacheLib — and is redistributed here inside
`environment/contract.patch` (allocator headers) and `solution/gold.patch` (allocator sources).
Upstream copyright headers do not survive in the patch context, so the attribution is recorded
here. **Modified:** `importSlab` / `importAllocations` APIs were added by Mooncake's change.
Full text at `LICENSES/Apache-2.0.txt`.

## RFC text — reuse basis open

`environment/specs/rfc-batch-oplog.md` is upstream-authored RFC prose rather than Emulated
writing: it cites Mooncake issue and PR numbers throughout and still carries the project's GitHub
issue-template footer at the end of the file.

## Build-time dependencies

Fetched on the buyer's machine during `docker build`, not redistributed here: yalantinglibs
(Apache-2.0), pybind11 (BSD-3-Clause), etcd v3.6.1 (Apache-2.0), the Go toolchain 1.25.10
(BSD-3-Clause), GoogleTest (BSD-3-Clause), the `ubuntu:22.04` base image, and the Ubuntu apt
packages named in `environment/Dockerfile` (glog, gflags, gRPC, protobuf, jsoncpp, boost,
hiredis, liburing, jemalloc, msgpack, zeromq, zstd, asio, xxhash, libibverbs). Each retains its
own licence in the built image.
