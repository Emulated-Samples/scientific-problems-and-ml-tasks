#!/usr/bin/env bash
# Oracle reference solution: apply the upstream implementation (PR #2826,
# implementation files only — tests are the verifier's) onto the staged
# workspace tree and rebuild once to confirm the library and the surviving
# agent-visible suites compile. The graded targets themselves don't exist
# until the verifier restores its golden sources.
set -euo pipefail

ROOT="${SOLUTION_TARGET_ROOT:-/workspace/Mooncake}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

patch -p1 -d "$ROOT" --forward < "$HERE/gold.patch"

cd "$ROOT/build"
make -j"$(nproc)" master_service_test buffer_allocator_test
