#!/usr/bin/env bash
set -euo pipefail

if [ "$(/usr/bin/id -u)" -ne 0 ]; then
  echo "pcabench provisioning must run as root" >&2
  exit 1
fi
if [ "$(/usr/bin/uname -s)" != "Linux" ]; then
  echo "pcabench rollouts require Linux" >&2
  exit 1
fi

# Request FULL curl (not curl-minimal) and pass --allowerasing (V1-F2 env-side
# fix, matching sc-silentbench's install-toolchains.sh). The fleet task-base
# image ships full `curl` swapped in over curl-minimal, so `dnf install
# curl-minimal` dies on the curl/curl-minimal conflict; full curl provides the
# same /usr/bin/curl this script's downloads use, and --allowerasing lets the
# swap happen cleanly on a bare amazonlinux:2023 base (dev/no-base-image path)
# where curl-minimal is preinstalled.
missing_packages=()
for package in bubblewrap curl python3.12 python3.12-pip util-linux; do
  if ! /usr/bin/rpm --quiet --query "$package"; then
    missing_packages+=("$package")
  fi
done
if [ "${#missing_packages[@]}" -gt 0 ]; then
  /usr/bin/dnf install -y --setopt=install_weak_deps=False --allowerasing "${missing_packages[@]}"
fi

if ! /usr/bin/getent passwd pcasub >/dev/null; then
  /usr/sbin/useradd --system --home-dir /nonexistent --no-create-home \
    --shell /usr/sbin/nologin pcasub
fi
if [ "$(/usr/bin/id -u pcasub)" -eq 0 ]; then
  echo "pcabench sandbox account must not be root" >&2
  exit 1
fi

runtime=/opt/hyperfocal/pcabench
requirements="$(/usr/bin/dirname "$0")/runtime-requirements.txt"
repo_root="$(/usr/bin/cd "$(/usr/bin/dirname "$0")/.." && /usr/bin/pwd)"
/usr/bin/python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
/usr/bin/python3.12 -m venv --clear "$runtime"
"$runtime/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  --upgrade "pip==26.0.1"
"$runtime/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  --only-binary=:all: --requirement "$requirements"
"$runtime/bin/python" - <<'PY'
from importlib.metadata import version
import sys

assert sys.version_info[:2] == (3, 12), sys.version
assert version("numpy") == "2.2.6"
assert version("scipy") == "1.15.3"
PY
/bin/chown -R root:root "$runtime"
/bin/chmod -R a+rX,go-w "$runtime"

secret_root=/opt/hyperfocal/pcabench-secrets
math_key="$secret_root/release-v3.math-key"
/usr/bin/install -d -m 0700 -o root -g root "$secret_root"
/usr/bin/install -m 0400 -o root -g root \
  "$repo_root/grader/private/release-v3.math-key" "$math_key"

guard=/opt/hyperfocal/pcabench-guard
/usr/bin/install -d -m 0555 -o root -g root "$guard"
/usr/bin/install -m 0444 -o root -g root \
  "$(/usr/bin/dirname "$0")/../grader/submission_runner.py" \
  "$guard/submission_runner.py"

# Host-backed bwrap state must live below a root-owned, non-listable parent.  Keeping it outside
# arbitrary grader output paths makes the launch boundary independent of their ancestor modes.
/usr/bin/install -d -m 0711 -o root -g root /opt/hyperfocal/pcabench-work

# --- verifier-container chroot -------------------------------------------------
# The static root-owned filesystem the grader's `--isolation verifier-container`
# mode chroots into (grade.py _CHROOT_ROOT). Packaged harbor tasks use that mode
# because harbor's ordinary Docker verifier container is not granted
# CAP_SYS_ADMIN, so bubblewrap cannot create its namespaces and its preflight
# dies with "No permissions to creating new namespace"; chroot(2) needs only
# CAP_SYS_CHROOT, which the verifier container DOES have. This block runs at bake
# (setupProblem during docker build), where CAP_MKNOD is available for the device
# nodes. The recipe mirrors the env's own reviewed verifier image
# (tasks/from_scratch_pca/tests/Dockerfile): the chroot holds ONLY the pinned
# runtime and ordinary system libraries -- no grader, truth, datasets, procfs, or
# sysfs. Native EC2 rollouts use bwrap and never enter this root.
sandbox_root=/opt/hyperfocal/pcabench-sandbox-root
/usr/bin/rm -rf "$sandbox_root"
/usr/bin/install -d -m 0555 -o root -g root "$sandbox_root"
/usr/bin/cp -a /usr "$sandbox_root/usr"
/usr/bin/install -d -m 0555 -o root -g root "$sandbox_root/opt" "$sandbox_root/opt/hyperfocal"
/usr/bin/cp -a "$runtime" "$sandbox_root/opt/hyperfocal/pcabench"
/usr/bin/ln -s usr/bin "$sandbox_root/bin"
/usr/bin/ln -s usr/lib "$sandbox_root/lib"
/usr/bin/ln -s usr/lib64 "$sandbox_root/lib64"
/usr/bin/ln -s usr/sbin "$sandbox_root/sbin"
/usr/bin/install -d -m 0555 -o root -g root \
  "$sandbox_root/etc" "$sandbox_root/dev" "$sandbox_root/run" "$sandbox_root/var"
for etc_file in ld.so.cache passwd group nsswitch.conf; do
  if [ -f "/etc/$etc_file" ]; then
    /usr/bin/cp -a "/etc/$etc_file" "$sandbox_root/etc/"
  fi
done
/usr/bin/install -d -m 1777 -o root -g root \
  "$sandbox_root/tmp" "$sandbox_root/var/tmp" "$sandbox_root/run/lock" \
  "$sandbox_root/dev/shm" "$sandbox_root/dev/mqueue"
/usr/bin/install -d -m 0711 -o root -g root "$sandbox_root/work"
/usr/bin/mknod -m 0666 "$sandbox_root/dev/null" c 1 3
/usr/bin/mknod -m 0666 "$sandbox_root/dev/zero" c 1 5
/usr/bin/mknod -m 0666 "$sandbox_root/dev/random" c 1 8
/usr/bin/mknod -m 0666 "$sandbox_root/dev/urandom" c 1 9
# The guard the grader execs inside the chroot (same immutable submission_runner
# as /opt/hyperfocal/pcabench-guard, but on the chroot's own filesystem).
/usr/bin/install -d -m 0555 -o root -g root "$sandbox_root/opt/hyperfocal/pcabench-guard"
/usr/bin/install -m 0444 -o root -g root \
  "$guard/submission_runner.py" \
  "$sandbox_root/opt/hyperfocal/pcabench-guard/submission_runner.py"
# grade.py hardcodes _CHROOT = /usr/sbin/chroot; guarantee it resolves on this
# base image (coreutils may install only /usr/bin/chroot).
if [ ! -x /usr/sbin/chroot ]; then
  chroot_bin="$(command -v chroot || true)"
  if [ -n "$chroot_bin" ]; then
    /usr/bin/ln -sf "$chroot_bin" /usr/sbin/chroot
  fi
fi
/usr/bin/test -x /usr/sbin/chroot
# --- end verifier-container chroot ---------------------------------------------

# The solver is promised Python 3.12 with NumPy/SciPy. Expose that exact pinned
# interpreter on the ordinary agent PATH so solutions never need to discover or
# hard-code a grader-private /opt path.
for command in python python3 python3.12; do
  /usr/bin/tee "/usr/local/bin/$command" >/dev/null <<EOF
#!/usr/bin/env bash
exec $runtime/bin/python "\$@"
EOF
  /bin/chmod 0755 "/usr/local/bin/$command"
done
/usr/local/bin/python3 -c 'import numpy, scipy, sys; assert sys.version_info[:2] == (3, 12)'

# The scored real-data arm is the official 1000 Genomes phase-3 GRCh37
# chromosome-22 callset. Keep the immutable bundle outside the repository and
# solver workspace; runTests hardlinks it into the root-owned grade directory.
data_root=/opt/hyperfocal/pcabench-data
vcf="$data_root/chr22.vcf"
truth="$vcf.truth.json"
panel="$data_root/integrated_call_samples_v3.20130502.ALL.panel"
representation_key="$data_root/.observed-representation.key"
source_url=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
panel_url=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
source_sha256=a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5
panel_sha256=b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858
source_size=205612353

/usr/bin/install -d -m 0700 -o root -g root "$data_root"
bundle_valid=0
if [ -f "$vcf" ] && [ -f "$truth" ] && [ -f "$panel" ] && [ -f "$representation_key" ]; then
  if PYTHONPATH="$repo_root" "$runtime/bin/python" - \
      "$truth" "$math_key" "$vcf" "$source_sha256" "$panel" "$panel_sha256" \
      "$representation_key" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from data.prepare_1kg import DERIVATION_VERSION
from data.release_key import RELEASE_ID, key_commitment, read_math_key

truth_path, key_path, vcf_path, source_sha, panel_path, panel_sha, representation_path = sys.argv[1:]
truth = json.load(open(truth_path))
spec = truth["spec"]
assert spec["derivation_version"] == DERIVATION_VERSION
assert spec["math_release"] == RELEASE_ID
assert spec["math_key_commitment"] == key_commitment(read_math_key(key_path))
assert spec["source_sha256"] == source_sha
assert spec["category"] == "observed" and spec["k"] == 8
assert spec["n_records"] >= 50_000
assert truth["n_samples"] == 500
assert truth["superpop_names"] == ["AFR", "AMR", "EAS", "EUR", "SAS"]
assert len(truth["population_names"]) >= 20
assert len(truth["sample_ids"]) == len(truth["sample_pop"]) == len(truth["sample_subpop"]) == 500
assert Path(representation_path).stat().st_size == 32

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()

assert Path(vcf_path).stat().st_size == spec["derived_bytes"]
assert digest(vcf_path) == spec["derived_sha256"]
assert digest(panel_path) == panel_sha
PY
  then
    bundle_valid=1
  fi
fi
if [ "$bundle_valid" -ne 1 ]; then
  compressed="$data_root/.chr22.download.vcf.gz"
  panel_download="$data_root/.panel.download"
  derived="$data_root/.chr22.vcf.derived"
  /usr/bin/rm -f "$compressed" "$panel_download" "$derived" "$derived.truth.json" "$representation_key"
  "$runtime/bin/python" - "$representation_key" <<'PY'
import secrets
import sys

with open(sys.argv[1], "xb") as destination:
    destination.write(secrets.token_bytes(32))
PY
  /usr/bin/curl --proto '=https' --tlsv1.2 -fL --retry 5 --retry-all-errors \
    --output "$compressed" "$source_url"
  /usr/bin/test "$(/usr/bin/stat -c %s "$compressed")" -eq "$source_size"
  /usr/bin/test "$(/usr/bin/sha256sum "$compressed" | /usr/bin/cut -d' ' -f1)" = "$source_sha256"
  /usr/bin/curl --proto '=https' --tlsv1.2 -fL --retry 5 --retry-all-errors \
    --output "$panel_download" "$panel_url"
  /usr/bin/test "$(/usr/bin/sha256sum "$panel_download" | /usr/bin/cut -d' ' -f1)" = "$panel_sha256"
  /usr/bin/mv "$panel_download" "$panel"
  PYTHONPATH="$repo_root" "$runtime/bin/python" \
    "$(/usr/bin/dirname "$0")/../data/prepare_1kg.py" \
    "$compressed" "$panel" "$derived" \
    --name observed_population_structure --category observed --weight 2.0 --k 8 \
    --samples-per-superpop 100 --record-modulus 7 \
    --math-key-file "$math_key" --representation-key-file "$representation_key" \
    --source-sha256 "$source_sha256"
  /usr/bin/mv "$derived" "$vcf"
  /usr/bin/mv "$derived.truth.json" "$truth"
  /usr/bin/rm -f "$compressed"
fi
PYTHONPATH="$repo_root" "$runtime/bin/python" - "$truth" "$source_sha256" "$math_key" <<'PY'
import json
import sys

from data.prepare_1kg import DERIVATION_VERSION
from data.release_key import RELEASE_ID, key_commitment, read_math_key

truth = json.load(open(sys.argv[1]))
assert truth["spec"]["source_sha256"] == sys.argv[2]
assert truth["spec"]["derivation_version"] == DERIVATION_VERSION
assert truth["spec"]["math_release"] == RELEASE_ID
assert truth["spec"]["math_key_commitment"] == key_commitment(read_math_key(sys.argv[3]))
assert truth["spec"]["category"] == "observed"
assert truth["spec"]["k"] == 8
assert len(truth["spec"]["derived_sha256"]) == 64
assert truth["spec"]["n_records"] >= 50_000
assert truth["n_samples"] == 500
assert truth["superpop_names"] == ["AFR", "AMR", "EAS", "EUR", "SAS"]
assert len(truth["population_names"]) >= 20
assert len(truth["sample_ids"]) == len(truth["sample_pop"]) == len(truth["sample_subpop"]) == 500
assert all(len(sample) == 21 and sample.startswith("p") for sample in truth["sample_ids"])
assert not any(sample.startswith(("HG", "NA")) for sample in truth["sample_ids"])
assert sorted(truth["sample_pop"].count(index) for index in range(5)) == [100] * 5
assert min(truth["sample_subpop"].count(index) for index in set(truth["sample_subpop"])) >= 10
PY
/usr/bin/test "$(/usr/bin/stat -c %s "$vcf")" -ge 100000000
/usr/bin/test "$(/usr/bin/stat -c %s "$vcf")" -le 2000000000
/usr/bin/test "$(/usr/bin/sha256sum "$vcf" | /usr/bin/cut -d' ' -f1)" = \
  "$("$runtime/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["spec"]["derived_sha256"])' "$truth")"
/bin/chown root:root "$vcf" "$truth" "$panel" "$representation_key" "$math_key"
/bin/chmod 0444 "$vcf"
/bin/chmod 0400 "$truth" "$panel" "$representation_key" "$math_key"

test -x /usr/bin/bwrap
test -x /usr/bin/setpriv
test -r "$guard/submission_runner.py"

# Evaluator-source seal (defense-in-depth for hfdev MEASUREMENT; the deployed Harbor eval is already
# safe -- separate minimal solver container per environment/Dockerfile). During an hfdev rollout the
# coding agent runs in $repo_root/workspace with the whole repo checked out beside it, so without
# this it could copy the gold solver (reference/fast_pca.py) or the ready-made reference/
# fast_submission/pca into its own submission and fake a near-perfect score -- the task's "don't
# inspect sibling paths" instruction is an alignment rule the filesystem did not enforce.
#
# Deny the coding agent every top-level checkout entry except its workspace and Hyperfocal's agent
# launcher. An allowlist is the only durable boundary: enumerating today's evaluator paths leaves
# tomorrow's new test, script, cache, or Git metadata readable. The orchestrator starts the agent by
# executing ``packages/env-orchestrator/bin/env-orchestrator.js`` *as the unprivileged agent*, so that
# platform-owned launcher tree must remain searchable/readable. It contains no task evaluator source.
# The prompt is delivered by the harness before the agent starts, and grading runs as root/from the
# separate verifier image; no other checkout entry is solver-visible.
#
# This is fail-closed. An unsealed rollout is not a trustworthy measurement, so a permission failure
# must abort setup instead of silently letting an agent copy the gold and manufacture a top score.
workspace="$repo_root/workspace"
agent_launcher="$repo_root/packages"
if [ ! -d "$workspace" ] || [ -L "$workspace" ]; then
  echo "coding workspace is missing or not an ordinary directory: $workspace" >&2
  exit 1
fi
if [ ! -d "$agent_launcher" ] || [ -L "$agent_launcher" ]; then
  echo "agent launcher is missing or not an ordinary directory: $agent_launcher" >&2
  exit 1
fi
top_level_entries="$runtime/top-level-checkout-entries"
# Keep ``find`` as a foreground command: a process-substitution producer's failure is invisible to
# ``set -e``, which could otherwise leave a partial inventory and silently expose an undiscovered
# evaluator path. The explicit command must finish successfully before permissions are accepted.
/usr/bin/find "$repo_root" -mindepth 1 -maxdepth 1 -print0 > "$top_level_entries"
while IFS= read -r -d '' entry; do
  if [ "$entry" = "$workspace" ] || [ "$entry" = "$agent_launcher" ]; then
    continue
  fi
  if [ -L "$entry" ] || { [ ! -d "$entry" ] && [ ! -f "$entry" ]; }; then
    echo "unreviewed top-level environment entry: $entry" >&2
    exit 1
  fi
  /bin/chown root:root "$entry"
  if [ -d "$entry" ]; then
    /bin/chmod 0700 "$entry"
  else
    /bin/chmod 0600 "$entry"
  fi
done < "$top_level_entries"
/usr/bin/rm -f "$top_level_entries"
/bin/chown root:root "$repo_root"
# Search-only access lets the unprivileged agent reach /hyperfocal/env/workspace without listing or
# reading the checkout around it. The workspace's existing ownership/mode remain untouched.
/bin/chmod 0711 "$repo_root"

echo "pcabench runtime provisioned at $runtime"
