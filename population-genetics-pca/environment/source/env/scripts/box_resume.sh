#!/bin/bash
# Run on the box after it recovers to finish validation + the headline in ONE careful,
# memory-safe pass (one heavy job at a time -- the earlier OOM came from running several
# chr22 full scans concurrently). Assumes /root/pcabench is synced and the
# production runtime has been provisioned.
set -uo pipefail
cd /root/pcabench
PY="/opt/hyperfocal/pcabench/bin/python"

echo "=== rebuild task mirror ==="
"$PY" scripts/build_task.py >/dev/null 2>&1

echo "=== guarantees (FN never rejected + every cheat crushed) ==="
"$PY" scripts/validate.py --data-dir data/generated 2>&1 | grep -E "OK |FAIL|LEAK|RESULT"

echo "=== chr22 headline (min of 4) ==="
"$PY" - <<'PYEOF'
import time
from pathlib import Path
import reference.fast_pca as fp
ts = []
for _ in range(4):
    t = time.perf_counter(); ids, S, kept, meta = fp.fit(Path("data/real/chr22.vcf"), 10); ts.append(time.perf_counter()-t)
print("chr22: n=%d kept=%d engine=%s min=%.2fs" % (len(ids), kept, meta["engine"], min(ts)))
PYEOF
echo "=== done ==="
