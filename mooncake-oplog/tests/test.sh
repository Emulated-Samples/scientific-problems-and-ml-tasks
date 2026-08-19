#!/usr/bin/env bash
# Harbor verifier entrypoint. Stop agent processes, seal the trusted test
# payload, then grade with grader.py (which always writes the reward file).
set -u

pkill -9 -u agent 2>/dev/null || true
chmod -R o-rwx,g-rwx /tests 2>/dev/null || true
mkdir -p /logs/verifier

reward_file=/logs/verifier/reward.json
export PYTHONUNBUFFERED=1

py="$(command -v python3 || command -v python)"
"$py" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grader.py"
rc=$?

if [[ ! -s "$reward_file" ]]; then
  printf '{"reward":0,"verifier_error":1,"grader_exit":%s}\n' "$rc" > "$reward_file"
fi

exit 0
