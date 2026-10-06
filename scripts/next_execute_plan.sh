#!/bin/sh
# Execute the shard plans written by next_stream_hash.sh in parallel:
# server-side copies for remote records (size / single-part ETag guarded),
# uploads for local ones, refusing destinations that already exist with a
# different SHA-256. R2 credentials come from the rclone `r2:` remote for these
# processes only.
#
#   scripts/next_execute_plan.sh PLAN_DIR [SHARDS] [BUCKET]
set -eu
OUT=$1; SHARDS=${2:-12}; BUCKET=${3:-charts}
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-"$HOME/venv/bin/python"}
eval "$(rclone config dump | "$PY" -c '
import json, sys, shlex
r = json.load(sys.stdin)["r2"]
print("export AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s R2_ENDPOINT_URL=%s"
      % tuple(shlex.quote(r[k]) for k in ("access_key_id", "secret_access_key", "endpoint")))')"
i=0; while [ "$i" -lt "$SHARDS" ]; do
  [ -f "$OUT/shard-$i.json" ] || { echo "missing $OUT/shard-$i.json" >&2; exit 1; }
  "$PY" "$HERE/next_version_archives.py" --from-plan "$OUT/shard-$i.json" --bucket "$BUCKET" > "$OUT/shard-$i.execute.log" 2>&1 &
  i=$((i+1))
done
wait
i=0; failed=0; while [ "$i" -lt "$SHARDS" ]; do
  grep -q "archives executed" "$OUT/shard-$i.execute.log" || { echo "shard $i did not finish; see $OUT/shard-$i.execute.log" >&2; failed=1; }
  i=$((i+1))
done
[ "$failed" -eq 0 ] && echo "all $SHARDS shards executed"
exit $failed
