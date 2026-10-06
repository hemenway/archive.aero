#!/bin/sh
# Hash R2 archives in parallel shards by streaming them (no local mirror), writing
# one plan per shard plus sparse directory stubs for next_build_manifest.py.
#
#   scripts/next_stream_hash.sh LISTING.json OUT_DIR STUB_DIR PREFIX [SHARDS]
#
# LISTING is an rclone/S3 inventory (Path or Key + Size). OUT_DIR receives
# shard-N.listing.json / shard-N.json / shard-N.log and, when every shard
# succeeds, plan.json and plan.json.sh. STUB_DIR must be on a local APFS disk:
# the stubs are sparse files whose apparent size is the archive's full size.
# R2 credentials come from the rclone `r2:` remote for this process only;
# nothing is printed or stored.
set -eu
LISTING=$1; OUT=$2; STUBS=$3; PREFIX=$4; SHARDS=${5:-6}
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-"$HOME/venv/bin/python"}
mkdir -p "$OUT" "$STUBS"
eval "$(rclone config dump | "$PY" -c '
import json, sys, shlex
r = json.load(sys.stdin)["r2"]
print("export AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s R2_ENDPOINT_URL=%s"
      % tuple(shlex.quote(r[k]) for k in ("access_key_id", "secret_access_key", "endpoint")))')"
"$PY" -c '
import json, sys
listing, out, shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = json.load(open(listing)); rows = rows.get("objects", rows.get("Contents")) if isinstance(rows, dict) else rows
# Largest objects first, dealt round-robin, so the shards finish at about the same time.
rows.sort(key=lambda o: -(o.get("Size") or o.get("size") or 0))
for i in range(shards): json.dump(rows[i::shards], open(f"{out}/shard-{i}.listing.json", "w"))
print(f"{len(rows)} objects in {shards} shards")' "$LISTING" "$OUT" "$SHARDS"
i=0; while [ "$i" -lt "$SHARDS" ]; do
  "$PY" "$HERE/next_version_archives.py" --listing "$OUT/shard-$i.listing.json" --read-remote --stubs "$STUBS" \
    --prefix "$PREFIX" --out "$OUT/shard-$i.json" > "$OUT/shard-$i.log" 2>&1 &
  i=$((i+1))
done
wait
"$PY" -c '
import json, sys, os
out, shards, here = sys.argv[1], int(sys.argv[2]), sys.argv[3]; sys.path.insert(0, here); records = []
for i in range(shards):
    path = f"{out}/shard-{i}.json"
    if not os.path.exists(path): sys.exit(f"shard {i} produced no plan; see {out}/shard-{i}.log")
    records += json.load(open(path))
from next_version_archives import shell_commands
json.dump(records, open(f"{out}/plan.json", "w"), indent=1); open(f"{out}/plan.json.sh", "w").write(shell_commands(records, "charts"))
print(f"{len(records)} archives planned -> {out}/plan.json (+ .sh)")' "$OUT" "$SHARDS" "$HERE"
