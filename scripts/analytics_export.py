#!/usr/bin/env python3
"""Export the tiles Worker's Analytics Engine log to local JSONL.

Analytics Engine keeps roughly three months of data, so the archive's view
history only accumulates if it is pulled down and kept. This writes one
gzipped JSONL file per UTC day under worklists/data/analytics/tile_logs/
(gitignored, like everything catalog-derived) and is safe to re-run: complete
past days are skipped unless --refresh is given; a day fetched while it was
still in progress is re-fetched later, tracked in _manifest.json.

Rows are pre-aggregated server-side. The grouping keys keep full fidelity for
the heatmap join — (hour, era, byte offset, byte length) — so the export is
lossless for anything analytics_heatmap.py needs while staying well inside the
SQL API's 10,000-row result cap. A chunk that hits the cap is subdivided into
hours and retried, so growing traffic degrades into more queries, not silent
truncation.

Auth: CLOUDFLARE_API_TOKEN (needs Account Analytics Read) if set, otherwise the
OAuth token wrangler already stores for this machine.

Usage:
  ~/venv/bin/python scripts/analytics_export.py                 # since first data
  ~/venv/bin/python scripts/analytics_export.py --days 7
  ~/venv/bin/python scripts/analytics_export.py --refresh       # re-pull every day
"""

import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ACCOUNT_ID = "ad20eec906d1b5a42931b881ed22232f"
DATASET = "tile_logs"
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "worklists/data/analytics/tile_logs"
WRANGLER_CFG = Path.home() / "Library/Preferences/.wrangler/config/default.toml"
# Which days have been fetched, and when. A day fetched while it was still
# running holds only the rows that existed at that moment, so completeness is
# "fetched after the day ended", not "a file exists".
MANIFEST_NAME = "_manifest.json"

# The SQL API truncates a result set at 10k rows with no error; stay under it
# and treat reaching it as "subdivide this window".
ROW_LIMIT = 10000

FIELDS = (
    "toStartOfHour(timestamp) AS hour, blob1 AS key, double3 AS off, double1 AS len, "
    "blob2 AS country, blob3 AS cache, blob4 AS ua, blob5 AS ref, "
    "sum(_sample_interval) AS views, count() AS n, avg(double2) AS ms"
)
GROUP = "GROUP BY hour, key, off, len, country, cache, ua, ref"


def api_token():
    tok = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CF_API_TOKEN")
    if tok:
        return tok
    if not WRANGLER_CFG.exists():
        sys.exit("no CLOUDFLARE_API_TOKEN and no wrangler login found")
    m = re.search(r'oauth_token\s*=\s*"([^"]+)"', WRANGLER_CFG.read_text())
    if not m:
        sys.exit(f"no oauth_token in {WRANGLER_CFG}")
    return m.group(1)


def query(sql, token):
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/analytics_engine/sql",
        data=sql.encode(),
        headers={"Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        hint = ""
        if e.code == 401:
            hint = ("\nThe wrangler OAuth token expires every few days — run "
                    "`npx wrangler whoami` to refresh it, then retry.")
        sys.exit(f"SQL API HTTP {e.code}: {e.read().decode()[:500]}"
                 f"\nquery: {sql[:300]}{hint}")


def fetch_window(start, end, token, depth=0):
    """Rows in [start, end). Subdivides when the result hits the row cap."""
    sql = (
        f"SELECT {FIELDS} FROM {DATASET} "
        f"WHERE timestamp >= toDateTime('{start:%Y-%m-%d %H:%M:%S}') "
        f"AND timestamp < toDateTime('{end:%Y-%m-%d %H:%M:%S}') "
        f"{GROUP} ORDER BY hour LIMIT {ROW_LIMIT}"
    )
    rows = query(sql, token).get("data", [])
    if len(rows) < ROW_LIMIT:
        return rows
    span = end - start
    if depth >= 4 or span <= timedelta(minutes=1):
        print(
            f"  ! {start:%Y-%m-%d %H:%M} +{span}: still {len(rows)} rows at the cap; "
            f"data may be truncated",
            file=sys.stderr,
        )
        return rows
    mid = start + span / 2
    print(f"  · splitting {start:%m-%d %H:%M}..{end:%m-%d %H:%M} (hit row cap)")
    return fetch_window(start, mid, token, depth + 1) + fetch_window(
        mid, end, token, depth + 1
    )


def normalize(row):
    """Trim the row to what downstream needs; era key matches the bundle's."""
    key = row["key"]
    era = key
    if era.startswith("sectionals/"):
        era = era[len("sectionals/") :]
    if era.endswith(".pmtiles"):
        era = era[: -len(".pmtiles")]
    return {
        "hour": row["hour"],
        "era": era,
        "off": int(row["off"]),
        "len": int(row["len"]),
        "country": row["country"],
        "cache": row["cache"],
        "ua": row["ua"],
        "ref": row["ref"],
        "views": int(row["views"]),
        "n": int(row["n"]),
        "ms": round(float(row["ms"]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, help="only the last N days (default: all)")
    ap.add_argument("--refresh", action="store_true", help="re-pull days already on disk")
    args = ap.parse_args()

    token = api_token()
    extent = query(
        f"SELECT min(timestamp) AS a, max(timestamp) AS b, count() AS n FROM {DATASET}",
        token,
    )["data"][0]
    if not int(extent["n"]):
        sys.exit("dataset is empty")
    first = datetime.strptime(extent["a"], "%Y-%m-%d %H:%M:%S").date()
    last = datetime.strptime(extent["b"], "%Y-%m-%d %H:%M:%S").date()
    today = datetime.now(timezone.utc).date()
    if args.days:
        first = max(first, today - timedelta(days=args.days - 1))
    print(f"dataset spans {extent['a']} .. {extent['b']} ({extent['n']} raw rows)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / MANIFEST_NAME
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    def is_complete(d):
        """True only if this day was fetched after the day had finished."""
        stamp = manifest.get(f"{d:%Y-%m-%d}")
        if not stamp:
            return False
        fetched = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        end = datetime.combine(d + timedelta(days=1), datetime.min.time(),
                               tzinfo=timezone.utc)
        return fetched >= end

    total = 0
    day = first
    while day <= last:
        path = OUT_DIR / f"{day:%Y-%m-%d}.jsonl.gz"
        if path.exists() and not args.refresh and is_complete(day):
            total += sum(1 for _ in gzip.open(path, "rt"))
            day += timedelta(days=1)
            continue
        start = datetime.combine(day, datetime.min.time())
        rows = fetch_window(start, start + timedelta(days=1), token)
        if rows:
            with gzip.open(path, "wt") as fh:
                for r in rows:
                    fh.write(json.dumps(normalize(r), separators=(",", ":")) + "\n")
            views = sum(int(r["views"]) for r in rows)
            print(f"{day}  {len(rows):>6} rows  {views:>7} sampled views  -> {path.name}")
        elif path.exists():
            path.unlink()
        manifest[f"{day:%Y-%m-%d}"] = (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        total += len(rows)
        day += timedelta(days=1)

    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    print(f"\n{total} rows in {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
