#!/usr/bin/env python3
"""Export Workers Logs (the observability platform's event store) to local JSONL.

Both Workers (`tiles`, `atc`) run with `[observability] enabled = true` at a
10 % head-sampling rate, so every sampled invocation log, console.* line and
uncaught exception lands in Workers Logs — which keeps them for seven days and
then drops them. This pulls the whole account's event store down before it
expires: one gzipped JSONL file per UTC day under
worklists/data/analytics/workers_logs/ (gitignored), events verbatim as the
telemetry query API returns them, so nothing is decided for the analysis here.

Re-running is safe: a day fetched after it ended is complete and skipped
unless --refresh is given; a day fetched while still running is re-fetched
(merged by $metadata.id, so events that have since aged out are kept).
Run it at least weekly or the gap is permanent.

Paging: the events view returns at most 2000 events per call and pages with a
cursor ($metadata.id of the last event). Each hour is paged to exhaustion; a
cursor that stops yielding new events while pages are still full splits the
window in half instead of silently truncating.

Auth: the wrangler OAuth token can't carry a Workers Observability scope, so
this needs an API token with Account > Workers Observability > Edit (the query
endpoint is a POST and requires the write permission). Looked up as
CF_OBSERVABILITY_TOKEN in the environment, then in the repo's .env.

Usage:
  ~/venv/bin/python scripts/workers_logs_export.py              # everything held (7 days)
  ~/venv/bin/python scripts/workers_logs_export.py --days 2
  ~/venv/bin/python scripts/workers_logs_export.py --refresh
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ACCOUNT_ID = "ad20eec906d1b5a42931b881ed22232f"
ENDPOINT = (f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
            "/workers/observability/telemetry/query")
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "worklists/data/analytics/workers_logs"
MANIFEST_NAME = "_manifest.json"
TOKEN_VAR = "CF_OBSERVABILITY_TOKEN"
RETENTION = timedelta(days=7)
PAGE = 2000  # API maximum for view=events


def api_token():
    tok = os.environ.get(TOKEN_VAR)
    if tok:
        return tok
    env = REPO / ".env"
    if env.exists():
        m = re.search(rf"^\s*{TOKEN_VAR}\s*=\s*['\"]?([^'\"\s]+)", env.read_text(), re.M)
        if m:
            return m.group(1)
    sys.exit(f"no {TOKEN_VAR} in the environment or {env}\n"
             "Create an API token with Account > Workers Observability > Edit "
             "and add it to .env as\n"
             f"  {TOKEN_VAR}=<token>")


def ms(dt):
    return int(dt.timestamp() * 1000)


def query(body, token, tries=6):
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(
            ENDPOINT, data=data, method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())["result"]
        except urllib.error.HTTPError as e:
            text = e.read().decode()[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                wait = 2 ** attempt * 2
                print(f"  · HTTP {e.code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            hint = ""
            if e.code in (401, 403) or '"code":10000' in text:
                hint = (f"\n{TOKEN_VAR} must be an API token with "
                        "Account > Workers Observability > Edit.")
            sys.exit(f"telemetry query HTTP {e.code}: {text}{hint}")
        except urllib.error.URLError as e:
            if attempt < tries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            sys.exit(f"telemetry query failed: {e}")


def page(start, end, token, cursor=None):
    body = {
        "queryId": "workers-logs-export",
        "dry": True,  # don't save every page as a query in the dashboard history
        "view": "events",
        "limit": PAGE,
        "timeframe": {"from": ms(start), "to": ms(end)},
        "parameters": {},
    }
    if cursor:
        body["offset"] = cursor
        body["offsetDirection"] = "next"
    events = (query(body, token) or {}).get("events") or {}
    return events.get("events") or []


def event_id(ev):
    return (ev.get("$metadata") or {}).get("id")


def fetch_window(start, end, token, depth=0):
    """Every event in [start, end), keyed by $metadata.id."""
    got, cursor = {}, None
    while True:
        evs = page(start, end, token, cursor)
        fresh = 0
        for ev in evs:
            k = event_id(ev) or json.dumps(ev, sort_keys=True)
            if k not in got:
                got[k] = ev
                fresh += 1
        if len(evs) < PAGE:
            return got
        if not fresh or not event_id(evs[-1]):
            break  # full page but the cursor is going nowhere
        cursor = event_id(evs[-1])
    span = end - start
    if depth >= 8 or span <= timedelta(seconds=30):
        print(f"  ! {start:%m-%d %H:%M:%S} +{span}: cursor stalled at {len(got)} "
              "events; window may be truncated", file=sys.stderr)
        return got
    mid = start + span / 2
    print(f"  · splitting {start:%m-%d %H:%M:%S}..{end:%H:%M:%S} (cursor stalled)")
    got.update(fetch_window(start, mid, token, depth + 1))
    got.update(fetch_window(mid, end, token, depth + 1))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, help="only the last N days (default: all 7 held)")
    ap.add_argument("--refresh", action="store_true", help="re-pull days already complete")
    args = ap.parse_args()

    token = api_token()
    now = datetime.now(timezone.utc)
    oldest = now - RETENTION
    first = oldest.date()
    if args.days:
        first = max(first, now.date() - timedelta(days=args.days - 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def day_end(d):
        return datetime.combine(d + timedelta(days=1), datetime.min.time(),
                                tzinfo=timezone.utc)

    def is_complete(d):
        entry = manifest.get(f"{d:%Y-%m-%d}")
        if not entry:
            return False
        return datetime.fromisoformat(entry["fetched"]) >= day_end(d)

    day = first
    total = 0
    while day <= now.date():
        path = OUT_DIR / f"{day:%Y-%m-%d}.jsonl.gz"
        if path.exists() and not args.refresh and is_complete(day):
            print(f"{day}  complete, skipped")
            day += timedelta(days=1)
            continue
        start = max(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc), oldest)
        end = min(day_end(day), now)
        got = {}
        if path.exists():  # keep events that have aged out of the store since
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    ev = json.loads(line)
                    got[event_id(ev) or line] = ev
        held = len(got)
        h = start
        while h < end:
            h2 = min(h.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), end)
            got.update(fetch_window(h, h2, token))
            h = h2
        evs = sorted(got.values(), key=lambda e: e.get("timestamp") or 0)
        if evs:
            with gzip.open(path, "wt") as fh:
                for ev in evs:
                    fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
        services = {}
        for ev in evs:
            s = (ev.get("$metadata") or {}).get("service") or "?"
            services[s] = services.get(s, 0) + 1
        manifest[f"{day:%Y-%m-%d}"] = {
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from": start.isoformat(timespec="seconds"),
            "events": len(evs),
            "services": dict(sorted(services.items())),
        }
        manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
        split = ", ".join(f"{k} {v}" for k, v in sorted(services.items()))
        print(f"{day}  {len(evs):>7} events (+{len(evs) - held} new)  [{split}]  -> {path.name}")
        total += len(evs)
        day += timedelta(days=1)

    print(f"\n{total} events in {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
