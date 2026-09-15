#!/usr/bin/env python3
"""Zone-level traffic for archive.aero, and whether it is growing.

The tile log (analytics_export.py) only sees byte-range reads of the PMTiles
archives. This pulls the other half from Cloudflare's zone analytics: page
views, unique visitors, and total requests, split by hostname so the site
(archive.aero) can be read separately from the tile CDN (data.archive.aero).

Two datasets, because neither alone does the job:
  httpRequests1dGroups        whole-zone daily totals incl. pageViews/uniques;
                              queryable across the full retention window
  httpRequestsAdaptiveGroups  adds the hostname dimension, but the API refuses
                              a range wider than ~8 days, so it is fetched in
                              weekly chunks and stitched

Growth is reported as the last N days against the N days before them, which is
the only comparison that survives this site's very spiky day-to-day traffic.

Auth: CLOUDFLARE_API_TOKEN, else the wrangler OAuth token (refresh a stale one
with `npx wrangler whoami`).

Usage:
  ~/venv/bin/python scripts/analytics_traffic.py
  ~/venv/bin/python scripts/analytics_traffic.py --days 7
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ZONE = "ae727fe7ccc5b8d1946b1cdce55ce100"  # archive.aero
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "worklists/data/analytics/traffic.json"
WRANGLER_CFG = Path.home() / "Library/Preferences/.wrangler/config/default.toml"
ENDPOINT = "https://api.cloudflare.com/client/v4/graphql"
# The adaptive dataset rejects a range wider than ~8 days AND refuses anything
# older than ~8 days on this plan, so the hostname split is only ever available
# for the tail of the window, however long the window is.
ADAPTIVE_CHUNK_DAYS = 7
ADAPTIVE_RETENTION_DAYS = 7


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


def gql(query, token):
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
    )
    try:
        payload = json.loads(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as e:
        hint = ("\nThe wrangler OAuth token expires every few days — run "
                "`npx wrangler whoami` to refresh it." if e.code == 401 else "")
        sys.exit(f"GraphQL HTTP {e.code}: {e.read().decode()[:400]}{hint}")
    if payload.get("errors"):
        sys.exit("GraphQL error: "
                 + "; ".join(e.get("message", "?") for e in payload["errors"]))
    return payload["data"]["viewer"]["zones"][0]


def daily_totals(since, until, token):
    q = f'''query {{ viewer {{ zones(filter: {{zoneTag: "{ZONE}"}}) {{
      httpRequests1dGroups(limit: 200,
        filter: {{date_geq: "{since}", date_leq: "{until}"}},
        orderBy: [date_ASC]) {{
        dimensions {{ date }}
        sum {{ requests pageViews bytes cachedRequests }}
        uniq {{ uniques }}
      }} }} }} }}'''
    return [
        {"date": r["dimensions"]["date"], **r["sum"], "uniques": r["uniq"]["uniques"]}
        for r in gql(q, token)["httpRequests1dGroups"]
    ]


def by_host(since, until, token):
    """Daily requests/visits per hostname, stitched from weekly chunks."""
    rows = []
    start = max(since, until - timedelta(days=ADAPTIVE_RETENTION_DAYS - 1))
    while start <= until:
        stop = min(start + timedelta(days=ADAPTIVE_CHUNK_DAYS - 1), until)
        q = f'''query {{ viewer {{ zones(filter: {{zoneTag: "{ZONE}"}}) {{
          httpRequestsAdaptiveGroups(limit: 2000,
            filter: {{datetime_geq: "{start}T00:00:00Z",
                     datetime_lt: "{stop + timedelta(days=1)}T00:00:00Z"}},
            orderBy: [date_ASC]) {{
            dimensions {{ date clientRequestHTTPHost }}
            count
            sum {{ visits edgeResponseBytes }}
          }} }} }} }}'''
        for r in gql(q, token)["httpRequestsAdaptiveGroups"]:
            rows.append({
                "date": r["dimensions"]["date"],
                "host": r["dimensions"]["clientRequestHTTPHost"],
                "requests": r["count"],
                "visits": r["sum"]["visits"],
                "bytes": r["sum"]["edgeResponseBytes"],
            })
        start = stop + timedelta(days=1)
    return rows


def split(rows, key, recent_days, last_day):
    """(recent values, prior values) for `key`, as two lists of daily numbers."""
    cut = last_day - timedelta(days=recent_days - 1)
    prev_cut = cut - timedelta(days=recent_days)
    recent, prev = [], []
    for r in rows:
        d = date.fromisoformat(r["date"])
        if cut <= d <= last_day:
            recent.append(r[key] or 0)
        elif prev_cut <= d < cut:
            prev.append(r[key] or 0)
    return recent, prev


def median(values):
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def pct(recent, prev):
    if not prev:
        return "n/a" if not recent else "+∞"
    return f"{(recent - prev) / prev * 100:+.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14,
                    help="length of the recent window (compared with the "
                         "equal window before it)")
    args = ap.parse_args()
    token = api_token()

    # Yesterday is the last COMPLETE UTC day. Ending the recent window on the
    # current, partial day compared N-1 full days plus a fraction against N
    # full days: totals read low by up to 1/N and growth read pessimistic.
    last_day = datetime.now(timezone.utc).date() - timedelta(days=1)
    first_day = last_day - timedelta(days=args.days * 2 - 1)
    print(f"archive.aero zone — {first_day} .. {last_day} "
          f"({args.days}d vs previous {args.days}d; complete UTC days only)\n")

    totals = daily_totals(first_day, last_day, token)
    hosts = by_host(first_day, last_day, token)
    host_from = max(first_day, last_day - timedelta(days=ADAPTIVE_RETENTION_DAYS - 1))

    # Both lenses, because they disagree when they matter: a one-day crawl can
    # add five figures to a window total while leaving the typical day alone.
    # The median is the growth number to quote; the total is the load number.
    print(f"{'metric':<20}{'median/day now':>15}{'was':>9}{'chg':>8}"
          f"{'  |':>3}{'window total':>14}{'was':>11}{'chg':>8}")
    print("-" * 89)
    summary = {}
    for label, key in (("page views", "pageViews"), ("unique visitors", "uniques"),
                       ("requests (all)", "requests"), ("bytes served", "bytes")):
        recent, prev = split(totals, key, args.days, last_day)
        rs, ps = sum(recent), sum(prev)
        rm, pm = median(recent), median(prev)
        summary[key] = {
            "recent_total": rs, "prior_total": ps, "total_change": pct(rs, ps),
            "recent_median_day": rm, "prior_median_day": pm,
            "median_change": pct(rm, pm),
        }
        if key == "bytes":
            cells = (f"{rm / 1e9:.2f} GB", f"{pm / 1e9:.2f} GB",
                     f"{rs / 1e9:.1f} GB", f"{ps / 1e9:.1f} GB")
        else:
            cells = (f"{rm:,.0f}", f"{pm:,.0f}", f"{rs:,}", f"{ps:,}")
        print(f"{label:<20}{cells[0]:>15}{cells[1]:>9}{pct(rm, pm):>8}"
              f"{'  |':>3}{cells[2]:>14}{cells[3]:>11}{pct(rs, ps):>8}")

    print("\nunique visitors is a per-day distinct count, so its window total "
          "double-counts\nanyone who returned on another day; the median column "
          "is the sound one.")

    # The hostname dimension lives in the short-retention dataset, so this
    # covers only the tail of the window and cannot be compared with a prior
    # period. Reported as a share of the split instead of as growth.
    print(f"\nby hostname, {host_from} .. {last_day} "
          f"(all the hostname dataset retains)")
    print(f"{'hostname':<30}{'requests':>12}{'share':>9}{'visits':>10}")
    print("-" * 61)
    host_summary = {}
    grand = sum(r["requests"] for r in hosts) or 1
    for host in sorted({r["host"] for r in hosts}):
        rows = [r for r in hosts if r["host"] == host]
        reqs = sum(r["requests"] for r in rows)
        visits = sum(r["visits"] or 0 for r in rows)
        if reqs < 50:
            continue
        host_summary[host] = {"requests": reqs, "visits": visits,
                              "share": round(reqs / grand * 100, 1)}
        print(f"{host:<30}{reqs:>12,}{reqs / grand * 100:>8.1f}%{visits:>10,}")

    daily = {t["date"]: t for t in totals}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zone": "archive.aero",
        "window_days": args.days,
        "range": [str(first_day), str(last_day)],
        "totals": summary,
        "by_host": host_summary,
        "by_host_range": [str(host_from), str(last_day)],
        "daily": daily,
        "daily_by_host": hosts,
    }, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
