#!/usr/bin/env ~/venv/bin/python
"""Sweep the atc worker's Analytics Engine log for unmatched 404s.

The cutover redirect map covers every URL the inventory knew about. This finds
the ones it did not know about — real misses that deserve a redirect-map entry —
and separates them from the noise that will always be there.

Bucketing matters more than counting. Raw 404 volume on this worker is mostly:

  self      our own verification probes (the top-100 inventory sweep replays
            ~73 junk paths; atc_redirect_check.py replays the bogus-sitemap
            guard). Analytics Engine records no client IP, so unlike the
            freeze-day log merge we cannot filter these by address — they are
            recognised by path shape instead.
  scanner   WordPress vulnerability probes (wlwmanifest.xml, xmlrpc.php,
            wp-login.php, /.env). Constant background on any ex-WP domain.
  junk      acme-challenge leftovers, .DS_Store, editor droppings.
  asset     a page's subresource is missing — high priority when the referer is
            one of our own pages, because a live page is visibly broken.
  content   a document URL nobody has mapped. This is the actual deliverable.

Only `content` and internally-refered `asset` rows are worth acting on; the
rest are reported as totals so a spike in them is still visible.

Auth: CLOUDFLARE_API_TOKEN (needs Account Analytics Read) if set, otherwise the
OAuth token wrangler already stores for this machine. That token expires every
few days — `npx wrangler whoami` refreshes it in place.

Usage:
  ~/venv/bin/python scripts/atc_404_sweep.py                 # last 24h
  ~/venv/bin/python scripts/atc_404_sweep.py --hours 72
  ~/venv/bin/python scripts/atc_404_sweep.py --since '2026-09-01 04:35:00'
  ~/venv/bin/python scripts/atc_404_sweep.py --no-probe      # skip live probes
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import subprocess
import urllib.request
from pathlib import Path

ACCOUNT_ID = "ad20eec906d1b5a42931b881ed22232f"
DATASET = "atc_logs"
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "worklists/data/atc"
WRANGLER_CFG = Path.home() / "Library/Preferences/.wrangler/config/default.toml"
ROW_LIMIT = 10000

# Worker blob layout, from log() in worker-atc/src/index.js:
#   blobs = [host, path[:96], referer[:96]]   doubles = [status]
# Note the 96-char truncation: a long path arrives clipped, so a reported miss
# may need widening before it can be matched against the maps.
FIELDS = ("blob1 AS host, blob2 AS path, blob3 AS ref, count() AS n, "
          "min(timestamp) AS first, max(timestamp) AS last")

OUR_HOSTS = {"archive.aero", "atchistory.org", "www.atchistory.org",
             "atc-staging.archive.aero"}
# Staging is a rehearsal surface — its 404s are our own test traffic and say
# nothing about what visitors cannot reach. Excluded from the sweep entirely.
SKIP_HOSTS = {"atc-staging.archive.aero"}

SELF_PROBE = (
    re.compile(r"/\.well-known/"),           # top-100 inventory replay
    re.compile(r"/wp-sitemap-bogus-1\.xml$"),  # redirect_check's 404 guard
)
# Anchor these on a path SEGMENT, not on the string start: everything on
# archive.aero arrives with the /atc prefix already attached, so "^/\.env"
# silently matches nothing and .env probes land in the content bucket.
SCANNER = (
    re.compile(r"wlwmanifest\.xml$"),
    re.compile(r"xmlrpc\.php$"),
    re.compile(r"wp-login\.php$"),
    re.compile(r"/wp-admin(/|$)"),
    re.compile(r"/\.env(\.|$)"),
    re.compile(r"/\.git(/|$)"),
    re.compile(r"/(vendor|autoload)\.php$"),
    re.compile(r"/administrator/"),          # Joomla
    re.compile(r"/media/system/js/"),        # Joomla
    re.compile(r"/(config|backup|dump)\.(bak|old|sql|zip)$"),
    re.compile(r"/language/en-GB/"),         # Joomla
    # The site is 100% static — no PHP exists anywhere under /atc, so any .php
    # request is somebody probing, never a real miss.
    re.compile(r"\.php$"),
    # WordPress is gone; these shapes are scanners and stale crawlers, and the
    # ones that map to real content are already in the redirect map.
    re.compile(r"/wp-content(/|$)"),
    re.compile(r"/wp-includes(/|$)"),
    re.compile(r"/wp-sitemap[\w-]*\.xsl$"),
)

# A srcset attribute requested as though it were one URL — "img.jpg 480w,
# https://…" percent-encoded into a single path. Every such row is the same
# defect in one rewrite pass, not N separate missing files, so they are bucketed
# together and reported once with a representative sample.
SRCSET = re.compile(r"%20\d+w(,|%2C)|,%20https?:")
JUNK = (
    re.compile(r"/\.DS_Store$"),
    re.compile(r"/Thumbs\.db$"),
    re.compile(r"^/favicon\.ico$"),
    re.compile(r"~$"),
)
ASSET_EXT = {".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
             ".woff", ".woff2", ".ttf", ".eot", ".map", ".webp"}


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
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
        f"/analytics_engine/sql",
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
        sys.exit(f"SQL API HTTP {e.code}: {e.read().decode()[:400]}{hint}")


def classify(path, ref):
    if SRCSET.search(path):
        return "srcset"
    for rx in SELF_PROBE:
        if rx.search(path):
            return "self"
    for rx in SCANNER:
        if rx.search(path):
            return "scanner"
    for rx in JUNK:
        if rx.search(path):
            return "junk"
    ext = Path(urllib.parse.urlparse(path).path).suffix.lower()
    if ext in ASSET_EXT:
        return "asset"
    return "content"


def internal(ref):
    if not ref:
        return False
    host = urllib.parse.urlparse(ref).netloc
    return host in OUR_HOSTS


# Inverse of the canonical mapping, enough to ask the old site the only
# question that separates a migration regression from an inherited dead link:
# "did YOU serve this?". A 404 here too means the link was already broken
# before the move and is not migration debt — worth fixing one day, but not
# evidence the cutover lost anything.
OLD_FORMS = (
    ("/atc/history/", "/History/"),
    ("/atc/class-photos/", "/classphotos/"),
    ("/atc/media/", "/wp-content/uploads/"),
    ("/atc/library/", "/pdf/"),
    ("/atc/assets/themes/", "/wp-content/themes/"),
    ("/atc/assets/plugins/", "/wp-content/plugins/"),
    ("/atc/assets/lib/", "/wp-includes/"),
    ("/atc/topics/", "/category/"),
    ("/atc/", "/"),
)
ORIGIN_IP = "74.220.207.111"   # HostMonster, bypassing Incapsula


def old_path_forms(path):
    for new, old in OLD_FORMS:
        if path.startswith(new):
            return old + path[len(new):]
    return None


def probe_origin(path):
    """HEAD the pre-cutover origin directly. mod_security 406s a bare request,
    hence the explicit Accept."""
    url = ("https://www.atchistory.org"
           + urllib.parse.quote(path, safe="/:@!$&()*+,;=~-._?#"))
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
           "--resolve", f"www.atchistory.org:443:{ORIGIN_IP}",
           "-H", "Accept: */*", "--max-time", "20", url]
    try:
        return int(subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30).stdout or 0)
    except Exception:
        return 0


def probe(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "archive.aero-404-sweep"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--since", help="absolute UTC start, 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip re-probing candidates live")
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    token = api_token()
    if args.since:
        window = f"timestamp >= toDateTime('{args.since}')"
        label = f"since {args.since} UTC"
    else:
        window = f"timestamp > now() - INTERVAL '{args.hours}' HOUR"
        label = f"last {args.hours}h"

    rows = query(
        f"SELECT {FIELDS} FROM {DATASET} WHERE double1 = 404 AND {window} "
        f"GROUP BY host, path, ref ORDER BY n DESC LIMIT {ROW_LIMIT}", token
    ).get("data", [])
    if len(rows) >= ROW_LIMIT:
        print(f"! {len(rows)} rows — at the SQL row cap, narrow the window",
              file=sys.stderr)

    # Collapse referer variants: one row per (host, path), keeping every
    # distinct referer so an internally-linked miss stays visible.
    agg = {}
    for r in rows:
        if r["host"] in SKIP_HOSTS:
            continue
        key = (r["host"], r["path"])
        a = agg.setdefault(key, {"n": 0, "refs": set(), "first": r["first"],
                                 "last": r["last"]})
        a["n"] += int(r["n"])
        if r["ref"]:
            a["refs"].add(r["ref"])
        a["first"] = min(a["first"], r["first"])
        a["last"] = max(a["last"], r["last"])

    out = []
    for (host, path), a in agg.items():
        bucket = classify(path, a["refs"])
        out.append({
            "host": host, "path": path, "hits": a["n"], "bucket": bucket,
            "internal_ref": any(internal(x) for x in a["refs"]),
            "old_path": old_path_forms(path) or "",
            "old_status": "",
            "refs": " | ".join(sorted(a["refs"])[:3]),
            "first": a["first"], "last": a["last"], "recheck": "",
        })

    # Re-probe the rows worth acting on. A miss that now 200s was already fixed
    # (or was a deploy-window artifact) and must not be reported as outstanding.
    actionable = [r for r in out
                  if r["bucket"] == "content"
                  or (r["bucket"] == "asset" and r["internal_ref"])]
    srcset = [r for r in out if r["bucket"] == "srcset"]
    if not args.no_probe:
        for r in actionable:
            r["recheck"] = probe(f"https://{r['host']}"
                                 f"{urllib.parse.quote(r['path'], safe='/:@!$&()*+,;=~-._?#')}")
            if r["recheck"] == 404 and r["old_path"]:
                r["old_status"] = probe_origin(r["old_path"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"404_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else
                           ["host", "path", "hits", "bucket", "internal_ref",
                            "refs", "first", "last", "recheck"])
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: -r["hits"]))

    totals = {}
    for r in out:
        totals[r["bucket"]] = totals.get(r["bucket"], 0) + r["hits"]
    print(f"atc_logs 404 sweep — {label}")
    print(f"{sum(r['hits'] for r in out)} hits over {len(out)} distinct "
          f"(host, path); by bucket:")
    for b, n in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"   {b:9} {n:6}")

    still = [r for r in actionable if r["recheck"] in (0, 404, "")]
    fixed = [r for r in actionable if r["recheck"] not in (0, 404, "")]
    print(f"\nactionable: {len(actionable)} "
          f"({len(still)} still missing, {len(fixed)} now resolve)")
    # The split that matters: did the old site serve it?
    regress = [r for r in still if r["old_status"] == 200]
    inherit = [r for r in still if r["old_status"] not in (200, "")]
    unknown = [r for r in still if r["old_status"] == ""]
    print(f"  MIGRATION REGRESSIONS (old site served it, we do not): "
          f"{len(regress)}")
    for r in sorted(regress, key=lambda r: -r["hits"]):
        flag = "INTERNAL" if r["internal_ref"] else "        "
        print(f"  {r['hits']:>4} {flag} {r['bucket']:7} {r['host']}{r['path']}")
        print(f"       was: {r['old_path']}")
        if r["refs"]:
            print(f"       ref: {r['refs'][:120]}")
    print(f"  inherited dead links (404 on the old site too): {len(inherit)}")
    for r in sorted(inherit, key=lambda r: -r["hits"])[:10]:
        flag = "INTERNAL" if r["internal_ref"] else "        "
        print(f"  {r['hits']:>4} {flag} {r['bucket']:7} {r['host']}{r['path']}"
              f"  (old: {r['old_status']})")
    if len(inherit) > 10:
        print(f"       … {len(inherit) - 10} more in the CSV")
    if unknown:
        print(f"  unmapped to an old form, needs an eye: {len(unknown)}")
        for r in sorted(unknown, key=lambda r: -r["hits"])[:10]:
            print(f"  {r['hits']:>4} {r['bucket']:7} {r['host']}{r['path']}")
    if fixed:
        print("\nalready resolving (no action):")
        for r in sorted(fixed, key=lambda r: -r["hits"]):
            print(f"  {r['hits']:>4} {r['recheck']} {r['host']}{r['path']}")
    if srcset:
        print(f"\nsrcset defect: {sum(r['hits'] for r in srcset)} hits over "
              f"{len(srcset)} mangled URLs — ONE bug in the rewrite pass, not "
              f"{len(srcset)} missing files. Sample:")
        for r in sorted(srcset, key=lambda r: -r["hits"])[:3]:
            print(f"  {r['hits']:>4} {r['path'][:110]}")
    print(f"\nfull report: {csv_path}")


if __name__ == "__main__":
    main()
