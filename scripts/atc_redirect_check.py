#!/Users/ryanhemenway/venv/bin/python
"""Cutover-day redirect verification for atchistory.org -> archive.aero/atc
(worklist 08, workstream D — runs at the 13:00 step, after MODE=redirect).

For every inventory row, across the full matrix http/https x www/apex:
  * redirect rows: exactly ONE 301, Location == the target route_map.json
    promises, query string dropped, and the target then answers 200 with no
    further redirect (single-hop total). Junk / wp-internal rows tolerate a
    non-200 target (the worker 301s unknowns and lets serving 404 honestly).
  * 410 set: wp internals, /forum/*, secondary feeds, drop-map tombstones.
  * exceptions that must keep answering 200 on the OLD hostnames:
    /robots.txt (permissive + Sitemap line), the GSC verification file, and
    the six frozen /wp-sitemap*.xml snapshots. A never-frozen sitemap variant
    must 404 (not 500 — regression guard for the null-response crash).
  * ?p=/?page_id= shortlinks: every p_map id on the primary variant (plus a
    matrix sample), routed through the full table — dropped slugs and /home/
    must one-hop; ?s= searches 301 to /atc/.

The expected-target model is a faithful port of worker-atc/src/routes.js
routeOldPath()/canonicalOf() + the redirectOldHost() handler in index.js —
keep them in lockstep.

Modes:
  cutover day     atc_redirect_check.py                   (full matrix, ~75k reqs)
  quick smoke     atc_redirect_check.py --quick           (matrix for top 1000)
  logic only      atc_redirect_check.py --offline         (no network; model +
                                                           chain check, runnable
                                                           any day)
  rehearsal       printf 'MODE=redirect\n' > worker-atc/.dev.vars
                  cd worker-atc && npx wrangler dev --port 8787 \
                      --local-upstream www.atchistory.org
                  atc_redirect_check.py --hosts www.atchistory.org \
                      --resolve www.atchistory.org:127.0.0.1 --port 8787 \
                      --schemes http --tag rehearsal
                  Wrangler's dev proxy REWRITES the Host header (a --var/Host
                  trick does not work): one upstream host per dev instance,
                  rerun per host. rm worker-atc/.dev.vars afterwards. The
                  R2-backed exceptions (GSC file, frozen sitemaps) 404 in
                  local dev unless seeded first with
                  `wrangler r2 object put atc-site/<key> --file <f> --local`.

Target 404s are tolerated for rows whose OLD path was already dead on live
(bug-for-bug ghosts) — judged from the freshest parity_live_cache.jsonl, so
run the freeze-day parity pass before the cutover-day redirect pass.

Preflight: each matrix variant must already be serving from the new stack
(cf-ray header + the worker's exact robots.txt body). Incapsula markers mean
DNS/MODE has not flipped — abort rather than "verify" the old hosting.

Reports in worklists/data/atc/: redirect_report{tag}.csv,
redirect_failures{tag}.csv, redirect_summary{tag}.txt.
Exit 0 = every assertion green.
"""

import argparse
import csv
import json
import re
import socket
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import requests

# ---------------------------------------------------------------- constants

REPO = Path("/Users/ryanhemenway/archive.aero")
DATA = REPO / "worklists" / "data" / "atc"
INV = DATA / "url_inventory.csv"
ROUTE_MAP = REPO / "worker-atc" / "src" / "route_map.json"
P_MAP_F = REPO / "worker-atc" / "src" / "p_map.json"

NEW_ORIGIN = "https://archive.aero"
PREFIX = "/atc"
OLD_HOSTS = ["atchistory.org", "www.atchistory.org"]

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 atc-redirect-check")

# The worker's OLD_ROBOTS body (index.js) — the preflight sentinel.
WORKER_ROBOTS_LINE = b"Sitemap: https://www.atchistory.org/wp-sitemap.xml"

FROZEN_SITEMAPS = {
    "/wp-sitemap.xml",
    "/wp-sitemap-posts-post-1.xml",
    "/wp-sitemap-posts-page-1.xml",
    "/wp-sitemap-taxonomies-category-1.xml",
    "/wp-sitemap-taxonomies-post_tag-1.xml",
    "/wp-sitemap-users-1.xml",
}

# worker-atc/src/routes.js GONE_RE — port verbatim.
GONE_RE = re.compile(
    r"^/(wp-admin(/|$)|wp-login\.php|xmlrpc\.php|wp-cron\.php|wp-json(/|$)|forum(/|$))")
# Buckets that tolerate a non-200 redirect TARGET (absence is by design;
# the 301 itself is still asserted). Mirrors atc_parity_check.py policy.
JUNK_RE = re.compile(
    r"(^|/)(\.DS_Store|Thumbs\.db|error_log[^/]*|\.htaccess[^/]*)$|"
    r"\.(php\d?|phtml|pl|cgi)$|^/+(backup|tmp|cgi-bin)/|^/+ads\.txt$|"
    r"^/+readme\.html$|"
    r"^/+\.well-known(/|$)|^/+20\d\d/$|^/+blog/$|"
    r"^/+wp-sitemap[\w-]*\.xsl$|"
    r"^/+___|_vti_cnf/", re.I)
WP_INTERNAL_RE = re.compile(
    r"^/+wp-(includes|content/(plugins|themes))/|"
    r"^/+wp-content/(uploads/)?$")

ROUTES = json.loads(ROUTE_MAP.read_text())
P_MAP = json.loads(P_MAP_F.read_text())
RULES, ALIAS, DROP = ROUTES["rules"], ROUTES["alias"], ROUTES["drop"]

# Live statuses from the parity harness (read-only): a redirect TARGET that
# 404s is tolerated when the old path was already dead on live — the same
# worse-than-live policy the parity closure uses. GET entries win over HEAD.
LIVE_CACHE = DATA / "parity_live_cache.jsonl"
LIVE_STATUS = {}
if LIVE_CACHE.exists():
    for _line in LIVE_CACHE.open():
        try:
            _j = json.loads(_line)
        except Exception:
            continue
        if _j.get("method") == "GET" or _j["path"] not in LIVE_STATUS:
            LIVE_STATUS[_j["path"]] = _j["status"]

# ------------------------------------------------- routes.js port (model)


def canonical_of(inner):
    hit = ALIAS.get(inner)
    if hit:
        return hit
    alt = inner[:-1] if inner.endswith("/") else inner + "/"
    hit = ALIAS.get(alt)
    if hit:
        return hit
    for old, new in RULES:
        if inner.startswith(old):
            return new + inner[len(old):]
    return None


def route_old_path(path):
    alt = path[:-1] if path.endswith("/") else path + "/"
    dropped = DROP.get(path)
    if dropped is None:
        dropped = DROP.get(alt)
    if dropped == "410":
        return {"status": 410}
    if dropped:
        return {"to": dropped}
    if GONE_RE.match(path):
        return {"status": 410}
    # Site-feed serialization variants 301 to the frozen feed (routes.js lockstep).
    if re.match(r"^/feed/(atom|rdf|rss2?)/?$", path, re.I):
        return {"to": PREFIX + "/feed"}
    if path not in ("/feed/", "/feed") and re.search(r"/feed/?$", path):
        return {"status": 410}
    canon = canonical_of(path)
    if canon and canon != PREFIX + path:
        return {"to": canon}
    return None


def expected_old_host(path, query):
    """Contract for scheme://old-host{path}?{query}: ('200', kind) |
    ('404', kind) | ('410', None) | ('301', decoded_absolute_target)."""
    if path == "/robots.txt":
        return ("200", "robots")
    if path == "/google994516164ab8ab88.html":
        return ("200", "gsc-verification")
    if re.match(r"^/wp-sitemap[\w-]*\.xml$", path):
        if path in FROZEN_SITEMAPS:
            return ("200", "frozen-sitemap")
        return ("404", "sitemap-variant")
    q = parse_qs(query, keep_blank_values=True)
    pid = (q.get("p") or q.get("page_id") or [None])[0]
    if pid and pid in P_MAP:
        r = route_old_path(P_MAP[pid]) or {}
        if r.get("status") == 410:
            return ("410", None)
        return ("301", NEW_ORIGIN + (r.get("to") or PREFIX + P_MAP[pid]))
    if "s" in q:
        return ("301", NEW_ORIGIN + PREFIX + "/")
    r = route_old_path(path) or {}
    if r.get("status") == 410:
        return ("410", None)
    if r.get("to"):
        return ("301", NEW_ORIGIN + r["to"])
    return ("301", NEW_ORIGIN + PREFIX + path)


def bucket_of(path):
    if GONE_RE.match(path):
        return "gone"
    if JUNK_RE.search(path):
        return "junk"
    if WP_INTERNAL_RE.match(path):
        return "wp_internal"
    return "content"


# ------------------------------------------------------------- http layer

RESOLVE = {}
_real_getaddrinfo = socket.getaddrinfo


def _pin(host, *a, **kw):
    return _real_getaddrinfo(RESOLVE.get(host, host), *a, **kw)


socket.getaddrinfo = _pin


class Throttle:
    def __init__(self, rate):
        self.min_gap = 1.0 / rate if rate else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        if not self.min_gap:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.min_gap
        d = t - now
        if d > 0:
            time.sleep(d)


def make_session(pool, insecure):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.verify = not insecure
    ad = requests.adapters.HTTPAdapter(pool_connections=pool,
                                       pool_maxsize=pool, max_retries=2)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def req(sess, thr, url, want_body=False, timeout=30):
    """One request, redirects NOT followed — observing hops is the point."""
    thr.wait()
    try:
        r = sess.get(url, allow_redirects=False, timeout=timeout,
                     stream=not want_body)
        body = r.content if want_body else b""
        if not want_body:
            r.close()
        return {"status": r.status_code,
                "location": r.headers.get("location", ""),
                "headers": {k.lower(): v for k, v in r.headers.items()},
                "body": body, "err": ""}
    except Exception as e:
        return {"status": 0, "location": "", "headers": {}, "body": b"",
                "err": f"{type(e).__name__}:{e}"[:160]}


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="expected-target model + chain check only, no network")
    ap.add_argument("--quick", action="store_true",
                    help="full matrix only for the top 1000 rows by hits; "
                         "primary variant for the rest")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--match", help="only paths containing this substring")
    ap.add_argument("--hosts", default=",".join(OLD_HOSTS))
    ap.add_argument("--schemes", default="https,http")
    ap.add_argument("--resolve", action="append", default=[],
                    metavar="HOST:IP", help="pin HOST to IP (repeatable)")
    ap.add_argument("--port", type=int, help="connect port override "
                    "(wrangler dev rehearsal)")
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--rps", type=float, default=0)
    ap.add_argument("--skip-targets", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run despite a failed preflight")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = ("_" + args.tag) if args.tag else ""

    for spec in args.resolve:
        host, _, ip = spec.partition(":")
        RESOLVE[host] = ip
    rehearsal = bool(args.resolve or args.port)
    portsfx = f":{args.port}" if args.port else ""

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    variants = [(s, h) for h in hosts for s in schemes]
    primary = variants[0]

    # ------------------------------------------------------------ work list
    rows = list(csv.DictReader(INV.open()))
    if args.match:
        rows = [r for r in rows if args.match in r["old_path"]]
    if args.limit:
        rows = rows[: args.limit]
    rows.sort(key=lambda r: -int(r["hits"] or 0))

    def split_pq(old_path):
        p = "/" + old_path.lstrip("/")
        if "?" in p:
            p, _, q = p.partition("?")
            return p, q
        return p, ""

    checks = []   # (kind, scheme, host, path, query)
    seen = set()

    def add(kind, scheme, host, path, query=""):
        k = (scheme, host, path, query)
        if k in seen:
            return
        seen.add(k)
        checks.append((kind, scheme, host, path, query))

    matrix_rows = rows if not args.quick else rows[:1000]
    single_rows = [] if not args.quick else rows[1000:]
    for r in matrix_rows:
        p, q = split_pq(r["old_path"])
        for s, h in variants:
            add("inventory", s, h, p, q)
    for r in single_rows:
        p, q = split_pq(r["old_path"])
        add("inventory", *primary, p, q)

    # specials — asserted even if absent from the inventory
    SPECIAL_410 = ["/wp-admin/", "/wp-login.php", "/xmlrpc.php", "/wp-cron.php",
                   "/wp-json/", "/forum/", "/forum/index.php",
                   "/lewiston/feed/", "/category/uncategorized/feed/"]
    SPECIAL_200 = ["/robots.txt", "/google994516164ab8ab88.html",
                   *sorted(FROZEN_SITEMAPS)]
    for pth in SPECIAL_410 + SPECIAL_200 + ["/wp-sitemap-bogus-1.xml", "/",
                                            "/home/"]:
        for s, h in variants:
            add("special", s, h, pth)
    # shortlinks: full p_map on the primary variant, matrix for a sample
    pids = sorted(P_MAP, key=int)
    for pid in pids:
        add("qs", *primary, "/", f"p={pid}")
    for pid in pids[:: max(1, len(pids) // 8)]:
        for s, h in variants:
            add("qs", s, h, "/", f"p={pid}")
    for pid in ("156", "10225"):
        add("qs", *primary, "/", f"page_id={pid}")
    add("qs", *primary, "/", "s=checklist")

    # ------------------------------------------------------------- offline
    if args.offline:
        chains, c = [], Counter()
        for kind, s, h, p, q in checks:
            want, tgt = expected_old_host(p, q)
            c[want if want != "200" else f"200:{tgt}"] += 1
            if want == "301":
                tp = unquote(urlsplit(tgt).path)
                if tp.startswith(PREFIX):
                    inner = tp[len(PREFIX):] or "/"
                    onward = route_old_path(inner)
                    if onward is not None and onward.get("to") != tp:
                        chains.append((p, q, tp, onward))
        print(f"offline model over {len(checks)} checks "
              f"({len(matrix_rows)} matrix rows x {len(variants)} variants"
              f"{f' + {len(single_rows)} single' if single_rows else ''}"
              f" + specials/qs)")
        print(f"expected outcomes: {dict(c.most_common())}")
        print(f"would-chain targets: {len(chains)}")
        for p, q, tp, onward in chains[:20]:
            print(f"  {p}{'?' + q if q else ''} -> {tp} -> "
                  f"{onward.get('to') or onward.get('status')}")
        sys.exit(1 if chains else 0)

    # ------------------------------------------------------------ preflight
    thr = Throttle(args.rps)
    sess = make_session(args.workers + 8, args.insecure)

    print(f"preflight: worker robots.txt sentinel on {len(variants)} variants"
          + (" [rehearsal: cf-ray not required]" if rehearsal else ""))
    pf_bad = []
    for s, h in variants:
        r = req(sess, thr, f"{s}://{h}{portsfx}/robots.txt", want_body=True)
        incap = ("x-iinfo" in r["headers"] or b"Incapsula" in r["body"]
                 or b"_Incap_" in r["body"])
        ok = (r["status"] == 200 and WORKER_ROBOTS_LINE in r["body"]
              and not incap and (rehearsal or "cf-ray" in r["headers"]))
        state = ("OK" if ok else
                 "INCAPSULA/OLD-STACK" if incap else
                 f"status={r['status']} err={r['err']} "
                 f"cf-ray={'y' if 'cf-ray' in r['headers'] else 'n'} "
                 f"worker-body={'y' if WORKER_ROBOTS_LINE in r['body'] else 'n'}")
        print(f"  {s}://{h}{portsfx}  {state}")
        if not ok:
            pf_bad.append((s, h))
    if pf_bad and not args.force:
        print("\nPREFLIGHT FAILED — these variants are not serving the "
              "redirect worker yet (DNS orange-cloud? routes added? "
              "MODE=redirect deployed?). --force to run anyway.")
        sys.exit(2)

    # --------------------------------------------------------------- phase 1
    print(f"{len(checks)} redirect checks, workers={args.workers}")
    t0 = time.time()
    results = []
    lock = threading.Lock()
    done = Counter()

    def check_one(item):
        kind, s, h, path, query = item
        url = (f"{s}://{h}{portsfx}" + quote(path, safe="/%")
               + (f"?{query}" if query else ""))
        want, tgt = expected_old_host(path, query)
        want_body = want == "200"
        r = req(sess, thr, url, want_body=want_body)
        rec = {"url": url, "old_path": path, "kind": kind,
               "bucket": bucket_of(path),
               "expected": want if want != "301" else f"301 {tgt}",
               "status": r["status"], "location": r["location"],
               "target_status": "", "result": "PASS", "detail": ""}
        if r["err"]:
            rec["result"], rec["detail"] = "FAIL", "fetch-error:" + r["err"]
        elif want in ("200", "404", "410"):
            if r["status"] != int(want):
                rec["result"] = "FAIL"
                rec["detail"] = f"expected {want} got {r['status']}"
            elif want == "200" and tgt == "robots" \
                    and WORKER_ROBOTS_LINE not in r["body"]:
                rec["result"], rec["detail"] = "FAIL", "robots-body-mismatch"
            elif want == "200" and tgt == "frozen-sitemap" \
                    and b"<" not in r["body"][:200]:
                rec["result"], rec["detail"] = "FAIL", "sitemap-not-xml"
            else:
                rec["detail"] = tgt or ""
        else:  # 301
            if r["status"] != 301:
                rec["result"] = "FAIL"
                rec["detail"] = f"expected 301 got {r['status']}"
            else:
                sp = urlsplit(r["location"])
                got = f"{sp.scheme}://{sp.netloc}" + unquote(sp.path)
                if sp.query:
                    rec["result"] = "FAIL"
                    rec["detail"] = f"query survived: {r['location'][:120]}"
                elif got != tgt:
                    rec["result"] = "FAIL"
                    rec["detail"] = f"location {got[:120]} != {tgt[:120]}"
        with lock:
            results.append(rec)
            done[rec["result"]] += 1
            n = sum(done.values())
            if n % 2000 == 0:
                print(f"  {n}/{len(checks)} ({time.time() - t0:.0f}s) "
                      f"{dict(done)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(check_one, checks))

    # ----------------------------------------- phase 2: targets, deduplicated
    if not args.skip_targets:
        targets = {}
        for rec in results:
            if rec["result"] == "PASS" and rec["expected"].startswith("301 "):
                targets.setdefault(rec["expected"][4:], []).append(rec)
        print(f"target verification: {len(targets)} distinct 301 targets")
        tstat = {}

        def check_target(t):
            sp = urlsplit(t)
            r = req(sess, thr, f"{sp.scheme}://{sp.netloc}"
                    + quote(sp.path, safe="/%"))
            tstat[t] = (r["status"], r["location"])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(check_target, sorted(targets)))
        for t, recs in targets.items():
            st, loc = tstat.get(t, (0, ""))
            for rec in recs:
                rec["target_status"] = st
                if 300 <= st < 400:
                    rec["result"] = "FAIL"
                    rec["detail"] = f"target redirects again -> {loc[:100]}"
                elif st != 200 and rec["bucket"] == "content":
                    ls = LIVE_STATUS.get(rec["old_path"])
                    if st == 404 and ls is not None and ls >= 400:
                        rec["result"] = "NOTE"
                        rec["detail"] = f"target 404, dead-on-live ({ls})"
                    else:
                        rec["result"] = "FAIL"
                        rec["detail"] = f"target {st}"
                elif st != 200:
                    rec["result"] = "NOTE"
                    rec["detail"] = f"target {st} tolerated ({rec['bucket']})"

    # ------------------------------------------------------------- reports
    DATA.mkdir(parents=True, exist_ok=True)
    cols = ["url", "old_path", "kind", "bucket", "expected", "status",
            "location", "target_status", "result", "detail"]
    order = {"FAIL": 0, "NOTE": 1, "PASS": 2}
    results.sort(key=lambda r: (order.get(r["result"], 3), r["url"]))
    with (DATA / f"redirect_report{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(results)
    fails = [r for r in results if r["result"] == "FAIL"]
    with (DATA / f"redirect_failures{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(fails)

    counts = Counter(r["result"] for r in results)
    summary = [
        f"run {time.strftime('%Y-%m-%d %H:%M')} checks={len(results)} "
        f"variants={variants}" + (" REHEARSAL" if rehearsal else "")
        + (" quick" if args.quick else ""),
        f"results: {dict(counts)}",
        f"single-hop 301 + exact target asserted for every check; "
        f"targets verified: {'no' if args.skip_targets else 'yes'}",
        f"elapsed {time.time() - t0:.0f}s",
    ]
    (DATA / f"redirect_summary{tag}.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    if fails:
        print("\ntop failures:")
        for r in fails[:30]:
            print(f"  {r['url'][:90]}  {r['detail'][:90]}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
