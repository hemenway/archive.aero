#!/Users/ryanhemenway/venv/bin/python
"""Old-vs-new parity harness for the atchistory.org -> archive.aero/atc migration
(worklist 08, workstream D).

For every row of worklists/data/atc/url_inventory.csv:
  live  = https://www.atchistory.org<path>   (origin-direct, Incapsula bypassed)
  new   = staging or production serve host   (--base staging|prod)

Checks, class-aware:
  * text rows (wp_* pages, static .htm, feeds): status, <title> equality,
    length band (AdSense strip + host rewrite shift bytes legitimately)
  * binary rows: exact Content-Length live==new, new==local site/ file,
    R2 ETag == local md5 (single-part uploads)
  * generated listings (autoindex replacements): new 200 + marker; live status
    recorded only (Apache autoindex output is not comparable)
  * policy buckets that never gate: wp internals, excluded junk, secondary
    feeds, legacy ghosts (dead on live, redirect-mode targets), host metadata,
    and rows the canonical map re-points to a DIFFERENT resource on purpose
    (author-archive collapse, drop-map junk pages) — for those the contract is
    the canonical assertion (exact landing, <=1 hop) plus a 200 at the target,
    never live-content equality
  * ASSET CLOSURE ("rendered appearance"): every same-site subresource
    referenced by every 200 HTML page — css/js/img/srcset/media/favicon/
    <body background>/inline url() — plus url()/@import refs inside every
    reachable stylesheet, recursively, must answer 200. This is the automated
    form of the 2026-08-09 asset-gap incident check. Broken refs are
    live-verified at the OLD URL space: the reported asset URL is canonical
    (§A3), so it is reverse-mapped through route_map.json (mirroring
    keyCandidates() in worker-atc/src/routes.js) before probing live — the
    §A4 false-negative fix (2026-08-20); the probed path is reported in the
    live_path column.

Live results are cached (parity_live_cache.jsonl) so fix-loop reruns only
re-hit the new stack; --refresh-live busts the cache. The old box is a shared
host: live traffic is throttled + capped in concurrency, and the run aborts if
Incapsula challenge pages ever appear (origin-direct should never see them).

Outputs in worklists/data/atc/:
  parity_report.csv, parity_failures.csv, asset_closure_failures.csv,
  parity_summary.txt

Exit code: 0 = green (no FAIL rows, closure clean), 1 = failures present.
"""

import argparse
import csv
import hashlib
import json
import re
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

# ---------------------------------------------------------------- constants

ORIGIN_IP = "74.220.207.111"  # HostMonster box; SNI/Host stay atchistory.org
LIVE_HOST = "www.atchistory.org"
BASES = {
    "staging": ("atc-staging.archive.aero", ""),
    "prod": ("archive.aero", "/atc"),
}
NEW_HOSTS_OK = {"atc-staging.archive.aero", "archive.aero"}

DATA = Path("/Users/ryanhemenway/archive.aero/worklists/data/atc")
INV = DATA / "url_inventory.csv"
SITE = Path("/Volumes/projects/atchistory_build/site")
LEGACY_MAP = Path("/Users/ryanhemenway/archive.aero/worker-atc/src/legacy_map.json")
ROUTE_MAP = Path("/Users/ryanhemenway/archive.aero/worker-atc/src/route_map.json")
LIVE_CACHE = DATA / "parity_live_cache.jsonl"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
LISTING_MARKER = b"<!-- generated directory listing"

TEXT_EXT = {".htm", ".html", ".shtml", ".css", ".js", ".txt", ".xml"}
GONE_RE = re.compile(
    r"^/+(wp-admin(/|$)|wp-login\.php|xmlrpc\.php|wp-cron\.php|wp-json(/|$)|forum(/|$))")
JUNK_RE = re.compile(
    r"(^|/)(\.DS_Store|Thumbs\.db|error_log[^/]*|\.htaccess[^/]*)$|"
    r"\.(php\d?|phtml|pl|cgi)$|^/+(backup|tmp|cgi-bin)/|^/+ads\.txt$|"
    r"^/+readme\.html$|"  # WP version-disclosure readme — never migrated
    # log-discovered bot probes (2026-08-30 freeze tail): ACME dirs, WP
    # year-archive guesses (§A3: no date partitioning), /BLOG/ case probes
    r"^/+\.well-known(/|$)|^/+20\d\d/$|^/+blog/$|"
    r"^/+___|_vti_cnf/", re.I)
WP_INTERNAL_RE = re.compile(
    r"^/+wp-(includes|content/(plugins|themes))/|"
    r"^/+wp-content/(uploads/)?$")  # bare autoindex dirs, log-discovered
HOST_META_RE = re.compile(r"^/+(robots\.txt|wp-sitemap[\w-]*\.(xml|xsl))$|wlwmanifest\.xml$")

TITLE_RE = re.compile(rb"<title[^>]*>(.*?)</title>", re.I | re.S)

# subresource attributes in HTML (FrontPage-era background= included)
ASSET_ATTR_RE = re.compile(
    rb"""<(?:link|script|img|source|video|audio|embed|input|frame|iframe|td|
         body|table|tr)\b[^>]*?\b(?:src|href|background|poster|data)\s*=\s*
         ("[^"]*"|'[^']*'|[^\s>]+)""", re.I | re.S | re.X)
LINK_REL_RE = re.compile(rb"<link\b[^>]*>", re.I | re.S)
REL_RE = re.compile(rb"""\brel\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.I)
SRCSET_RE = re.compile(rb"""\bsrcset\s*=\s*("[^"]*"|'[^']*')""", re.I | re.S)
CSS_URL_RE = re.compile(rb"""url\(\s*(?:"([^"]*)"|'([^']*)'|([^)'"\s]+))\s*\)""", re.I)
CSS_IMPORT_RE = re.compile(rb"""@import\s+(?:"([^"]*)"|'([^']*)')""", re.I)
STYLE_BLOCK_RE = re.compile(rb"<style\b[^>]*>(.*?)</style>", re.I | re.S)

# ------------------------------------------------------------- origin pin

_real_getaddrinfo = socket.getaddrinfo


def _pin(host, *a, **kw):
    if host in ("www.atchistory.org", "atchistory.org"):
        host = ORIGIN_IP
    return _real_getaddrinfo(host, *a, **kw)


socket.getaddrinfo = _pin

# ------------------------------------------------------------- http layer


def make_session(pool):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    ad = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool,
                                       max_retries=2)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


class Throttle:
    """Token-ish limiter: at most `rate` requests/second across all threads."""

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


def looks_challenged(status, body):
    return status == 200 and len(body) < 4000 and (
        b"Incapsula" in body or b"_Incap_" in body)


def norm_title(raw):
    if raw is None:
        return None
    t = raw.decode("latin-1", "replace")
    return re.sub(r"\s+", " ", t).strip()


def extract_title(body):
    m = TITLE_RE.search(body[:65536])
    return norm_title(m.group(1)) if m else ""


def fetch(session, throttle, url, method="GET", max_hops=4, want_body=True,
          allowed_hosts=None, timeout=40, headers=None):
    """Follow same-site redirects manually; return dict with final result."""
    hops = []
    body = b""
    for _ in range(max_hops + 1):
        throttle.wait()
        try:
            r = session.request(method, url, allow_redirects=False,
                                timeout=timeout, stream=not want_body,
                                headers=headers)
            status = r.status_code
            if status in (301, 302, 303, 307, 308):
                loc = r.headers.get("location", "")
                r.close()
                nxt = urljoin(url, loc)
                host = urlsplit(nxt).hostname or ""
                hops.append(loc)
                if allowed_hosts and host not in allowed_hosts:
                    return {"status": status, "len": 0, "ctype": "",
                            "title": "", "hops": hops, "url": url,
                            "err": f"offsite-redirect:{loc[:120]}"}
                url = nxt
                continue
            ctype = (r.headers.get("content-type") or "").lower()
            clen = r.headers.get("content-length")
            etag = (r.headers.get("etag") or "").strip('W/"')
            crange = r.headers.get("content-range") or ""
            if want_body and method == "GET":
                body = r.content
                length = len(body)
            else:
                length = int(clen) if clen is not None else -1
                r.close()
            # a 0-0 range probe knows the real size from Content-Range
            m = re.search(r"bytes \d+-\d+/(\d+)", crange)
            if m:
                length = int(m.group(1))
                if status == 206:
                    status = 200
            return {"status": status, "len": length, "ctype": ctype,
                    "title": extract_title(body) if body else "",
                    "body": body if want_body else b"", "etag": etag,
                    "hops": hops, "url": url, "err": ""}
        except Exception as e:
            return {"status": 0, "len": -1, "ctype": "", "title": "",
                    "hops": hops, "url": url, "err": f"{type(e).__name__}:{e}"[:160]}
    return {"status": -1, "len": -1, "ctype": "", "title": "", "hops": hops,
            "url": url, "err": "too-many-redirects"}


# ------------------------------------------------------------ classification


def classify(path, cls, legacy):
    p = "/" + path.lstrip("/")
    if p == "/":
        return "homepage"
    if GONE_RE.match(p):
        return "gone"
    if JUNK_RE.search(p):
        return "junk"
    if WP_INTERNAL_RE.match(p):
        return "wp_internal"
    if HOST_META_RE.search(p):
        return "host_meta"
    if p == "/feed/":
        return "feed_root"
    if re.match(r"^/+feed/(atom|rdf|rss2?)/?$", p, re.I):
        return "feed_variant"    # site-feed serialization -> 301 to /atc/feed
    if re.search(r"/feed/?$", p):
        return "feed_secondary"
    if p in legacy:
        return "legacy_ghost"
    if re.match(r"^/+author(/|$)", p):
        return "author_collapse"   # single-author site; §A3 folds these away
    ext = Path(unquote(p).split("?")[0]).suffix.lower()
    if p.endswith("/"):
        return "dirpage"          # WP slug dir or static dir (listing/index)
    if ext in TEXT_EXT:
        return "text"
    if ext == "":
        return "extensionless"    # WP slug w/o slash (log-derived)
    return "binary"


def local_file(path):
    """site/ file that should back this URL (None if n/a)."""
    rel = unquote(path).lstrip("/")
    if not rel or rel.endswith("/"):
        for idx in ("index.html", "index.htm", "Default.htm"):
            f = SITE / rel / idx
            if f.is_file():
                return f
        return None
    f = SITE / rel
    if f.is_file():
        return f
    if "." not in rel.rsplit("/", 1)[-1]:
        for idx in ("index.html", "index.htm", "Default.htm"):
            f2 = SITE / rel / idx
            if f2.is_file():
                return f2
    return None


# --------------------------------------------------------------- asset scan


def _dequote(b):
    b = b.strip()
    if len(b) >= 2 and b[:1] in (b'"', b"'") and b[-1:] == b[:1]:
        return b[1:-1]
    return b


def html_assets(base_url, body):
    """Same-site subresource URLs referenced by an HTML page."""
    out = set()

    def add(raw):
        u = raw.decode("latin-1", "replace").strip()
        if (not u or u.startswith(("#", "data:", "mailto:", "javascript:",
                                   "about:", "tel:"))):
            return
        absu = urljoin(base_url, u)
        sp = urlsplit(absu)
        if sp.scheme not in ("http", "https"):
            return
        if sp.hostname not in NEW_HOSTS_OK:
            return
        # main-site chrome (/, /about.html…) is GH Pages, not this worker
        if sp.hostname == "archive.aero" and not sp.path.startswith("/atc/"):
            return
        out.add(sp.scheme + "://" + sp.netloc + sp.path)

    for m in ASSET_ATTR_RE.finditer(body):
        tag = m.group(0)[:200].lower()
        val = _dequote(m.group(1))
        if tag.startswith(b"<link"):
            rel = REL_RE.search(m.group(0))
            relv = _dequote(rel.group(1)).lower() if rel else b""
            if not any(k in relv for k in
                       (b"stylesheet", b"icon", b"preload", b"apple-touch")):
                continue
        if tag.startswith((b"<a", b"<area")):
            continue
        add(val)
    for m in SRCSET_RE.finditer(body):
        for cand in _dequote(m.group(1)).split(b","):
            part = cand.strip().split()
            if part:
                add(part[0])
    for m in STYLE_BLOCK_RE.finditer(body):
        for mm in CSS_URL_RE.finditer(m.group(1)):
            add(next(g for g in mm.groups() if g is not None))
    return out


def css_assets(base_url, body):
    out = set()
    for rx in (CSS_URL_RE, CSS_IMPORT_RE):
        for m in rx.finditer(body):
            raw = next((g for g in m.groups() if g), b"")
            u = raw.decode("latin-1", "replace").strip()
            if not u or u.startswith(("data:", "#")):
                continue
            absu = urljoin(base_url, u.split("?")[0].split("#")[0])
            sp = urlsplit(absu)
            if sp.hostname in NEW_HOSTS_OK and sp.scheme in ("http", "https"):
                if sp.hostname == "archive.aero" and not sp.path.startswith("/atc/"):
                    continue
                out.add(sp.scheme + "://" + sp.netloc + sp.path)
    return out


# ------------------------------------------------- canonical -> old reverse


def live_probe_paths(routes, new_path):
    """Live-site (old URL space) paths to probe for a new-stack asset path.

    Closure assets are reported at their NEW-stack URL, which lives in the
    §A3 canonical URI space; the live site only answers at the OLD paths
    (R2 keys are the unchanged old paths). Mirrors keyCandidates() in
    worker-atc/src/routes.js — keep the two in lockstep. Returns candidate
    live paths, most likely first; a path the canonical scheme never renamed
    comes back unchanged. (§A4 fix: probing live at the canonical path made
    every renamed-prefix asset — /assets/*, /media/* — look live-broken-too.)
    """
    idx = ("index.html", "index.htm", "Default.htm")
    canon = new_path if new_path == "/atc" or new_path.startswith("/atc/") \
        else "/atc" + new_path
    exact = routes["key"].get(canon)
    if exact is not None:
        if exact.endswith("/"):  # dir-style key: live serves the dir URL
            return ["/" + exact] + ["/" + exact + n for n in idx]
        return ["/" + exact.lstrip("/")]
    inner = canon[len("/atc"):] or "/"
    for old, new in routes["rules"]:
        cn = new[len("/atc"):]
        if inner.startswith(cn):
            inner = old + inner[len(cn):]
            break
    base = "" if inner == "/" else inner.strip("/")
    if base == "":
        return ["/"]
    if inner.endswith("/"):
        return ["/" + base + "/"]
    if re.search(r"\.[a-z0-9]{1,5}$", base, re.I):
        return ["/" + base]
    return ["/" + base, "/" + base + "/"]


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=("staging", "prod"), default="staging")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--match", help="only paths containing this substring")
    ap.add_argument("--refresh-live", action="store_true")
    ap.add_argument("--skip-live", action="store_true",
                    help="new-stack + closure checks only")
    ap.add_argument("--skip-assets", action="store_true")
    ap.add_argument("--no-md5", action="store_true")
    ap.add_argument("--live-workers", type=int, default=6)
    ap.add_argument("--live-rps", type=float, default=8.0)
    ap.add_argument("--new-workers", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for report filenames")
    args = ap.parse_args()
    tag = ("_" + args.tag) if args.tag else ""

    new_host, new_prefix = BASES[args.base]
    legacy = json.loads(LEGACY_MAP.read_text()) if LEGACY_MAP.exists() else {}

    rows = list(csv.DictReader(INV.open()))
    if args.match:
        rows = [r for r in rows if args.match in r["old_path"]]
    if args.limit:
        rows = rows[: args.limit]

    # live cache -----------------------------------------------------------
    cache = {}
    if LIVE_CACHE.exists() and not args.refresh_live:
        for line in LIVE_CACHE.open():
            try:
                j = json.loads(line)
                # key must mirror get_live()'s "<METHOD> <path>" or every
                # rerun silently re-crawls the live site
                cache[j.get("method", "GET") + " " + j["path"]] = j
            except Exception:
                pass
        print(f"live cache: {len(cache)} entries loaded")
    cache_f = LIVE_CACHE.open("a")
    cache_lock = threading.Lock()

    live_sess = make_session(args.live_workers)
    new_sess = make_session(args.new_workers + 8)
    live_thr = Throttle(args.live_rps)
    new_thr = Throttle(0)

    challenged = Counter()

    def live_url(path):
        return "https://" + LIVE_HOST + quote(path, safe="/%")

    def new_url(path):
        return "https://" + new_host + new_prefix + quote(path, safe="/%")

    def get_live(path, method):
        key = method + " " + path
        if key in cache:
            return cache[key]
        r = fetch(live_sess, live_thr, live_url(path), method=method,
                  want_body=(method == "GET"),
                  allowed_hosts={LIVE_HOST, "atchistory.org"})
        # mod_security 406s HEAD for some extensions (fonts) — size-probe via
        # a 1-byte range GET instead
        if method == "HEAD" and r["status"] in (403, 405, 406):
            r = fetch(live_sess, live_thr, live_url(path), method="GET",
                      want_body=False, allowed_hosts={LIVE_HOST, "atchistory.org"},
                      headers={"Range": "bytes=0-0", "Accept": "*/*"})
        if method == "GET" and looks_challenged(r["status"], r.get("body", b"")):
            challenged["n"] += 1
            r["err"] = "incapsula_challenge"
            if challenged["n"] > 5:
                print("ABORT: Incapsula challenges appearing origin-direct — "
                      "stopping live traffic.", file=sys.stderr)
                raise SystemExit(3)
        rec = {"path": path, "method": method, "status": r["status"],
               "len": r["len"], "ctype": r["ctype"], "title": r.get("title", ""),
               "hops": len(r["hops"]), "err": r["err"], "ts": int(time.time())}
        with cache_lock:
            cache[key] = rec
            cache_f.write(json.dumps(rec) + "\n")
        return rec

    # ------------------------------------------------------------ row check
    results = []
    res_lock = threading.Lock()
    css_seen = {}       # asset closure: fetched stylesheets
    asset_refs = defaultdict(set)   # asset url -> sample referrers

    def note_assets(page_url, body, ctype):
        if args.skip_assets:
            return
        if "html" in ctype:
            found = html_assets(page_url, body)
        elif "css" in ctype:
            found = css_assets(page_url, body)
        else:
            return
        for a in found:
            if len(asset_refs[a]) < 3:
                asset_refs[a].add(page_url)

    # ---- canonical URI policy (worklist 08 §A3)
    # Every old path must reach its canonical URI in at most ONE hop, and land
    # on exactly the address the map promises. A chain or a drift here is
    # invisible to content parity — the page still renders — so it needs its
    # own assertion.
    ROUTES = json.loads(ROUTE_MAP.read_text())
    canon_problems = []

    def expected_canonical(path):
        """Same precedence as routeOldPath() in worker-atc/src/routes.js."""
        alt = path[:-1] if path.endswith("/") else path + "/"
        drop = ROUTES["drop"].get(path, ROUTES["drop"].get(alt))
        if drop:
            return None if drop == "410" else drop
        hit = ROUTES["alias"].get(path, ROUTES["alias"].get(alt))
        if hit is not None:
            return hit
        # Site-feed serialization variants 301 to the frozen feed
        # (routes.js lockstep, 2026-08-31).
        if re.match(r"^/+feed/(atom|rdf|rss2?)/?$", path, re.I):
            return "/atc/feed"
        for old, new in ROUTES["rules"]:
            if path.startswith(old):
                return new + path[len(old):]
        return "/atc" + path

    def check_canonical(rec, path, n):
        want = expected_canonical(path)
        if want is None or n["status"] in (0, -1):
            return
        got = unquote(urlsplit(n["url"]).path)
        if new_prefix == "":                 # staging serves the site at root
            want_path = want[len("/atc"):] or "/" if want.startswith("/atc") else want
        else:
            want_path = want
        if len(n["hops"]) > 1:
            rec["canon"] = f"CHAIN:{len(n['hops'])} hops -> {got}"
            canon_problems.append((path, rec["canon"]))
        elif got != want_path:
            rec["canon"] = f"DRIFT: landed {got}, map says {want_path}"
            canon_problems.append((path, rec["canon"]))

    def check_row(row):
        path = "/" + row["old_path"].lstrip("/")
        bucket = classify(path, row["class"], legacy)
        rec = {"old_path": path, "class": row["class"], "bucket": bucket,
               "hits": row["hits"], "live_status": "", "live_len": "",
               "new_status": "", "new_len": "", "new_hops": 0,
               "result": "PASS", "detail": "", "canon": ""}

        if bucket in ("gone", "junk"):
            rec["result"] = "SKIP"
            rec["detail"] = "policy:" + bucket
            return rec

        text_like = bucket in ("homepage", "feed_root", "dirpage", "text",
                               "extensionless", "host_meta")
        n = fetch(new_sess, new_thr, new_url(path),
                  method="GET" if text_like else "HEAD",
                  want_body=text_like, allowed_hosts=NEW_HOSTS_OK)
        # worker HEAD on binaries returns no CL for plain 200 R2 gets; retry GET
        if not text_like and n["status"] == 200 and n["len"] < 0:
            n = fetch(new_sess, new_thr, new_url(path), method="GET",
                      want_body=False, allowed_hosts=NEW_HOSTS_OK)
        rec["new_status"] = n["status"]
        rec["new_len"] = n["len"]
        rec["new_hops"] = len(n["hops"])
        check_canonical(rec, path, n)
        body = n.get("body", b"")
        if body:
            note_assets(n["url"], body, n["ctype"])

        is_listing = body.startswith(LISTING_MARKER)

        if bucket == "wp_internal":
            rec["result"] = "INFO"
            rec["detail"] = f"new={n['status']} (closure decides)"
            return rec
        if bucket == "host_meta":
            rec["result"] = "SKIP"
            rec["detail"] = f"policy:host_meta new={n['status']}"
            return rec
        if bucket == "feed_secondary":
            rec["result"] = "SKIP"
            rec["detail"] = "policy:410-at-cutover"
            return rec
        if bucket == "feed_variant":
            rec["result"] = "SKIP"
            rec["detail"] = "policy:301-to-frozen-feed-at-cutover"
            return rec
        if bucket == "legacy_ghost":
            rec["result"] = "SKIP"
            rec["detail"] = f"legacy-redirect-target new={n['status']}"
            return rec
        if bucket == "homepage":
            rec["result"] = "PASS" if n["status"] == 200 else "FAIL"
            rec["detail"] = "landing page replaces WP home (no live compare)"
            return rec

        # §A3 re-points: the map deliberately sends these to a different
        # resource (author collapse, drop-map junk -> home). check_canonical
        # already pinned WHERE they land; the only remaining requirement is
        # that the landing exists.
        alt = path[:-1] if path.endswith("/") else path + "/"
        dropped = ROUTES["drop"].get(path) or ROUTES["drop"].get(alt)
        if bucket == "author_collapse" or (dropped and dropped != "410"):
            rec["result"] = "SKIP" if n["status"] == 200 else "FAIL"
            rec["detail"] = (f"policy:re-pointed->{dropped or 'archive'} "
                             f"new={n['status']}")
            return rec

        live = None
        if not args.skip_live:
            live = get_live(path, "GET" if text_like or is_listing else "HEAD")
            rec["live_status"] = live["status"]
            rec["live_len"] = live["len"]

        # ---- verdict
        if n["err"] and n["status"] == 0:
            rec["result"] = "FAIL"
            rec["detail"] = "new-fetch-error:" + n["err"]
            return rec

        if live is None:
            if n["status"] != 200:
                lf = local_file(path)
                if lf is None:
                    rec["result"] = "NOTE"
                    rec["detail"] = "absent-locally-too"
                else:
                    rec["result"] = "FAIL"
                    rec["detail"] = f"new={n['status']} but local file exists"
            return rec

        ls, ns = live["status"], n["status"]
        if ls in (0,) and live["err"]:
            rec["result"] = "WARN"
            rec["detail"] = "live-fetch-error:" + live["err"]
            return rec
        if ls >= 400:
            if ns == 200:
                rec["result"] = "NOTE"
                rec["detail"] = "dead-on-live; recovered-from-backup"
            else:
                rec["result"] = "NOTE"
                rec["detail"] = f"dead-on-live ({ls}); new={ns}"
            return rec
        if ns != 200:
            rec["result"] = "FAIL"
            rec["detail"] = f"live={ls} but new={ns}"
            return rec

        if is_listing:
            rec["result"] = "PASS"
            rec["detail"] = "generated-listing"
            return rec

        if text_like or "html" in n["ctype"] or "xml" in n["ctype"]:
            lt, nt = live.get("title", ""), n.get("title", "")
            ll, nl = live["len"], n["len"]
            if lt != nt:
                # windows-1252 vs entity variance is possible; only exact
                # match passes, everything else is at least WARN
                rec["result"] = "FAIL" if lt and nt and lt != nt else "WARN"
                rec["detail"] = f"title live={lt[:60]!r} new={nt[:60]!r}"
                return rec
            if ll > 0:
                delta = abs(nl - ll)
                if delta > max(6144, 0.20 * ll):
                    rec["result"] = "FAIL"
                    rec["detail"] = f"len live={ll} new={nl}"
                    return rec
                if delta > max(4096, 0.15 * ll):
                    rec["result"] = "WARN"
                    rec["detail"] = f"len-drift live={ll} new={nl}"
                    return rec
            rec["detail"] = "text-ok"
            return rec

        # binary: exact size, local backing, md5 when cheap
        lf = local_file(path)
        if live["len"] >= 0 and n["len"] >= 0 and live["len"] != n["len"]:
            rec["result"] = "FAIL"
            rec["detail"] = f"size live={live['len']} new={n['len']}"
            return rec
        if lf is not None and n["len"] >= 0 and lf.stat().st_size != n["len"]:
            rec["result"] = "FAIL"
            rec["detail"] = f"size local={lf.stat().st_size} new={n['len']}"
            return rec
        if (not args.no_md5 and lf is not None and n.get("etag")
                and "-" not in n["etag"] and lf.stat().st_size <= 64 << 20):
            md5 = hashlib.md5(lf.read_bytes()).hexdigest()
            if md5 != n["etag"]:
                rec["result"] = "FAIL"
                rec["detail"] = f"md5 local={md5[:12]} r2etag={n['etag'][:12]}"
                return rec
        rec["detail"] = "binary-ok"
        return rec

    print(f"{len(rows)} inventory rows -> base={args.base} "
          f"(new host {new_host}{new_prefix or '/'})"
          + (" [skip-live]" if args.skip_live else ""))
    t0 = time.time()
    done = Counter()

    def run(row):
        try:
            rec = check_row(row)
        except SystemExit:
            raise
        except Exception as e:
            rec = {"old_path": row["old_path"], "class": row["class"],
                   "bucket": "?", "hits": row["hits"], "live_status": "",
                   "live_len": "", "new_status": "", "new_len": "",
                   "new_hops": "", "result": "FAIL",
                   "detail": f"harness-error:{type(e).__name__}:{e}"[:200]}
        with res_lock:
            results.append(rec)
            done[rec["result"]] += 1
            n = sum(done.values())
            if n % 500 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(rows)} ({el:.0f}s) {dict(done)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.new_workers) as ex:
        list(ex.map(run, rows))

    # ---------------------------------------------------- asset closure pass
    closure_fail = []
    if not args.skip_assets:
        # recursively pull css url()/@import refs of reachable stylesheets
        todo = [u for u in asset_refs if u.split("?")[0].lower().endswith(".css")]
        seen_css = set()
        while todo:
            u = todo.pop()
            if u in seen_css:
                continue
            seen_css.add(u)
            r = fetch(new_sess, new_thr, u, want_body=True,
                      allowed_hosts=NEW_HOSTS_OK)
            css_seen[u] = r["status"]
            if r["status"] == 200 and r.get("body"):
                for a in css_assets(u, r["body"]):
                    if len(asset_refs[a]) < 3:
                        asset_refs[a].add(u)
                    if a.split("?")[0].lower().endswith(".css") and a not in seen_css:
                        todo.append(a)

        print(f"asset closure: {len(asset_refs)} unique same-site refs "
              f"from served pages ({len(seen_css)} stylesheets recursed)")
        a_lock = threading.Lock()

        def check_asset(a):
            r = fetch(new_sess, new_thr, a, method="HEAD", want_body=False,
                      allowed_hosts=NEW_HOSTS_OK)
            if r["status"] != 200:
                r2 = fetch(new_sess, new_thr, a, method="GET", want_body=False,
                           allowed_hosts=NEW_HOSTS_OK)
                if r2["status"] != 200:
                    with a_lock:
                        closure_fail.append(
                            {"asset": a, "status": r2["status"],
                             "live_status": "", "live_path": "", "verdict": "",
                             "referrers": " | ".join(sorted(asset_refs[a]))})

        with ThreadPoolExecutor(max_workers=args.new_workers) as ex:
            list(ex.map(check_asset, sorted(asset_refs)))

        # broken on new: is it broken on live too? (worse-than-live = FAIL,
        # matching live breakage = NOTE). GET, not HEAD — mod_security quirk.
        # Runs even under --skip-live: it's only ever a handful of requests
        # and an unverified break is a useless verdict. The asset URL is
        # canonical-space; live must be probed at the OLD path (§A4 fix) —
        # every candidate has to fail for a live-broken-too verdict.
        for cf in closure_fail:
            p = unquote(urlsplit(cf["asset"]).path)
            lr = {"status": 0}
            for lp in live_probe_paths(ROUTES, p):
                lr = fetch(live_sess, live_thr, live_url(lp), method="GET",
                           want_body=False,
                           allowed_hosts={LIVE_HOST, "atchistory.org"},
                           headers={"Range": "bytes=0-0", "Accept": "*/*"})
                cf["live_path"] = lp
                if lr["status"] == 200:
                    break
            cf["live_status"] = lr["status"]
            cf["verdict"] = ("FAIL-recoverable" if lr["status"] == 200
                             else "live-broken-too")
        closure_fail.sort(key=lambda c: (not c["verdict"].startswith("FAIL"),
                                         c["asset"]))

    # ------------------------------------------------------------- reports
    DATA.mkdir(parents=True, exist_ok=True)
    cols = ["old_path", "class", "bucket", "hits", "live_status", "live_len",
            "new_status", "new_len", "new_hops", "result", "detail", "canon"]
    with (DATA / f"parity_report{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["result"] != "FAIL",
                                                -int(r["hits"] or 0))):
            w.writerow(r)
    fails = [r for r in results if r["result"] == "FAIL"]
    with (DATA / f"parity_failures{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(sorted(fails, key=lambda r: -int(r["hits"] or 0)))
    with (DATA / f"asset_closure_failures{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["asset", "status", "live_status", "live_path",
                               "verdict", "referrers"])
        w.writeheader()
        w.writerows(closure_fail)

    closure_hard = [c for c in closure_fail if c["verdict"].startswith("FAIL")]
    counts = Counter(r["result"] for r in results)
    buckets = Counter(r["bucket"] for r in results)
    summary = [
        f"run {time.strftime('%Y-%m-%d %H:%M')} base={args.base} rows={len(rows)}"
        + (" skip-live" if args.skip_live else ""),
        f"results: {dict(counts)}",
        f"buckets: {dict(buckets)}",
        f"canonical URIs: {len(canon_problems)} chains/drifts "
        f"(every old path must reach its canonical in <=1 hop)",
        f"asset closure: {len(asset_refs)} refs, {len(closure_fail)} broken "
        f"({len(closure_hard)} recoverable=FAIL, "
        f"{len(closure_fail) - len(closure_hard)} live-broken-too)",
        f"elapsed {time.time() - t0:.0f}s",
    ]
    (DATA / f"parity_summary{tag}.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    if fails:
        print("\ntop failures:")
        for r in sorted(fails, key=lambda r: -int(r["hits"] or 0))[:25]:
            print(f"  {r['hits']:>7} {r['old_path'][:80]}  {r['detail'][:90]}")
    if canon_problems:
        print(f"\ncanonical URI problems ({len(canon_problems)}):")
        for path, why in canon_problems[:25]:
            print(f"  {path[:80]}  {why[:90]}")
    if closure_fail:
        print("\nbroken assets:")
        for r in closure_fail[:25]:
            print(f"  new={r['status']:>3} live={r['live_status']} "
                  f"[{r['verdict']}] {r['asset'][:80]} "
                  f"(live probe {r['live_path']})")
    sys.exit(1 if (fails or closure_hard or canon_problems) else 0)


if __name__ == "__main__":
    main()
