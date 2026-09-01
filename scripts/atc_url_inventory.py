#!/Users/ryanhemenway/venv/bin/python
"""Build the atchistory.org URL inventory for the /atc/ migration (worklist 08).

Merges four sources into worklists/data/atc/url_inventory.csv:
  1. live wp-sitemap-*.xml          (the WP URL set = flatten targets)
  2. traffic_analysis/report_data.json  content + downloads (all-time awstats hits)
  3. static-tree walk of the backup (FrontPage-era files, exact paths)
  4. wp-content/uploads walk        (hotlinked assets)

Every row gets new_path = /atc/<old_path>. Rerun any time; output is regenerated.
"""
import csv
import json
import re
import socket
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

# Incapsula serves a JS challenge for HTML documents (bit us at the freeze-day
# rerun, 2026-08-30), so fetch straight from the origin box like atc_fetch_wp.py:
# DNS-only override — SNI + Host header stay www.atchistory.org, and the
# origin's AutoSSL cert verifies cleanly.
ORIGIN_IP = "74.220.207.111"
_real_getaddrinfo = socket.getaddrinfo


def _origin_getaddrinfo(host, *args, **kwargs):
    if host in ("www.atchistory.org", "atchistory.org"):
        host = ORIGIN_IP
    return _real_getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = _origin_getaddrinfo

BACKUP = Path("/Volumes/projects/atchistory_backup")
PUBLIC = BACKUP / "full_mirror/public_html"
REPORT = BACKUP / "traffic_analysis/report_data.json"
OUT = Path("/Users/ryanhemenway/archive.aero/worklists/data/atc/url_inventory.csv")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SITEMAP_INDEX = "https://www.atchistory.org/wp-sitemap.xml"

# Static trees copied from the backup (workstream B rsync list).
STATIC_TREES = ["History", "Images", "Masters", "Search", "_borders", "_fpclass",
                "_overlay", "classphotos", "pdf", "video"]
ROOT_FILES = ["index.htm", "favicon.ico", "ntqrfavicon.ico", "500.shtml",
              "google994516164ab8ab88.html", "readme.html",
              "faa_world_apr_1979.pdf", "faa_world_dec_1976.pdf"]


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt} {url}: {e}", file=sys.stderr)
            time.sleep(3 * attempt)


def norm_path(p):
    """Canonical inventory key: decoded path, leading slash, no host."""
    if p.startswith("http"):
        p = urlsplit(p).path
    return unquote(p) or "/"


def main():
    rows = {}  # path -> dict

    def add(path, source, hits=0, cls=""):
        path = norm_path(path)
        r = rows.setdefault(path, {"old_path": path, "sources": set(),
                                   "hits": 0, "class": cls})
        r["sources"].add(source)
        r["hits"] = max(r["hits"], hits)
        if cls and not r["class"]:
            r["class"] = cls

    # 1. live sitemaps
    idx = fetch(SITEMAP_INDEX).decode("utf-8", "replace")
    children = re.findall(r"<loc>([^<]+)</loc>", idx)
    print(f"sitemap index: {len(children)} child sitemaps")
    for child in children:
        kind = re.search(r"wp-sitemap-([a-z_-]+?)-\d+\.xml", child)
        kind = kind.group(1) if kind else "unknown"
        body = fetch(child).decode("utf-8", "replace")
        locs = re.findall(r"<loc>([^<]+)</loc>", body)
        print(f"  {child.rsplit('/', 1)[-1]}: {len(locs)} URLs")
        for loc in locs:
            add(loc, "sitemap", cls=f"wp_{kind}")
        time.sleep(1)

    # 2. awstats report (all-time hit counts; keys may be list- or dict-shaped)
    rep = json.loads(REPORT.read_text())
    for key, cls in (("content", ""), ("downloads", "download")):
        seq = rep.get(key) or []
        items = seq.items() if isinstance(seq, dict) else seq
        for path, hits in items:
            add(path, f"awstats_{key}", hits=int(hits), cls=cls)

    # 3. static trees from disk (exact paths, incl. spaces)
    n_static = 0
    skip = re.compile(r"\.(php\d?|phtml|pl|cgi)$|^error_log|^\.htaccess", re.I)
    for tree in STATIC_TREES:
        base = PUBLIC / tree
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            if f.is_file() and not skip.search(f.name):
                add("/" + str(f.relative_to(PUBLIC)), "disk", cls="static")
                n_static += 1
    for name in ROOT_FILES:
        if (PUBLIC / name).is_file():
            add("/" + name, "disk", cls="static")
    print(f"disk static files: {n_static}")

    # 4. uploads
    up = PUBLIC / "wp-content/uploads"
    n_up = 0
    for f in up.rglob("*"):
        if f.is_file() and not skip.search(f.name):
            add("/" + str(f.relative_to(PUBLIC)), "disk", cls="upload")
            n_up += 1
    print(f"uploads files: {n_up}")

    # 5. raw access logs (cPanel Raw Access downloads; filenames atchistory.org*)
    logs_dir = Path("/Volumes/projects/atchistory_build/logs")
    log_re = re.compile(r'"(?:GET|HEAD) ([^ "]+) HTTP[^"]*" (\d{3})')
    n_lines = 0
    for lf in sorted(logs_dir.glob("atchistory.org*")):
        tag = "rawlog_2020" if "-20" in lf.name[15:] else "rawlog_current"
        for line in lf.read_text(errors="replace").splitlines():
            m = log_re.search(line)
            if not m:
                continue
            n_lines += 1
            path, status = m.group(1), int(m.group(2))
            vhost = line.rsplit(" ", 2)[-2]
            if status not in (200, 304) or vhost not in ("atchistory.org",
                                                         "www.atchistory.org"):
                continue
            path = path.split("?")[0]
            if path.startswith("http") or len(path) > 300:
                continue
            add(path, tag)
    print(f"raw log lines parsed: {n_lines}")

    # 6. crawl output (definitive WP flatten set incl. probed pagination pages)
    crawl = Path("/Volumes/projects/atchistory_build/crawl")
    n_crawl = 0
    for f in crawl.rglob("*"):
        if not f.is_file() or f.name.startswith("_"):
            continue
        rel = "/" + str(f.relative_to(crawl))
        if rel.endswith("/index.html"):
            rel = rel[: -len("index.html")]
        cls = "wp_pagination" if "/page/" in rel else ""
        add(rel, "crawl", cls=cls)
        n_crawl += 1
    print(f"crawl files: {n_crawl}")

    add("/", "manual", cls="wp_home")
    log_only = sum(1 for r in rows.values()
                   if all(s.startswith("rawlog") for s in r["sources"]))
    print(f"paths seen ONLY in raw logs (new discoveries): {log_only}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["old_path", "old_url", "new_path", "class", "hits", "sources"])
        for r in sorted(rows.values(), key=lambda r: (-r["hits"], r["old_path"])):
            enc = quote(r["old_path"], safe="/")
            w.writerow([r["old_path"],
                        f"https://www.atchistory.org{enc}",
                        "/atc" + ("" if r["old_path"] == "/" else r["old_path"]),
                        r["class"], r["hits"], "|".join(sorted(r["sources"]))])
    print(f"\nwrote {len(rows)} rows -> {OUT}")
    for cls in ("wp_posts-post", "wp_posts-page", "static", "upload", "download"):
        n = sum(1 for r in rows.values() if r["class"] == cls)
        print(f"  {cls or '(blank)'}: {n}")


if __name__ == "__main__":
    main()
