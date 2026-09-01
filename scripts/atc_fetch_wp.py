#!/Users/ryanhemenway/venv/bin/python
"""Fetch the live WordPress pages listed in the atc inventory into the flatten
build dir (worklist 08, workstream B). Static trees/uploads come from the backup,
NOT from here — this only fetches wp_* class rows plus the homepage and feed.

Resumable: existing output files are skipped. Throttled ~1.5s/request. Aborts if
Incapsula starts serving challenge pages so we never hammer through a block.

Usage: atc_fetch_wp.py [--limit N]
"""
import csv
import random
import socket
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

# Incapsula serves a JS challenge for HTML documents, so fetch straight from the
# origin box (same server mail/webdisk point at). DNS-only override: SNI + Host
# header stay www.atchistory.org, and the origin's AutoSSL cert verifies cleanly.
ORIGIN_IP = "74.220.207.111"
_real_getaddrinfo = socket.getaddrinfo


def _origin_getaddrinfo(host, *args, **kwargs):
    if host in ("www.atchistory.org", "atchistory.org"):
        host = ORIGIN_IP
    return _real_getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = _origin_getaddrinfo

INV = Path("/Users/ryanhemenway/archive.aero/worklists/data/atc/url_inventory.csv")
OUT = Path("/Volumes/projects/atchistory_build/crawl")
LOG = OUT / "_fetch_log.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
EXTRA = ["/", "/feed/", "/wp-sitemap.xml", "/wp-sitemap-posts-post-1.xml",
         "/wp-sitemap-posts-page-1.xml", "/wp-sitemap-taxonomies-category-1.xml",
         "/wp-sitemap-taxonomies-post_tag-1.xml", "/wp-sitemap-users-1.xml"]


def out_file(path):
    rel = path.lstrip("/")
    if path.endswith("/") or path == "":
        rel += "index.html"
    elif "." not in rel.rsplit("/", 1)[-1]:
        rel += "/index.html"
    return OUT / rel


def looks_challenged(status, body):
    return status == 200 and len(body) < 4000 and (
        b"Incapsula" in body or b"_Incap_" in body or b"iframe" in body[:600])


def fetch_one(path):
    url = "https://www.atchistory.org" + quote(path, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def probe_pagination():
    """WP archive pagination (/category/x/page/N/) is invisible to sitemaps —
    probe sequentially per archive base until the first 404 (log evidence shows
    /category/facilities/page/100+ exists). Run AFTER the main crawl."""
    bases = ["/"]
    with INV.open() as fh:
        for row in csv.DictReader(fh):
            if row["class"] in ("wp_taxonomies-category", "wp_users"):
                bases.append(row["old_path"].rstrip("/") + "/")
    found = 0
    with (OUT / "_pagination_log.csv").open("w", newline="") as lf:
        log = csv.writer(lf)
        log.writerow(["path", "status", "bytes"])
        for base in bases:
            n = 2
            while n <= 150:
                path = f"{base}page/{n}/"
                f = out_file(path)
                if f.exists():
                    n += 1
                    continue
                status, body = fetch_one(path)
                log.writerow([path, status, len(body)])
                lf.flush()
                if status != 200:
                    break
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(body)
                found += 1
                n += 1
                time.sleep(1.0 + random.random())
            print(f"  {base}: pages 2..{n - 1}")
    print(f"pagination pages saved: {found}")


def main():
    if "--probe-pagination" in sys.argv:
        probe_pagination()
        return
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    paths = list(EXTRA)
    with INV.open() as fh:
        for row in csv.DictReader(fh):
            if row["class"].startswith("wp_") and row["old_path"] not in ("/",):
                paths.append(row["old_path"])

    todo = [p for p in paths if not out_file(p).exists()]
    print(f"{len(paths)} targets, {len(paths) - len(todo)} already fetched, "
          f"{len(todo)} to go")
    if limit:
        todo = todo[:limit]

    OUT.mkdir(parents=True, exist_ok=True)
    challenged = 0
    logmode = "a" if LOG.exists() else "w"
    with LOG.open(logmode, newline="") as lf:
        log = csv.writer(lf)
        if logmode == "w":
            log.writerow(["path", "status", "bytes", "note"])
        for i, path in enumerate(todo, 1):
            url = "https://www.atchistory.org" + quote(path, safe="/")
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            status, body, note = 0, b"", ""
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    status, body = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, note = e.code, "http_error"
            except Exception as e:
                note = f"error:{e}"
            if looks_challenged(status, body):
                challenged += 1
                note = "incapsula_challenge"
                if challenged >= 3:
                    print(f"ABORT: {challenged} consecutive challenge pages at "
                          f"{path} — Incapsula is blocking; switch to local-WP "
                          f"fallback or slow down.", file=sys.stderr)
                    log.writerow([path, status, len(body), note])
                    sys.exit(2)
            elif status == 200:
                challenged = 0
                f = out_file(path)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(body)
            log.writerow([path, status, len(body), note])
            lf.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} (last: {status} {path})")
            time.sleep(1.0 + random.random())
    print("done")


if __name__ == "__main__":
    main()
