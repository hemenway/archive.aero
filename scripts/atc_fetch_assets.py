#!/Users/ryanhemenway/venv/bin/python
"""Fetch the WP asset closure for the atc flatten (worklist 08).

The page crawl captured HTML only; uploads came from the backup — but theme,
plugin, and wp-includes assets were never copied. This walks every crawled
page, extracts same-host asset references (css/js/img/fonts/media), fetches
missing ones origin-direct into crawl/ (so the rewrite/merge pass picks them
up), then resolves url(...) refs inside fetched CSS one level deep.

Rerun-safe: existing files are skipped.
"""
import re
import socket
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import urllib.request

ORIGIN_IP = "74.220.207.111"
_real = socket.getaddrinfo


def _gai(host, *a, **k):
    if host in ("www.atchistory.org", "atchistory.org"):
        host = ORIGIN_IP
    return _real(host, *a, **k)


socket.getaddrinfo = _gai

CRAWL = Path("/Volumes/projects/atchistory_build/crawl")
SITE = Path("/Volumes/projects/atchistory_build/site")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

ASSET_EXT = r"(?:css|js|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|eot|otf|map|mp4|mp3)"
REF_RE = re.compile(
    rb'(?:https?:)?//(?:www\.)?atchistory\.org(/[^"\'()<>\s?#]+\.' +
    ASSET_EXT.encode() + rb')(?:[?#]|["\'()<>\s])', re.I)
ROOTREL_RE = re.compile(
    rb'(?:href|src)=["\'](/(?:wp-content|wp-includes)/[^"\'?#]+\.' +
    ASSET_EXT.encode() + rb')', re.I)
CSSURL_RE = re.compile(
    rb'url\(\s*["\']?([^"\')\s?#]+\.' + ASSET_EXT.encode() +
    rb')(?:[?#][^"\')\s]*)?["\']?\s*\)', re.I)


def fetch(path):
    url = "https://www.atchistory.org" + urllib.parse.quote(path, safe="/")
    # Accept: */* matters — mod_security on the origin 406s bare requests for
    # some font extensions (2026-08-09 parity finding)
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def wanted_paths():
    found = set()
    for f in CRAWL.rglob("*.html"):
        data = f.read_bytes()
        for m in REF_RE.finditer(data):
            found.add(unquote(m.group(1).decode("latin-1")))
        for m in ROOTREL_RE.finditer(data):
            found.add(unquote(m.group(1).decode("latin-1")))
    return found


def missing(paths):
    out = []
    for p in sorted(paths):
        rel = p.lstrip("/")
        if not (CRAWL / rel).exists() and not (SITE / rel).exists() \
                and not (Path("/Volumes/projects/atchistory_build/static") / rel).exists():
            out.append(p)
    return out


def save(path, data):
    f = CRAWL / path.lstrip("/")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data)
    return f


def main():
    refs = wanted_paths()
    todo = missing(refs)
    print(f"asset refs: {len(refs)}; missing: {len(todo)}")
    fetched_css, got, miss = [], 0, []
    for i, p in enumerate(todo, 1):
        status, data = fetch(p)
        if status == 200:
            f = save(p, data)
            got += 1
            if p.lower().endswith(".css"):
                fetched_css.append((p, f))
        else:
            miss.append((status, p))
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(0.5)

    # second level: fonts/images referenced from CSS. Scan EVERY css file in
    # the build (not just ones fetched this run) — the rerun-safe skip in pass
    # one otherwise hides refs inside previously-fetched stylesheets, which is
    # how the collapsing-category-list fonts were missed (2026-08-09).
    all_css = list(fetched_css)
    for root in (CRAWL, Path("/Volumes/projects/atchistory_build/static")):
        for cfile in root.rglob("*.css"):
            all_css.append(("/" + str(cfile.relative_to(root)), cfile))
    css_refs = set()
    for cpath, cfile in all_css:
        base = "https://www.atchistory.org" + cpath
        for m in CSSURL_RE.finditer(cfile.read_bytes()):
            u = m.group(1).decode("latin-1")
            if u.startswith("data:"):
                continue
            absu = urljoin(base, u)
            sp = urlsplit(absu)
            if sp.netloc in ("www.atchistory.org", "atchistory.org"):
                css_refs.add(unquote(sp.path))
    todo2 = missing(css_refs)
    print(f"css-level refs: {len(css_refs)}; missing: {len(todo2)}")
    for p in todo2:
        status, data = fetch(p)
        if status == 200:
            save(p, data)
            got += 1
        else:
            miss.append((status, p))
        time.sleep(0.5)

    print(f"fetched: {got}; failures: {len(miss)}")
    for s, p in miss[:20]:
        print(f"  {s} {p}")


if __name__ == "__main__":
    main()
