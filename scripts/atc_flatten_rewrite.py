#!/Users/ryanhemenway/venv/bin/python
"""Merge crawl/ + static/ into site/ (the R2 upload set) and run the byte-safe
rewrite + hygiene scan (worklist 08, workstream B).

- BYTES ONLY: FrontPage-era pages are windows-1252; a utf-8 decode would corrupt
  them. All regexes operate on bytes; files are rewritten byte-for-byte.
- Host rewrite: every scheme/protocol-relative/JSON-escaped reference to
  (www.)atchistory.org becomes https://archive.aero/atc. mailto: and bare-text
  mentions are deliberately untouched (mail keeps working; prose stays prose).
- Root-relative href/src/url(/...) get the /atc prefix.
- Strips (D3 + hygiene): AdSense, SiteLock badges, Incapsula resources.
- wp-sitemap*.xml are moved to oldhost/ (served on the OLD hostname as frozen
  old-URL sitemaps; they must keep atchistory.org URLs, so they skip rewriting).
- Scan report: residual host references classified, external script/iframe host
  histogram, injected-eval sniff -> worklists/data/atc/rewrite_report.txt

Idempotent: rebuilds site/ from scratch each run (APFS clones when supported).
"""
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
import posixpath
from pathlib import Path
from urllib.parse import unquote

BUILD = Path("/Volumes/projects/atchistory_build")
CRAWL, STATIC, SITE, OLDHOST = (BUILD / d for d in
                                ("crawl", "static", "site", "oldhost"))
REPORT = Path("/Users/ryanhemenway/archive.aero/worklists/data/atc/"
              "rewrite_report.txt")
LANDING = Path("/Users/ryanhemenway/archive.aero/worker-atc/static/index.html")
JUNK_NAMES = {".DS_Store", "Thumbs.db"}

REWRITE_EXT = {".htm", ".html", ".css", ".xml", ".shtml"}
NEW = b"https://archive.aero/atc"

HOST = rb"(?:https?:)?//(?:www\.)?atchistory\.org"
HOST_JSON = rb"https?:\\/\\/(?:www\.)?atchistory\.org"

STRIPS = [
    ("adsense_script", re.compile(
        rb"(?is)<script[^>]*(?:googlesyndication|adsbygoogle)[^>]*>\s*</script>")),
    ("adsense_inline", re.compile(
        rb"(?is)<script[^>]*>(?:(?!</script>).)*adsbygoogle(?:(?!</script>).)*</script>")),
    ("adsense_ins", re.compile(rb"(?is)<ins[^>]+adsbygoogle.*?</ins>")),
    ("sitelock_img", re.compile(rb"(?is)<img[^>]*sitelock[^>]*/?>")),
    ("sitelock_link", re.compile(rb"(?is)<a[^>]*sitelock[^>]*>.*?</a>")),
    ("incapsula", re.compile(
        rb"(?is)<script[^>]*_Incapsula_Resource[^>]*>\s*</script>")),
]
REWRITES = [
    ("host", re.compile(rb"(?i)" + HOST), NEW),
    ("host_json", re.compile(rb"(?i)" + HOST_JSON),
     NEW.replace(b"/", b"\\/")),
    ("host_pctenc", re.compile(
        rb"(?i)https?%3A%2F%2F(?:www\.)?atchistory\.org"),
     b"https%3A%2F%2Farchive.aero%2Fatc"),
    ("rootrel_attr", re.compile(
        rb'(?i)\b(href|src|action|data-src|poster)=(["\'])/(?!/|atc/)'),
     rb"\1=\2/atc/"),
    ("rootrel_css", re.compile(rb'(?i)url\((["\']?)/(?!/|atc/)'),
     rb"url(\1/atc/"),
]
# --- canonical URI pass (worklist 08 workstream A) ------------------------
# The host pass above lands every reference on /atc/<old path>. This second
# pass walks those onto the canonical URI so internal links point at the
# permanent address directly — otherwise every image on every page would cost a
# 301. Old spellings still resolve; they just are not what we ship.
# Map: worker-atc/src/route_map.json (scripts/atc_canonical_map.py).
ROUTE_MAP = Path("/Users/ryanhemenway/archive.aero/worker-atc/src/route_map.json")
_R = json.loads(ROUTE_MAP.read_text())
_ALIAS, _RULES, _DROP = _R["alias"], _R["rules"], _R["drop"]


def canonical_of(inner):
    """Old site path -> canonical URI, or None if already canonical/unknown.
    Mirrors routeOldPath()+canonicalOf() in worker-atc/src/routes.js — same
    order: drops are authoritative decisions and must beat the prefix rules,
    or a legacy-mapped page like /History/FacilityPhotos/index.htm would be
    rewritten to a rule-derived alias instead of its living successor."""
    alt = inner[:-1] if inner.endswith("/") else inner + "/"
    target = _DROP.get(inner, _DROP.get(alt))
    if target:
        return None if target == "410" else target
    hit = _ALIAS.get(inner, _ALIAS.get(alt))
    if hit is not None:
        return hit
    for old, new in _RULES:
        if inner.startswith(old):
            return new + inner[len(old):]
    return None


# Reference forms produced by the host pass, in their three encodings. Each
# captures the path after /atc so it can be looked up and replaced.
CANON_PLAIN = re.compile(rb'/atc(/[^\s"\'<>)\\]*)')
CANON_JSON = re.compile(rb'/atc((?:\\/[^\s"\\]*)*)')
CANON_PCT = re.compile(rb'(?i)%2Fatc((?:%2F[^\s"\'<>)]*)*)')


def _canon_bytes(raw, sep, prefix):
    """raw: captured path bytes with "/" separators, i.e. what followed "/atc".
    Returns the FULL replacement for the match (prefix included) or None to
    leave the bytes alone. Any ?query / #fragment rides along untouched."""
    text = raw.decode("latin-1")
    path, cut = text, ""
    for mark in ("?", "#"):
        i = path.find(mark)
        if i >= 0:
            path, cut = path[:i], path[i:] + cut
    canon = canonical_of(unquote(path.replace("&amp;", "&")))
    if not canon:
        return None
    # A target may legitimately leave the /atc space — A2 sends the old
    # /History/Maps/Maps.htm to the chart viewer at the site root — so the
    # replacement covers the whole match, "/atc" prefix included.
    if canon.startswith("/atc"):
        out, head = canon[len("/atc"):], prefix
        if out == path:
            return None                      # already canonical, leave bytes be
    else:
        out, head = canon, b""
    # Canonical paths are ASCII with no spaces (slugify/despace guarantee it),
    # so they need no percent-encoding — only "&" re-escaping in HTML context.
    if "&amp;" in text:
        out = out.replace("&", "&amp;")
    if sep != "/":
        out = out.replace("/", sep)
    return head + (out + cut).encode("latin-1")


# FrontPage pages link each other with RELATIVE hrefs, which the host pass
# never touches. Those still resolve — a relative link is depth-relative and
# canonicalization never changes a page's depth — but the target's own name may
# have changed (extension dropped, spaces slugified), so each click would cost
# a 301. Resolve against the containing page and emit the canonical directly.
CANON_REL = re.compile(
    rb'''(?i)\b(href|src)=(["'])(?![a-z]+:|//|/|\#)([^"'>]*?\.(?:s?html?|xls|pdf))(["'\#?])''')


def canonicalize_relative(data, rel_dir, counts):
    def sub(m):
        attr, q, target, close = m.groups()
        t = unquote(target.decode("latin-1").replace("&amp;", "&"))
        old = posixpath.normpath(posixpath.join("/" + rel_dir, t))
        canon = canonical_of(old)
        if not canon or canon == "/atc" + old:
            return m.group(0)
        counts["canon_relative"] += 1
        return attr + b"=" + q + canon.encode("latin-1") + close
    return CANON_REL.sub(sub, data)


# Per-category/author RSS alternates. Policy retires those feeds as 410 (only
# the site feed survives as a frozen snapshot), so leaving the <link> tags in
# means our own pages advertise dead endpoints to feed readers and crawlers.
# Runs after canonicalization, when the hrefs have settled into /atc/ form.
DEAD_FEED_ALT = re.compile(
    rb'''(?is)<link[^>]+href=["']?[^"'>]*/atc/(?:category|author)/'''
    rb'''[^"'>]*feed/?["']?[^>]*>\s*''')


def canonicalize(data, counts):
    def make(name, sep, unescape, prefix):
        def sub(m):
            if not m.group(1):
                return m.group(0)
            r = _canon_bytes(unescape(m.group(1)), sep, prefix)
            if r is None:
                return m.group(0)
            counts[name] += 1
            return r
        return sub

    data = CANON_JSON.sub(
        make("canon_json", "\\/", lambda b: b.replace(rb"\/", b"/"), b"/atc"), data)
    data = CANON_PCT.sub(
        make("canon_pct", "%2F", lambda b: re.sub(rb"(?i)%2F", b"/", b),
             b"%2Fatc"), data)
    data = CANON_PLAIN.sub(make("canon_plain", "/", lambda b: b, b"/atc"), data)
    data, n = DEAD_FEED_ALT.subn(b"", data)
    counts["dead_feed_alt_stripped"] += n
    return data


SCAN_RESIDUAL = re.compile(rb"(?i)atchistory\.org")
# Old-shaped references that survived the canonical pass — these would each
# cost a 301 on a page the archive itself serves.
# Case-SENSITIVE on purpose: canonical space uses /atc/history/ and
# /atc/images/, so a case-insensitive match would flag the correct form.
SCAN_ALIASREF = re.compile(
    rb'/atc/(wp-content/|wp-includes/|category/|author/|page/|History/|'
    rb'classphotos/|pdf/|Images/|Masters/)')
SCAN_EVAL = re.compile(rb"(?i)eval\s*\(\s*(?:base64|atob|unescape|String\.fromCharCode)")
SCAN_EXT_SRC = re.compile(
    rb'(?i)<(?:script|iframe)[^>]+src=["\']?(?:https?:)?//([^/"\'>\s]+)')


def merge():
    # The projects volume intermittently fails deletes mid-tree (.DS_Store /
    # indexer races). Retry rm; if it still won't die, shove the remnant aside
    # so the rebuild is never blocked and never partial.
    import time
    for attempt in range(4):
        if not SITE.exists():
            break
        r = subprocess.run(["rm", "-rf", str(SITE)], capture_output=True)
        if r.returncode == 0 and not SITE.exists():
            break
        time.sleep(2)
    if SITE.exists():
        aside = SITE.with_name(f"site.trash.{int(time.time())}")
        SITE.rename(aside)
        print(f"note: stubborn site/ moved aside to {aside.name}; delete later")
    SITE.mkdir()
    OLDHOST.mkdir(exist_ok=True)
    # APFS clone when possible; cp falls back to plain copy otherwise
    subprocess.run(["cp", "-Rc", str(STATIC) + "/.", str(SITE)], check=True)
    subprocess.run(["cp", "-Rc", str(CRAWL) + "/.", str(SITE)], check=True)
    for junk in SITE.glob("_*.csv"):
        junk.unlink()
    n_junk = 0
    for j in SITE.rglob("*"):
        if j.name in JUNK_NAMES and j.is_file():
            j.unlink()
            n_junk += 1
    n_old = 0
    for sm in SITE.glob("wp-sitemap*.xml"):
        shutil.move(str(sm), OLDHOST / sm.name)
        n_old += 1
    print(f"merged -> site/ ; {n_old} old-URL sitemaps -> oldhost/ ; "
          f"{n_junk} Finder junk files stripped")


def install_landing():
    """The /atc/ landing page replaces the WP homepage (workstream B). Source
    is versioned in worker-atc/static/index.html; runs AFTER the rewrite pass
    (its root links to the main site must not gain the /atc prefix). The
    Recent-additions list is injected from the frozen feed so a freeze-day
    rebuild refreshes it automatically. The canonical pass runs here too, as a
    backstop for hand-edited links — but only over /atc/ refs, so the page's
    links out to the main site (href="/", /about) stay untouched."""
    counts = Counter()
    page = canonicalize(LANDING.read_bytes(), counts).decode("utf-8")
    if any(counts.values()):
        print(f"landing page: canonicalized {dict(counts)} refs "
              f"(fix them at source in {LANDING.name})")
    feed = (SITE / "feed/index.html").read_bytes().decode("utf-8", "replace")
    items = re.findall(r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>"
                       r".*?<pubDate>(.*?)</pubDate>", feed, re.S)
    lis = []
    for title, link, pub in items[:8]:
        path = link.strip()
        if path.startswith("https://archive.aero/"):
            path = path[len("https://archive.aero"):]
        d = pub.strip().split()  # "Thu, 24 Jul 2026 17:56:02 +0000"
        date = f"{d[2]} {int(d[1])}, {d[3]}" if len(d) >= 4 else ""
        lis.append(f'            <li><a href="{path}">{title.strip()}</a>'
                   f"<time>{date}</time></li>")
    if lis:
        page = re.sub(r"<!-- RECENT_POSTS:BEGIN.*?RECENT_POSTS:END -->",
                      "<!-- RECENT_POSTS:BEGIN -->\n" + "\n".join(lis)
                      + "\n            <!-- RECENT_POSTS:END -->",
                      page, count=1, flags=re.S)
    (SITE / "index.html").write_text(page)
    print(f"landing page installed -> site/index.html "
          f"(recent posts: {len(lis)})")


def main():
    if "--landing-only" in sys.argv:
        install_landing()
        return
    merge()
    strip_counts, rw_counts = Counter(), Counter()
    residual, ext_hosts, evals, aliasrefs = [], Counter(), [], Counter()
    n_files = n_changed = 0

    for f in sorted(SITE.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        rel = f.relative_to(SITE)
        in_uploads = rel.parts[:2] == ("wp-content", "uploads")
        if ext not in REWRITE_EXT and not (ext == ".js"):
            continue
        data = orig = f.read_bytes()
        n_files += 1
        if not (in_uploads and ext == ".xml"):
            for name, rx in STRIPS:
                data, n = rx.subn(b"", data)
                strip_counts[name] += n
            for name, rx, repl in REWRITES:
                data, n = rx.subn(repl, data)
                rw_counts[name] += n
            # /atc/<old path> -> /atc/<canonical>: ship the permanent URI, not
            # an alias that costs a 301 on every hit.
            data = canonicalize(data, rw_counts)
            data = canonicalize_relative(data, str(rel.parent) + "/"
                                         if str(rel.parent) != "." else "",
                                         rw_counts)
        if data != orig:
            f.write_bytes(data)
            n_changed += 1
        # hygiene scan on the final bytes
        for m in SCAN_RESIDUAL.finditer(data):
            ctx = data[max(0, m.start() - 30):m.end() + 10]
            kind = ("mailto" if b"mailto:" in ctx else
                    "text")
            if kind == "text" and re.search(rb'(?i)(href|src|url|content)\s*=',
                                            ctx):
                kind = "ATTR!"
            residual.append((str(rel), kind, ctx.decode("latin-1")))
        for m in SCAN_EXT_SRC.finditer(data):
            ext_hosts[m.group(1).decode("latin-1").lower()] += 1
        if SCAN_EVAL.search(data):
            evals.append(str(rel))
        for m in SCAN_ALIASREF.finditer(data):
            aliasrefs[m.group(1).decode("latin-1")] += 1

    with REPORT.open("w") as r:
        r.write(f"files processed: {n_files}; changed: {n_changed}\n")
        r.write(f"rewrites: {dict(rw_counts)}\n")
        r.write(f"strips: {dict(strip_counts)}\n\n")
        r.write(f"external script/iframe hosts ({len(ext_hosts)}):\n")
        for h, c in ext_hosts.most_common():
            r.write(f"  {c:6d}  {h}\n")
        r.write(f"\nalias-shaped refs still shipped "
                f"(each costs a 301): {sum(aliasrefs.values())}\n")
        for k, c in aliasrefs.most_common():
            r.write(f"  {c:6d}  /atc/{k}\n")
        r.write(f"\neval-obfuscation hits ({len(evals)}):\n")
        for p in evals:
            r.write(f"  {p}\n")
        bad = [x for x in residual if x[1] == "ATTR!"]
        r.write(f"\nresidual atchistory.org refs: {len(residual)} "
                f"(attr-context: {len(bad)})\n")
        for p, kind, ctx in residual[:400]:
            r.write(f"  [{kind}] {p}: …{ctx}…\n")

    kinds = Counter(k for _, k, _ in residual)
    print(f"processed {n_files} text files, changed {n_changed}")
    print(f"rewrites: {dict(rw_counts)}")
    print(f"strips: {dict(strip_counts)}")
    print(f"alias-shaped refs remaining: {sum(aliasrefs.values())} "
          f"{dict(aliasrefs)}")
    print(f"residual refs by kind: {dict(kinds)}; "
          f"eval hits: {len(evals)}; external hosts: {len(ext_hosts)}")
    print(f"report -> {REPORT}")
    install_landing()


if __name__ == "__main__":
    main()
