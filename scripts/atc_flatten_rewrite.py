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
# WordPress post ids -> old path (worker-atc/src/p_map.json): every preserved
# page ships <link rel="shortlink" href=".../atc/?p=N">, which the serving
# Worker now resolves, but a flattened page should not need a redirect to
# name itself. Rewritten to the canonical URI below.
P_MAP = json.loads((ROUTE_MAP.parent / "p_map.json").read_text())


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
    query = cut.split("#", 1)[0] if cut.startswith("?") else ""
    canon = None
    if path in ("/", "") and query:
        # /atc/?p=N (or page_id=N): a WP shortlink; resolve through the post map
        # and drop the query, exactly as the Worker would.
        m_pid = re.search(r"[?&](?:p|page_id)=(\d+)", query.replace("&amp;", "&"))
        if m_pid and m_pid.group(1) in P_MAP:
            old_path = P_MAP[m_pid.group(1)]
            canon = canonical_of(old_path) or ("/atc" + old_path)
            cut = cut[len(query):]
    if canon is None:
        canon = canonical_of(unquote(path.replace("&amp;", "&")))
    if not canon:
        return None
    # A target may legitimately leave the /atc space — A2 sends the old
    # /History/Maps/Maps.htm to the chart viewer at the site root — so the
    # replacement covers the whole match, "/atc" prefix included.
    if canon.startswith("/atc"):
        out, head = canon[len("/atc"):], prefix
        # Already canonical: leave the bytes be -- including a percent-encoded
        # spelling of a rule-derived name (the generated listings link
        # /atc/history/photos/Omaha%20Antenna.jpg); rewriting it would only
        # un-encode the space.
        if out == path or out == unquote(path.replace("&amp;", "&")):
            return None
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
    rb'''(?is)<link[^>]+href=["']?[^"'>]*/atc/(?:category|author|comments)/'''
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


# --- contact + donation pass (2026-09-01) --------------------------------
# The crawl carries the previous operation's contact address and its PayPal
# donation calls on nearly every page. The mailbox forwards, but the archive is
# ours now and the PayPal account is not: every "donate" button on the site
# still paid the old operator. Both are retargeted -- support goes to the
# archive's own Buy Me a Coffee, contact to the archive's own address. Runs
# AFTER the host/canonical passes, so the donate hrefs it matches are already
# in their /atc/ form.
CONTACT = b"ryan@archive.aero"
MAILTO = b'<a href="mailto:' + CONTACT + b'">' + CONTACT + b"</a>"
COFFEE = b"https://buymeacoffee.com/ryanhemenway"
# Self-contained button: no external image, so the page gains no new host.
# #2b5f71 is the escapade theme's own accent (see escapade-inline-css).
COFFEE_BTN = (
    b'<a href="' + COFFEE + b'" target="_blank" rel="noopener" '
    b'style="display:inline-block;padding:8px 15px;border:1px solid #2b5f71;'
    b'border-radius:4px;color:#2b5f71;font-family:Oswald,sans-serif;'
    b'font-weight:600;text-decoration:none">Buy me a coffee</a>')
SUPPORT_COPY = (
    b"This archive is free to read and free to search. If it is useful to "
    b'you, you can <a href="' + COFFEE + b'" target="_blank" '
    b'rel="noopener">buy me a coffee</a> &mdash; it goes towards hosting, '
    b"storage, and the scanning that keeps the collection growing. Material "
    b"helps just as much: photographs, directories, and documents are always "
    b"welcome at " + MAILTO + b".")

CONTACTS = [
    # 1. the address itself, in mailto: and in prose alike
    ("contact_mail", re.compile(rb"(?i)archive@atchistory\.org"), CONTACT),
    # 2. the /donate page body: PayPal form + its ask, replaced wholesale.
    #    The URI stays live (URI-POLICY); only what it says changes.
    ("donate_page", re.compile(
        rb"(?is)<p><strong>Thank you for donating.*?"
        rb"The Air Traffic Control History Archive</p>"),
     b"<p><strong>Thank you for supporting this archive.</strong></p>\n"
     b"<p>The Air Traffic Control History collection is preserved and "
     b'maintained as part of <a href="/">archive.aero</a>, free to read and '
     b"free to search. If it has been useful to you, you can chip in towards "
     b"what keeps it online: hosting, storage, and the scanning that adds to "
     b"it.</p>\n"
     b"<p>" + COFFEE_BTN + b"</p>\n"
     b"<p>Contributions of material help just as much &mdash; photographs of "
     b"air traffic facilities, equipment, and the people who ran them; "
     b"directories, manuals, and other documents. Scans of at least "
     b"300&nbsp;dpi are best. Write to " + MAILTO + b".</p>\n"
     b"<p>Many Thanks!<br />\nThe Air Traffic Control History Archive</p>"),
    # 3. the sidebar PayPal button widget, on ~1,850 pages
    ("donate_button", re.compile(
        rb'(?is)<a href="[^"]*/donate[^"]*">\s*'
        rb"<img[^>]*btn_donateCC_LG\.gif[^>]*>\s*</a>"),
     COFFEE_BTN),
    # 4. FrontPage-era PayPal buttons (encrypted _s-xclick forms)
    ("donate_form_fp", re.compile(
        rb'(?is)<form action="https://www\.paypal\.com/cgi-bin/webscr"'
        rb"[^>]*>.*?</form>"),
     COFFEE_BTN),
    # 5. the two donation asks in prose (WP home, FrontPage class-photo index)
    ("donate_prose_wp", re.compile(
        rb"(?is)<p>Any monetary donations will be used.*?</p>"),
     b"<p>" + SUPPORT_COPY + b"</p>"),
    # ...but the FrontPage one sits in the table cell right beside the button,
    # so it gets a variant that does not repeat the link.
    ("donate_prose_fp", re.compile(
        rb"(?is)Donations will go towards.*?Thank you for your "
        rb"consideration\."),
     b"Support goes towards hosting, storage, and the scanning that keeps "
     b"this archive growing. Material helps just as much: photographs, "
     b"directories, and documents are always welcome at " + MAILTO + b"."),
    # 6. /contact ran on Ninja Forms -- a WordPress plugin that POSTs to PHP.
    #    There is no PHP behind the static archive, so the form rendered but
    #    could never deliver a message. Replaced by the address it would have
    #    mailed. Only this page carries the plugin, so its front-end bundle and
    #    Backbone templates (~200 KB, next rule) go with it.
    ("contact_form", re.compile(
        rb'(?is)<noscript class="ninja-forms-noscript-message">.*?'
        rb"nfForms\.push\(form\);</script>"),
     b"<p>Email " + MAILTO + b".</p>\n"
     b"<p>Photographs, documents, corrections, and questions about the "
     b"collection are all welcome. Scans of at least 300&nbsp;dpi are "
     b"best.</p>"),
    ("contact_form_assets", re.compile(
        rb'(?is)<script id=[\'"](?:nf-front-end|tmpl-nf-)[^\'"]*[\'"]'
        rb"[^>]*>.*?</script>\s*"),
     b""),
    ("contact_form_css", re.compile(
        rb'(?is)<link[^>]+id=[\'"]nf-[^\'"]*[\'"][^>]*>\s*'),
     b""),
]

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
            for name, rx, repl in CONTACTS:
                data, n = rx.subn(repl, data)
                rw_counts[name] += n
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
