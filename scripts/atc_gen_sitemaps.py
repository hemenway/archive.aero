#!/Users/ryanhemenway/venv/bin/python
"""Generate archive.aero sitemaps (worklist 08, workstream F).

Emits, at the repo root (committed at the cutover 11:00 step):
  sitemap.xml       sitemapindex -> the two children below
  sitemap-core.xml  the chart-viewer site's own pages (/, about, contribute,
                    sources)
  sitemap-atc.xml   the ATC collection: CANONICAL URIs ONLY (§A3 rule) — the
                    frozen wp-sitemap snapshots (posts, pages, categories)
                    mapped through route_map.json, plus the /atc/ landing.
                    lastmod preserved from WordPress where it existed.

Deliberately excluded: the users sitemap (author URIs are collapsed away),
the one-entry post_tag sitemap (/tag/marker-beacon/ serves at its old shape
but never entered canonical space — §A3 treats it like FrontPage
scaffolding: reachable, never canonical, never in a sitemap).

Old-URL discovery is NOT this file's job: the frozen wp-sitemap*.xml
snapshots stay served on the atchistory.org hostnames so crawlers re-fetch
the old URLs and see the 301s.

Rerun after the freeze-day crawl refreshes oldhost/wp-sitemap-*.xml, then
`--verify` (every URL must answer 200 on prod with zero redirect hops)
before the 11:00 commit.
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
from atc_redirect_check import route_old_path  # noqa: E402  (shared model)

REPO = Path("/Users/ryanhemenway/archive.aero")
OLDHOST = Path("/Volumes/projects/atchistory_build/oldhost")
ORIGIN = "https://archive.aero"
NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

SOURCES = ["wp-sitemap-posts-post-1.xml", "wp-sitemap-posts-page-1.xml",
           "wp-sitemap-taxonomies-category-1.xml"]
# Extensionless canonicals (URI-POLICY.md, 2026-08-21): GH Pages serves these
# natively; the .html spellings are permanent 301 aliases via CF redirect rules.
CORE_PAGES = ["/", "/about", "/contribute", "/sources"]


def parse_wp(path):
    for url in ET.parse(path).getroot().iter(NS + "url"):
        loc = url.findtext(NS + "loc") or ""
        lastmod = url.findtext(NS + "lastmod") or ""
        if loc:
            yield loc, lastmod


def urlset(entries):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for loc, lastmod in entries:
        lm = f"<lastmod>{escape(lastmod)}</lastmod>" if lastmod else ""
        lines.append(f" <url><loc>{escape(loc)}</loc>{lm}</url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="fetch every emitted URL: 200 required, 0 hops")
    args = ap.parse_args()

    canon = {}          # canonical URI -> lastmod
    problems, collapsed = [], []
    canon["/atc/"] = ""  # the designed landing page (replaces the WP home)

    for src in SOURCES:
        f = OLDHOST / src
        if not f.is_file():
            sys.exit(f"missing frozen sitemap: {f} — refresh the freeze-day "
                     f"snapshot first")
        for loc, lastmod in parse_wp(f):
            path = urlsplit(loc).path
            if path == "/":
                target = "/atc/"      # front page -> landing
            else:
                r = route_old_path(path)
                if r is None or r.get("status"):
                    problems.append((path, r))
                    continue
                target = r["to"]
            if target == "/atc/" and path != "/":
                collapsed.append(path)   # e.g. /home/ (drop-map)
            if not target.startswith("/atc"):
                problems.append((path, r))
                continue
            prev = canon.get(target, "")
            canon[target] = max(prev, lastmod)  # keep newest lastmod

    if problems:
        print(f"{len(problems)} sitemap URLs failed to map — not writing:")
        for p, r in problems[:20]:
            print(f"  {p} -> {r}")
        sys.exit(1)

    # The landing is ours, not WP's: install_landing() refreshes its Recent
    # additions on every rebuild, so generation date beats the old front
    # page's lastmod.
    canon["/atc/"] = time.strftime("%Y-%m-%d")
    atc_entries = [(ORIGIN + u, lm) for u, lm in sorted(canon.items())]
    core_entries = [(ORIGIN + p, "") for p in CORE_PAGES]
    today = time.strftime("%Y-%m-%d")

    (REPO / "sitemap-atc.xml").write_text(urlset(atc_entries))
    (REPO / "sitemap-core.xml").write_text(urlset(core_entries))
    (REPO / "sitemap.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
 <sitemap><loc>{ORIGIN}/sitemap-atc.xml</loc><lastmod>{today}</lastmod></sitemap>
 <sitemap><loc>{ORIGIN}/sitemap-core.xml</loc><lastmod>{today}</lastmod></sitemap>
</sitemapindex>
""")
    print(f"sitemap-atc.xml: {len(atc_entries)} canonical URLs "
          f"({len(collapsed)} old URLs collapsed into the landing: "
          f"{collapsed or '—'})")
    print(f"sitemap-core.xml: {len(core_entries)} URLs; sitemap.xml: index")

    if not args.verify:
        return

    import requests
    from concurrent.futures import ThreadPoolExecutor
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 atc-sitemap-verify"
    bad = []

    def check(url):
        try:
            r = sess.get(url, allow_redirects=False, timeout=30, stream=True)
            st, loc = r.status_code, r.headers.get("location", "")
            r.close()
        except Exception as e:
            st, loc = 0, f"{type(e).__name__}"
        if st != 200:
            bad.append((url, st, loc))

    urls = [u for u, _ in atc_entries + core_entries]
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(check, urls))
    if bad:
        print(f"\nVERIFY FAILED — {len(bad)} URLs not a clean 200:")
        for u, st, loc in sorted(bad)[:30]:
            print(f"  {st} {u} {loc}")
        sys.exit(1)
    print(f"verify: all {len(urls)} URLs answer 200 with 0 redirect hops")


if __name__ == "__main__":
    main()
