#!/Users/ryanhemenway/venv/bin/python
"""Build the canonical URI space for /atc/ and the permanent alias map into it
(worklist 08, workstream A).

W3C "Cool URIs don't change" (https://www.w3.org/Provider/Style/URI) applied to
the atchistory.org corpus. The canonical space carries none of the things that
doc says kill a URI: no software mechanism (wp-content, wp-includes, category,
author, page/N), no file extensions on documents, no collision suffixes, no
mixed-case top-level trees, no spaces. Every historical path from BOTH
generations (FrontPage 1998-2015, WordPress 2016-2026) stays alive forever as
an alias -> single-hop 301.

Storage is NOT renamed. R2 keys stay the old paths byte-for-byte; the worker
holds both directions. That is the doc's own prescription -- decouple the URI
from the file location rather than moving files to match URIs.

Deliberate boundaries (stated so they don't drift):
  * DOCUMENTS get the full treatment now; ASSET paths get prefix normalization
    only (top-level tree rename + space removal). Deeper mixed-case components
    -- /atc/history/Pubs/, .../FacilityPhotos/ -- stay. The asymmetry is
    principled, not lazy: document URIs are what people link and what search
    engines rank, so changing them later costs a second redirect event on
    ranked URLs. Asset paths are referenced only from our own pages, so
    normalizing them later costs nothing (rewrite our hrefs, add aliases).
    Deferring the free half keeps the cutover diff small.
  * Leaf filenames keep their bytes -- they are the artifact's own name, stable
    25 years. Only hostile components (spaces) and linkable documents get
    slugified.
  * Format extensions on real files (.pdf, .xls, .jpg) stay. The doc objects to
    .html/.cgi because they name the *server's* technology, not the artifact's
    format.
  * FrontPage scaffolding (_borders, _fpclass, _overlay, Search) keeps serving
    at its old paths -- pages reference it -- but never enters canonical space
    and never enters a sitemap.
  * No date partitioning of articles: WP post dates are 2016+ bulk-import
    timestamps, not authorship dates, so /2016/ would encode noise. (The one
    thing the doc recommends keeping IN; declined on evidence.)

Outputs:
  worker-atc/src/route_map.json      rules + alias + key + drop (worker data)
  worklists/data/atc/canonical_report.txt   review report, incl. anything the
                                            automatic rules could not settle
"""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/ryanhemenway/archive.aero")
INVENTORY = REPO / "worklists/data/atc/url_inventory.csv"
LEGACY = REPO / "worker-atc/src/legacy_map.json"
OUT_MAP = REPO / "worker-atc/src/route_map.json"
REPORT = REPO / "worklists/data/atc/canonical_report.txt"

PREFIX = "/atc"

# Ordered prefix rules: old path prefix -> canonical prefix. First match wins,
# and the worker applies them in both directions (canonical -> R2 key).
RULES = [
    ("/wp-content/uploads/", "/atc/media/"),
    ("/wp-content/themes/", "/atc/assets/themes/"),
    ("/wp-content/plugins/", "/atc/assets/plugins/"),
    ("/wp-includes/", "/atc/assets/lib/"),
    ("/History/", "/atc/history/"),
    ("/classphotos/", "/atc/class-photos/"),
    ("/pdf/", "/atc/library/"),
    ("/Images/", "/atc/images/"),
    ("/Masters/", "/atc/masters/"),
    ("/video/", "/atc/video/"),
]

# Served but never canonical: FrontPage scaffolding + WP internals that pages
# still reference. No sitemap entry, no rel=canonical, noindex.
NOCANON = ("/_borders/", "/_fpclass/", "/_overlay/", "/Search/", "/wp-json/")
# Server furniture, not resources: no canonical URI of their own.
NOCANON_FILES = {"/500.shtml", "/favicon.ico", "/ntqrfavicon.ico", "/robots.txt",
                 "/google994516164ab8ab88.html"}
INDEX_NAMES = {"index.htm", "index.html", "default.htm", "default.html"}

# Theme demo / test artifacts that must never become permanent archive.aero
# URIs. 301 to the collection home.
JUNK_SLUGS = {
    "advertising-area-one", "example-location", "font-text",
    "test-page-with-columns", "ad-test",
}

# Per-path decisions the rules cannot make. Every entry needs a reason.
MANUAL = {
    # curated class-photos home; the directory root serves a generated listing,
    # so this keeps its own URI rather than taking /atc/class-photos/
    "/classphotos/PhotoHome.htm": "/atc/class-photos/photo-home",
    # nested page chain flattened; "-2" is a WP slug collision against the
    # parent page /historical-maps/, so it cannot be auto-stripped
    "/historical-maps/": "/atc/historical-maps",
    "/historical-maps/u-s-historical-airway-maps/": "/atc/airway-maps",
    "/historical-maps/u-s-historical-airway-maps/historical-maps-2/":
        "/atc/us-historical-airway-maps",
    "/facility-directories/directory-of-all-radio-fss-stations-between-1920-and-1955/":
        "/atc/fss-station-directory-1920-1955",
    "/feed/": "/atc/feed",
}

# Old paths that resolve to a status, not a canonical URI. Seeded with the WP
# front page's slug URL: live WP 301s /home/ -> / (configured static front
# page), and ?page_id=156 resolves here via p_map — the path never appears in
# the inventory, so without a seed it would mint a dead /atc/home/ (found
# 2026-08-20 building the redirect checker). Mirror live: one hop to landing.
DROP = {"/home/": PREFIX + "/"}
GONE_RE = re.compile(
    r"^/(wp-admin(/|$)|wp-login\.php|xmlrpc\.php|wp-cron\.php|wp-json(/|$)"
    r"|forum(/|$)|readme\.html$|backup(/|$)|ads\.txt$|___proxy_subdomain)")
# Author archives are the same posts as the site archive, addressed by a name
# the doc forbids putting in a URI. They redirect to the equivalent archive
# page rather than dying — the reader keeps their place.
AUTHOR_PAGE_RE = re.compile(r"^/author/[^/]+/page/(\d+)/?$")
AUTHOR_RE = re.compile(r"^/author/")
# Listing views that ARE real resources: the site archive and topic archives
# past page 1. 473 of them, and the landing page's "Older posts" leads here.
ROOT_PAGE_RE = re.compile(r"^/page/(\d+)/?$")
CAT_PAGE_RE = re.compile(r"^/category/(.+)/page/(\d+)/?$")

DOC_EXT = re.compile(r"\.(s?html?)$", re.I)
# WP appends -2..-9 when a slug is already taken. Require a letter in the stem
# so numeric post-id slugs (/11769-2/) are not "fixed" into bare id URIs.
COLLISION = re.compile(r"^(.*[a-z].*?)-([2-9])$")
JUNK_FILE = re.compile(r"/(\.DS_Store|Thumbs\.db)$")


def norm(path):
    """Collapse duplicate slashes; guarantee a leading slash."""
    return re.sub(r"/{2,}", "/", "/" + path.lstrip("/"))


def slugify(name):
    """Lowercase hyphenated slug, preserving a format extension."""
    stem, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 5 or not ext.isalnum():
        stem, ext = name, ""
    s = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    s = re.sub(r"-{2,}", "-", s)
    return f"{s}.{ext.lower()}" if ext else s


def apply_rules(path):
    for old, new in RULES:
        if path.startswith(old):
            return new + path[len(old):]
    return None


def despace(path):
    """Slugify only the path components that contain a space. Spaces survive
    a percent-encoding round trip badly and mod_security has already bitten us
    on them; everything else keeps its bytes (see docstring boundaries)."""
    parts = [slugify(p) if " " in p else p for p in path.split("/")]
    return "/".join(parts)


SITE = Path("/Volumes/projects/atchistory_build/site")
PROBE = ["", "/index.html", "/index.htm", "/Default.htm", ".htm", ".html"]


def build_has(old_path):
    """Is this inventory path actually in the build set? 179 of them are not --
    log ghosts, wp-admin, uncaptured WP core assets no public page references.
    Minting a canonical URI for a resource that does not exist would turn a
    clean 404 into a 301-to-404. If the build volume is not mounted the filter
    is skipped (map still generates, verify reports the difference)."""
    if not SITE.is_dir():
        return True
    q = old_path.lstrip("/")
    if (SITE / q).is_file():
        return True
    d = SITE / q.rstrip("/")
    return d.is_dir() and any((d / n).is_file()
                              for n in ("index.html", "index.htm", "Default.htm"))


def resolve_key(canon, keymap):
    """Canonical URI -> R2 key candidates, mirroring the worker exactly.
    Keep this in lockstep with keyOf() in worker-atc/src/index.js."""
    if canon in keymap:
        k = keymap[canon]
        if k.endswith("/"):                    # key names a directory
            return [k + n for n in ("index.html", "index.htm", "Default.htm")]
        return [k]
    inner = canon[len(PREFIX):] if canon.startswith(PREFIX) else canon
    for old, new in RULES:                     # reverse the prefix rules
        cn = new[len(PREFIX):]
        if inner.startswith(cn):
            inner = old + inner[len(cn):]
            break
    base = inner.lstrip("/").rstrip("/") if inner != "/" else ""
    if canon.endswith("/"):
        return [f"{base}/index.html" if base else "index.html",
                f"{base}/index.htm" if base else "index.htm",
                f"{base}/Default.htm" if base else "Default.htm"]
    return [base + suffix for suffix in PROBE]


def verify(payload, claimed, alias):
    """Prove every canonical URI resolves to a real file, and that no alias
    points at something that is itself an alias (a second hop)."""
    if not SITE.is_dir():
        print(f"verify: {SITE} not mounted — skipped")
        return 1
    keymap = payload["key"]
    unresolved, chained = [], []
    for canon in sorted(claimed):
        if any(canon.startswith(PREFIX + n) for n in payload["nocanon"]):
            continue
        if not any((SITE / c).is_file() for c in resolve_key(canon, keymap) if c):
            unresolved.append((canon, resolve_key(canon, keymap)[0]))
    for old, canon in alias.items():
        inner = canon[len(PREFIX):] or "/"
        for spelling in (inner, inner + "/", inner.rstrip("/")):
            if spelling != old and spelling in alias and alias[spelling] != canon:
                chained.append(f"{old} -> {canon} (but {spelling} -> "
                               f"{alias[spelling]})")
                break
    print(f"verify: {len(claimed)} canonical URIs, "
          f"{len(unresolved)} unresolved, {len(chained)} would chain")
    for c, k in unresolved[:25]:
        print(f"  UNRESOLVED {c}   (tried key: {k!r})")
    for c in chained[:25]:
        print(f"  CHAIN {c}")
    return 0 if not unresolved and not chained else 2


def classify(row):
    """-> ('doc' | 'asset' | 'listing' | 'skip')"""
    p, cls = row["old_path"], row["class"]
    if JUNK_FILE.search(p):
        return "skip"
    if cls.startswith("wp_posts") or cls.startswith("wp_taxonomies"):
        return "doc"
    if p.endswith("/"):
        return "listing"
    if DOC_EXT.search(p):
        return "doc"
    return "asset"


def main():
    rows = list(csv.DictReader(INVENTORY.open()))
    for r in rows:
        r["old_path"] = norm(r["old_path"])
    hits = {r["old_path"]: float(r["hits"] or 0) for r in rows}
    by_path = {r["old_path"]: r for r in rows}
    paths = sorted(by_path)
    claimed = set()          # canonical paths already taken
    alias = {}               # old_path -> canonical
    key = {}                 # canonical -> R2 key (only when not rule-derivable)
    dir_owner = {}           # dir-shaped canonical -> first old path claiming it
    notes = defaultdict(list)
    stats = Counter()

    # Workstream A2's legacy bonus map already routes 12 dead-on-live URLs to
    # living successors. Those decisions win: they are drops, not canonicals.
    for old, target in json.loads(LEGACY.read_text()).items():
        DROP[norm(old)] = target

    # Slugs occupied by the corpus, used to decide whether a "-2" strip is free.
    doc_slugs = {p.strip("/") for p in paths
                 if "/" not in p.strip("/") and p.endswith("/")}
    top_dirs = {p.strip("/").split("/")[0] for p in paths if "/" in p.strip("/")}

    # A WP "-2" suffix is only strippable when the post that took the bare slug
    # is gone AND no sibling variant also wants it. Decide up front: stripping
    # 4 variants of class-t-304-63 into one slug would just re-invent suffixes.
    want_bare = defaultdict(list)
    for p in paths:
        s = p.strip("/")
        if "/" in s or not p.endswith("/"):
            continue
        m = COLLISION.match(s)
        if m and m.group(1) not in doc_slugs and m.group(1) not in top_dirs:
            want_bare[m.group(1)].append(p)
    strip_to = {v[0]: bare for bare, v in want_bare.items() if len(v) == 1}
    for bare, v in want_bare.items():
        if len(v) > 1:
            notes["suffix kept — siblings contend for the bare slug"].append(
                f"{bare}: {', '.join(v)}")

    def take(canon, old):
        """Claim a canonical path. On collision fall back to the untransformed
        path rather than inventing a fresh -N suffix (that is the disease)."""
        if canon not in claimed:
            claimed.add(canon)
            return canon
        fallback = PREFIX + old.rstrip("/")
        stats["collision_fallback"] += 1
        notes["canonical collisions (fell back to old path)"].append(
            f"{old} wanted {canon} -> {fallback}")
        claimed.add(fallback)
        return fallback

    # Deterministic order: highest traffic first, so the busiest page wins a
    # contested slug.
    for old in sorted(paths, key=lambda p: (-hits.get(p, 0), p)):
        row = by_path[old]
        kind = classify(row)
        if kind == "skip":
            stats["skipped_junk"] += 1
            continue
        if old in DROP:                       # legacy map decided this one
            stats["legacy_map"] += 1
            continue
        if GONE_RE.match(old):
            DROP[old] = "410"
            stats["gone_410"] += 1
            continue
        m_author_page = AUTHOR_PAGE_RE.match(old)
        if m_author_page:                     # same posts, forbidden address
            DROP[old] = f"{PREFIX}/archive/{m_author_page.group(1)}"
            stats["author_page_to_archive"] += 1
            continue
        if AUTHOR_RE.match(old):
            DROP[old] = PREFIX + "/"
            stats["author_to_home"] += 1
            continue
        # Both spellings of the site feed survive (routes.js serves /atc/feed);
        # a bare "/feed" from the raw logs must not become a 410 drop.
        if old not in ("/feed", "/feed/") and re.search(r"/feed/?$", old):
            DROP[old] = "410"
            stats["gone_410"] += 1
            continue
        if any(old.startswith(n) for n in NOCANON) or old in NOCANON_FILES:
            stats["serve_only"] += 1          # reachable, never canonical
            continue
        if not build_has(old):
            stats["absent_from_build"] += 1
            notes["absent from the build set (no canonical minted)"].append(old)
            continue

        if old in MANUAL:
            canon = take(MANUAL[old], old)
            stats["manual"] += 1
        elif old == "/":
            canon, stats["root"] = PREFIX + "/", 1
            claimed.add(canon)
        elif ROOT_PAGE_RE.match(old):
            # The site archive. Page 1 was the WP homepage, which the designed
            # landing page replaced, so numbering starts where content does.
            canon = take(f"{PREFIX}/archive/{ROOT_PAGE_RE.match(old).group(1)}",
                         old)
            stats["archive_page"] += 1
        elif CAT_PAGE_RE.match(old):
            m = CAT_PAGE_RE.match(old)
            canon = take(f"{PREFIX}/topics/"
                         f"{slugify(m.group(1).split('/')[-1])}/{m.group(2)}",
                         old)
            stats["topic_page"] += 1
        elif old.startswith("/category/"):
            leaf = old.strip("/").split("/")[-1]
            canon = take(f"{PREFIX}/topics/{slugify(leaf)}", old)
            stats["topic"] += 1
        elif "/" not in old.strip("/") and old.endswith("/"):
            slug = old.strip("/")
            if slug in JUNK_SLUGS:
                DROP[old] = PREFIX + "/"
                stats["junk_page"] += 1
                continue
            if old in strip_to:
                slug = strip_to[old]          # the collision partner is gone
                stats["suffix_stripped"] += 1
                notes["collision suffixes stripped"].append(
                    f"{old} -> /atc/{slug}  ({int(hits.get(old, 0))} hits)")
            elif COLLISION.match(slug):
                stats["suffix_kept"] += 1
            canon = take(f"{PREFIX}/{slug}", old)
            stats["doc"] += 1
        else:
            base = despace(apply_rules(old) or PREFIX + old)
            if kind == "listing":
                canon = base                  # collections keep the slash
                dir_owner.setdefault(canon, old)
                if dir_owner[canon] != old:
                    notes["directory canonical collisions"].append(
                        f"{old} and {dir_owner[canon]} -> {canon}")
                claimed.add(canon)
                stats["listing"] += 1
            else:
                head, _, leaf = base.rpartition("/")
                if leaf.lower() in INDEX_NAMES:
                    # A directory index IS the directory. /x/index.htm must not
                    # canonicalize to /x/index — that URI names the mechanism.
                    # Shares the claim with the listing row for the same dir:
                    # one resource, two old spellings.
                    canon = head + "/"
                    claimed.add(canon)
                    stats["index_to_dir"] += 1
                else:
                    new_leaf = leaf
                    if DOC_EXT.search(leaf):  # document: drop server extension
                        new_leaf = slugify(DOC_EXT.sub("", leaf))
                        stats["extension_dropped"] += 1
                    canon = take(f"{head}/{new_leaf}", old)
                    stats["asset" if kind == "asset" else "doc"] += 1

        alias[old] = canon
        # An explicit key entry is needed whenever the canonical path is not
        # recoverable from the rules by prefix swap alone.
        derived = apply_rules(old) or (PREFIX + old)
        if canon != derived:
            key.setdefault(canon, old.lstrip("/"))   # highest-traffic wins

    # A2's legacy targets were written before canonical space existed, in the
    # old "/atc/<old path>" shape. Re-point them through the map, or every
    # legacy redirect lands on an alias and costs a second hop.
    for old, target in list(DROP.items()):
        if target == "410" or not target.startswith(PREFIX):
            continue
        inner = target[len(PREFIX):] or "/"
        resolved = alias.get(inner) or alias.get(inner.rstrip("/") + "/")
        if resolved is None and inner.rstrip("/"):
            resolved = alias.get(inner.rstrip("/"))
        if resolved:
            if resolved != target:
                stats["legacy_target_recanonicalized"] += 1
            DROP[old] = resolved
        else:
            derived = despace(apply_rules(inner) or PREFIX + inner)
            DROP[old] = derived
            notes["legacy targets not in the inventory (derived by rule)"].append(
                f"{old} -> {target} => {derived}")

    payload = {
        "_source": "scripts/atc_canonical_map.py",
        "_policy": "https://www.w3.org/Provider/Style/URI — aliases are "
                   "permanent; entries are never deleted.",
        "rules": [list(r) for r in RULES],
        "nocanon": list(NOCANON),
        # Published, not hidden: every canonical URI that still carries a WP
        # collision suffix, either because a sibling holds the bare slug or
        # because the stem is a bare post id (/atc/11769 would read as an
        # internal identifier — the same disease). The route test asserts the
        # canonical space is clean apart from exactly this list.
        "kept_suffix": sorted(c for c in claimed if re.search(r"-[2-9]$", c)),
        "alias": {k: v for k, v in sorted(alias.items())
                  if v != (apply_rules(k) or PREFIX + k)},
        "key": dict(sorted(key.items())),
        "drop": dict(sorted(DROP.items())),
    }
    OUT_MAP.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")

    with REPORT.open("w") as r:
        r.write("canonical URI map — worklist 08 workstream A\n")
        r.write("W3C https://www.w3.org/Provider/Style/URI\n\n")
        r.write(f"inventory rows: {len(rows)}  unique paths: {len(set(paths))}\n")
        r.write(f"canonical URIs claimed: {len(claimed)}\n")
        r.write(f"alias entries emitted: {len(payload['alias'])} "
                f"(rest are rule-derivable)\n")
        r.write(f"explicit key entries: {len(key)}\n")
        r.write(f"drops: {len(DROP)}\n\n")
        for k, v in sorted(stats.items()):
            r.write(f"  {v:6d}  {k}\n")
        for title, items in notes.items():
            r.write(f"\n== {title} ({len(items)}) ==\n")
            for line in items[:200]:
                r.write(f"  {line}\n")
        r.write("\n== top 40 canonical URIs by traffic ==\n")
        for old in sorted(alias, key=lambda p: -hits.get(p, 0))[:40]:
            r.write(f"  {int(hits.get(old,0)):>7}  {old}\n"
                    f"           -> {alias[old]}\n")

    print(f"{len(claimed)} canonical URIs; {len(payload['alias'])} alias "
          f"entries, {len(key)} key entries, {len(DROP)} drops")
    print(f"map    -> {OUT_MAP}")
    print(f"report -> {REPORT}")
    if "--verify" in sys.argv:
        return verify(payload, claimed, alias)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
