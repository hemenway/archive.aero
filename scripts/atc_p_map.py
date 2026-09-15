#!/Users/ryanhemenway/venv/bin/python
"""Extract post_id -> permalink-path map from an UpdraftPlus DB dump, for the
atc worker's `?p=N` / `?page_id=N` redirect rule (worklist 08 §2.3).

WP shortlinks like atchistory.org/?p=8123 circulate in old forum posts and
emails; the worker 301s them to /atc/<slug>/. Regenerate from the freeze-day
dump before cutover.

Also extracts wp_postmeta `_wp_old_slug` rows -> old_slugs.json. WP keeps a
post's previous slugs there and 301s them on the live site; the static archive
has no such lookup, so scripts/atc_canonical_map.py turns them into drops
(found 2026-09-14: Search Console held two renamed airway-map slugs as 404s).

Usage: atc_p_map.py [dump.gz]   (default: newest *-db.gz in the backup updraft dir)
"""
import gzip
import json
import re
import sys
from pathlib import Path

UPDRAFT = Path("/Volumes/projects/atchistory_backup/full_mirror/public_html"
               "/wp-content/updraft")
OUT = Path("/Users/ryanhemenway/archive.aero/worklists/data/atc/p_map.json")
OUT_SLUGS = OUT.with_name("old_slugs.json")

dump = (Path(sys.argv[1]) if len(sys.argv) > 1
        else max(UPDRAFT.glob("*-db.gz"), key=lambda p: p.stat().st_mtime))
print(f"reading {dump.name}")

# wp_posts columns: ID, post_author, post_date, post_date_gmt, post_content,
# post_title, post_excerpt, post_status, comment_status, ping_status,
# post_password, post_name, ... post_type is the 21st column. Row-tuple regex
# parsing of INSERT statements is fragile against embedded quotes in
# post_content, so instead scan value-tuples with a stateful splitter.

text = gzip.open(dump, "rt", encoding="utf-8", errors="replace").read()
inserts = re.findall(r"INSERT INTO `wp_posts`[^(]*VALUES\s*(.+?);\n",
                     text, re.S)
print(f"wp_posts INSERT statements: {len(inserts)}")


def split_tuples(block):
    """Yield top-level (...) tuples from a VALUES blob, respecting quotes."""
    depth, in_str, esc, start = 0, False, False, None
    for i, ch in enumerate(block):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
        elif ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                yield block[start + 1:i]
                start = None


def split_fields(tup):
    """Split one tuple into raw fields, respecting quotes."""
    out, cur, in_str, esc = [], [], False, False
    for ch in tup:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
            cur.append(ch)
        elif ch == ",":
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


rows, skipped = {}, 0
for block in inserts:
    for tup in split_tuples(block):
        f = split_fields(tup)
        if len(f) < 21:
            skipped += 1
            continue
        pid, status, name = f[0], f[7].strip("'"), f[11].strip("'")
        parent, ptype = f[17], f[20].strip("'")
        rows[pid] = (status, name, parent, ptype)

# Pages are hierarchical: the permalink is the parent-chain of slugs
# (/historical-maps/u-s-historical-airway-maps/historical-maps-2/), so walk
# post_parent. Posts use root-level slugs.
pmap = {}
for pid, (status, name, parent, ptype) in rows.items():
    if status != "publish" or ptype not in ("post", "page") or not name:
        continue
    segs, cur, hops = [name], parent, 0
    while ptype == "page" and cur not in ("0", "") and cur in rows and hops < 10:
        segs.insert(0, rows[cur][1])
        cur = rows[cur][2]
        hops += 1
    pmap[pid] = "/" + "/".join(segs) + "/"

OUT.write_text(json.dumps(pmap, indent=0, sort_keys=True))
n_nested = sum(1 for v in pmap.values() if v.count("/") > 2)
print(f"published post/page ids mapped: {len(pmap)} ({n_nested} nested page "
      f"paths; skipped {skipped} malformed tuples) -> {OUT}")

# Renamed slugs: wp_postmeta (meta_id, post_id, meta_key, meta_value) rows with
# meta_key `_wp_old_slug`. Keyed by the old path in permalink shape so the
# canonical map can treat them exactly like the legacy map's entries.
meta_inserts = re.findall(r"INSERT INTO `wp_postmeta`[^(]*VALUES\s*(.+?);\n",
                          text, re.S)
old_slugs, orphans = {}, 0
for block in meta_inserts:
    for tup in split_tuples(block):
        f = split_fields(tup)
        if len(f) < 4 or f[2].strip("'") != "_wp_old_slug":
            continue
        pid, slug = f[1], f[3].strip("'")
        if pid not in pmap or not slug:
            orphans += 1                     # unpublished post, or empty
            continue
        old_slugs[f"/{slug}/"] = pmap[pid]
OUT_SLUGS.write_text(json.dumps(old_slugs, indent=0, sort_keys=True) + "\n")
print(f"renamed slugs mapped: {len(old_slugs)} (skipped {orphans} on "
      f"unpublished posts) -> {OUT_SLUGS}")
