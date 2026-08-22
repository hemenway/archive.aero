#!/usr/bin/env python3
"""Extract a per-airfield index (name, coords, deep-link URL) from a
www.airfields-freeman.com mirror, for the archive.aero map.

Page structure (stable across the site since ~2002): each regional page holds
several airfields, separated by long ``____`` rules. An entry is:

    <A NAME="anchor">                       (24-ish per page, most entries)
    Newtok Airport (EWU / WWT), Newtok, AK  (title, may wrap)
    60.938, -164.64 (West of Anchorage, AK) (decimal, or "25.86 North / 80.21
                                             West" on a few old pages)

The extractor rebuilds the text stream with source offsets for <a name>
anchors, splits on the underscore rules, takes the FIRST plausible coordinate
match per segment as the entry header, the text before it as the title, and
the nearest preceding anchor as the deep link (#anchor).

Outputs GeoJSON (map-ready) + CSV, plus a QA summary on stdout.

Usage:
  airfields_freeman_extract.py --tree .../www.airfields-freeman.com \
      --out-geojson airfields.geojson --out-csv airfields.csv
"""
import argparse
import csv
import json
import os
import posixpath
import re

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, Tag

BASE_URL = "https://www.airfields-freeman.com/"

SEP_RE = re.compile(r"_{20,}")
# integer degrees occur ("39.79, -86"); digit-boundary guards keep years etc. out
DEC_RE = re.compile(r"(?<![\d.])(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-\d{1,3}(?:\.\d+)?)(?![\d.])")
NW_RE = re.compile(
    r"(\d{1,2}\.\d+)\s*(?:North|N\.?)\s*/\s*(\d{1,3}\.\d+)\s*(?:(West|W\.?)|(East|E\.?))",
    re.IGNORECASE,
)
REVISED_RE = re.compile(r"\(?\s*revised\s+[\d/]+\s*\)?\s*$", re.IGNORECASE)
WS_RE = re.compile(r"\s+")

SKIP_STRING_TYPES = (Comment, Declaration, Doctype)


def walk_page(soup):
    """Return (text, anchors) where text mirrors '\n'.join(stripped_strings)
    and anchors is [(offset_in_text, anchor_name), ...]."""
    parts = []
    length = 0
    anchors = []
    for el in soup.descendants:
        if isinstance(el, Tag):
            if el.name == "a" and el.get("name"):
                anchors.append((length, el["name"].strip()))
        elif isinstance(el, NavigableString) and not isinstance(el, SKIP_STRING_TYPES):
            s = str(el).strip()
            if s:
                if parts:
                    length += 1  # the joining '\n'
                parts.append(s)
                length += len(s)
    return "\n".join(parts), anchors


def plausible(lat, lon):
    return 15.0 <= lat <= 75.0 and (60.0 <= abs(lon) <= 180.0)


def bridge_digits(s):
    """FrontPage wraps lines inside numbers ("42.2\n24, -87.954" = 42.224).
    Remove whitespace runs between digits when the token before the gap is
    "digits.digits" or "digits." AND the continuation is a bare integer run
    (not itself followed by '.'). So "42.2\n24" and "35.\n28" fuse, while
    "1944\n42.19" and a sentence-ending "...95.\n28.36, -81.52" do not.
    Returns (bridged, posmap) with posmap[bridged_idx] = original_idx."""
    out = []
    posmap = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            k = len(out) - 1
            tok = []
            while k >= 0 and (out[k].isdigit() or out[k] == "."):
                tok.append(out[k])
                k -= 1
            tok = "".join(reversed(tok))
            k2 = j
            while k2 < n and s[k2].isdigit():
                k2 += 1
            cont_starts_decimal = k2 < n and s[k2] == "."
            if (
                j < n
                and s[j].isdigit()
                and not cont_starts_decimal
                and re.fullmatch(r"\d+\.\d*", tok)
            ):
                i = j  # drop the gap
                continue
        out.append(c)
        posmap.append(i)
        i += 1
    return "".join(out), posmap


def find_coords(segment):
    """All plausible coordinate matches in segment text, sorted by position.
    Each hit is (pos, end, lat, lon, style); offsets are into `segment`."""
    bridged, posmap = bridge_digits(segment)

    def orig_span(m):
        return posmap[m.start()], posmap[m.end() - 1] + 1

    hits = []
    for m in DEC_RE.finditer(bridged):
        lat, lon = float(m.group(1)), float(m.group(2))
        if plausible(lat, lon):
            hits.append((*orig_span(m), lat, lon, "decimal"))
    for m in NW_RE.finditer(bridged):
        lat, lon = float(m.group(1)), float(m.group(2))
        if m.group(4):  # East
            pass
        else:
            lon = -lon
        if plausible(lat, lon):
            hits.append((*orig_span(m), lat, lon, "north_west"))
    hits.sort()
    return hits


def clean_title(raw):
    t = WS_RE.sub(" ", raw).strip(" \t-–—_:;")
    t = REVISED_RE.sub("", t).strip(" \t-–—_:;")
    return t


def extract_page(path, rel_url, keep_segment=False):
    html = open(path, encoding="utf-8", errors="replace").read()
    soup = BeautifulSoup(html, "lxml")
    text, anchors = walk_page(soup)

    bounds = [0]
    for m in SEP_RE.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))

    entries = []
    n_multi = n_nocoords = 0
    for seg_start, seg_end in zip(bounds, bounds[1:]):
        segment = text[seg_start:seg_end]
        hits = find_coords(segment)
        # a stray coordinate pair sometimes sits right after the separator
        # (leftover of the previous entry): take the first match that has an
        # actual title before it
        hit = None
        title = ""
        cursor = 0
        for h in hits:
            t = clean_title(segment[cursor : h[0]].lstrip("_ \n"))
            if any(c.isalpha() for c in t):
                hit, title = h, t
                break
            cursor = h[1]  # skip the stray pair; title region resumes after it
        if hit is None:
            n_nocoords += 1
            continue
        # one page glues a superseded pair right before the corrected one
        # ("...FL39.497, -75.645\n28.35, -81.55 (West of Orlando, FL)"): the
        # real pair is the adjacent one carrying the (relative location) paren
        def has_paren(h):
            return bool(re.match(r"\s*\(", segment[h[1] :]))

        if not has_paren(hit):
            for h in hits:
                if hit[1] < h[0] <= hit[1] + 120 and has_paren(h):
                    hit = h
                    break
        if len(hits) > 1:
            n_multi += 1
        n_hits = len(hits)
        pos, end, lat, lon, style = hit
        if len(title) > 300:
            title = title[-300:].lstrip()  # keep the tail nearest the coords
        m = re.match(r"\s*\(([^)]{0,140})\)", segment[end:])
        rel_loc = WS_RE.sub(" ", m.group(1)).strip() if m else ""
        coord_abs = seg_start + pos
        anchor = ""
        for off, name in anchors:
            if off <= coord_abs:
                if off >= seg_start - 120:
                    anchor = name
            else:
                break
        url = BASE_URL + rel_url + (f"#{anchor}" if anchor else "")
        entries.append(
            {
                **({"_segment": segment} if keep_segment else {}),
                "name": title,
                "lat": lat,
                "lon": lon,
                "state": rel_url.split("/")[0] if "/" in rel_url else "",
                "page": rel_url,
                "anchor": anchor,
                "url": url,
                "rel_location": rel_loc,
                "coord_style": style,
                "extra_coord_matches": n_hits - 1,
            }
        )
    return entries, n_nocoords, n_multi


def dedup_entries(tree, entries):
    """During Freeman's page splits the same entry appears on both the old
    (menu-orphaned) page and its replacement, and occasionally twice on one
    page. Keep one row per (state, name): prefer pages still linked from the
    state index, then rows with an anchor, then stable page order."""
    by_key = {}
    order = []
    for e in entries:
        by_key.setdefault((e["state"], e["name"]), []).append(e)
        order.append((e["state"], e["name"]))
    index_links = {}

    def canonical(state, page):
        if state not in index_links:
            links = set()
            idx = os.path.join(tree, state, f"Airfields_{state}.htm")
            if os.path.exists(idx):
                html = open(idx, encoding="utf-8", errors="replace").read()
                links = {m.group(1).rsplit("/", 1)[-1].lower()
                         for m in re.finditer(r'href="([^"#]+\.html?)"', html, re.I)}
            index_links[state] = links
        return page.rsplit("/", 1)[-1].lower() in index_links[state]

    dropped = 0
    out = []
    seen = set()
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        group = by_key[key]
        if len(group) > 1:
            linked = [e for e in group if canonical(key[0], e["page"])]
            pool = linked or group
            pool.sort(key=lambda e: (0 if e["anchor"] else 1, e["page"]))
            dropped += len(group) - 1
            out.append(pool[0])
        else:
            out.append(group[0])
    return out, dropped


def iter_pages(tree_root):
    for dirpath, dirnames, filenames in os.walk(tree_root):
        dirnames.sort()
        for f in sorted(filenames):
            if f.lower().endswith((".htm", ".html")):
                yield os.path.join(dirpath, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True, help="mirror root (the www.airfields-freeman.com dir)")
    ap.add_argument("--out-geojson", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    all_entries = []
    n_pages = tot_nocoords = tot_multi = 0
    for path in iter_pages(tree):
        rel = os.path.relpath(path, tree)
        rel_url = posixpath.join(*rel.split(os.sep))
        if rel_url == "index.html" and os.path.exists(os.path.join(tree, "index.htm")):
            continue  # duplicate of index.htm saved by wget for "/"
        entries, n_nocoords, n_multi = extract_page(path, rel_url)
        all_entries.extend(entries)
        n_pages += 1
        tot_nocoords += n_nocoords
        tot_multi += n_multi

    all_entries, n_dropped = dedup_entries(tree, all_entries)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]},
            "properties": {k: e[k] for k in ("name", "state", "url", "page", "anchor", "rel_location")},
        }
        for e in all_entries
    ]
    with open(args.out_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_entries[0].keys()))
        w.writeheader()
        w.writerows(all_entries)

    n_anchor = sum(1 for e in all_entries if e["anchor"])
    states = len({e["state"] for e in all_entries})
    print(
        f"{n_pages} pages -> {len(all_entries)} airfields with coords "
        f"({n_anchor} with deep-link anchors, {states} states/territories, "
        f"{n_dropped} cross-page duplicates dropped); "
        f"segments without coords: {tot_nocoords}; segments with extra coord matches: {tot_multi}"
    )
    print(f"geojson: {args.out_geojson}\ncsv: {args.out_csv}")


if __name__ == "__main__":
    main()
