#!/usr/bin/env python3
"""
Build timeline_data.json for the chart-inventory timeline prototypes.

Reads master_dole_v2.csv (via dole_v2) plus the extent/sectional shapefiles
and emits one JSON blob with:

  - locations: per-location chart list (date, end_date, edition, source url,
    flags), gap spans parsed from the GAP notes, and missing edition numbers
    inferred from holes in ascending edition sequences
  - groups: modern sectional -> member old-era charts, computed spatially
    (intersection area of the old extent vs every modern sectional extent)
  - simplified extent rings (NAD83 lon/lat) for the map prototype

Usage:  ~/venv/bin/python build_timeline_data.py   (needs GDAL for geometry)
"""

import collections
import json
import re
from pathlib import Path

from osgeo import ogr, osr

import dole_v2

ROOT = Path(__file__).resolve().parent.parent
SHAPE_DIR = ROOT / "shapefiles"
OUT = ROOT / "timeline_data.json"

GAP_RE = re.compile(r"GAP (\d+)d before next \((\d{4}-\d{2}-\d{2})\)")
TYP_RE = re.compile(r"typ(\d+)d")


def load_shape_geom(ref):
    """Union of all features in shapefiles/<ref>.shp as one ogr geometry,
    reprojected to NAD83 lon/lat when the shapefile is in a projected CRS."""
    path = SHAPE_DIR / f"{ref}.shp"
    ds = ogr.GetDriverByName("ESRI Shapefile").Open(str(path), 0)
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayer(0)
    srs = layer.GetSpatialRef()
    transform = None
    if srs is not None and srs.IsProjected():
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4269)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs = srs.Clone()
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(srs, dst)
    geom = None
    for feature in layer:
        g = feature.GetGeometryRef().Clone()
        if transform is not None:
            g.Transform(transform)
        geom = g if geom is None else geom.Union(g)
    ds = None
    return geom


def geom_rings(geom, max_pts=48):
    """Outer ring(s) of a (multi)polygon, decimated and rounded for the web."""
    polys = []
    if geom.GetGeometryName() == "MULTIPOLYGON":
        polys = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
    else:
        polys = [geom]
    rings = []
    for poly in polys:
        ring = poly.GetGeometryRef(0)
        n = ring.GetPointCount()
        step = max(1, n // max_pts)
        pts = [
            (round(ring.GetX(i), 3), round(ring.GetY(i), 3))
            for i in range(0, n, step)
        ]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        rings.append(pts)
    return rings


def parse_int(s):
    try:
        return int(str(s).strip())
    except ValueError:
        return None


def main():
    rows = dole_v2.load_rows(ROOT / "master_dole_v2.csv")

    # ---- group rows by location, pick the majority cutline ref per location
    by_loc = collections.defaultdict(list)
    for r in rows:
        if r["date"]:
            by_loc[r["location"]].append(r)

    loc_ref = {}
    for loc, lrows in by_loc.items():
        refs = collections.Counter(
            r["cutline"].strip() for r in lrows if r["cutline"].strip()
        )
        loc_ref[loc] = refs.most_common(1)[0][0] if refs else None

    # not-yet-georeferenced charts with no cutline anywhere: place by hand
    # (Key West 1928-35 sits on the modern Miami sectional)
    HAND_GROUP = {"Key West": "sectional/miami"}

    # ---- geometry per unique ref
    geoms = {}
    for ref in sorted({v for v in loc_ref.values() if v}):
        try:
            geoms[ref] = load_shape_geom(ref)
        except FileNotFoundError:
            print(f"WARN missing shapefile: {ref}")

    # ---- spatial grouping: every old (extents/) ref -> best modern
    # (sectional/) ref by share of the old extent's area
    # antimeridian-crossing extents (Aleutians) degenerate into
    # globe-spanning lon/lat polygons — exclude them from spatial matching
    def wraps(g):
        x0, x1, _, _ = g.GetEnvelope()
        return (x1 - x0) > 90

    modern_refs = [r for r in geoms
                   if r.startswith("sectional/") and not wraps(geoms[r])]
    old_to_modern = {}
    for ref, g in geoms.items():
        if not ref.startswith("extents/"):
            continue
        area = g.GetArea()
        best, best_share, shares = None, 0.0, []
        for mref in modern_refs:
            inter = g.Intersection(geoms[mref])
            share = (inter.GetArea() / area) if inter and area else 0.0
            if share > 0.02:
                shares.append((mref, round(share, 3)))
            if share > best_share:
                best, best_share = mref, share
        shares.sort(key=lambda t: -t[1])
        old_to_modern[ref] = {"best": best, "share": round(best_share, 3),
                              "all": shares[:4]}

    # ---- per-location payload
    locations = {}
    for loc, lrows in by_loc.items():
        lrows.sort(key=lambda r: r["date"])
        ref = loc_ref[loc]
        if ref:
            era = "old" if ref.startswith("extents/") else "modern"
        else:
            era = "old" if max(r["date"] for r in lrows) < "1972" else "modern"

        charts, gaps = [], []
        for r in lrows:
            note = r["note"] or ""
            ed = r["edition"].strip()
            charts.append({
                "d": r["date"],
                "e": r["end_date"] or None,
                "ed": ed,
                "url": r["download_link"],
                "f": ("L" if "latest-in-collection" in note else "")
                     + ("B" if "BLANK_MAP" in note else ""),
            })
            m = GAP_RE.search(note)
            if m:
                days, nxt = int(m.group(1)), m.group(2)
                t = TYP_RE.search(note)
                typ = int(t.group(1)) if t else None
                est = max(1, round(days / typ) - 1) if typ else None
                gaps.append({
                    "from": r["end_date"] or r["date"],
                    "to": nxt,
                    "days": days,
                    "est": est,
                })

        # missing edition numbers from holes in ascending int sequences
        eds = [(parse_int(c["ed"]), c["d"]) for c in charts]
        eds = [(e, d) for e, d in eds if e is not None and e < 1500]
        missing_eds = []
        for (e0, d0), (e1, d1) in zip(eds, eds[1:]):
            if e0 < e1 - 1 and (e1 - e0) < 40:
                missing_eds.extend(range(e0 + 1, e1))
        locations[loc] = {
            "era": era,
            "ref": ref,
            "charts": charts,
            "gaps": gaps,
            "missing_eds": sorted(set(missing_eds)),
        }

    # ---- groups keyed by modern sectional ref
    groups = collections.defaultdict(lambda: {"modern": [], "old": []})
    for loc, info in locations.items():
        ref = info["ref"]
        if info["era"] == "modern":
            groups[ref]["modern"].append(loc)
        else:
            g = old_to_modern.get(ref) or {}
            key = g.get("best") or HAND_GROUP.get(loc) or "__unmatched__"
            groups[key]["old"].append(loc)
            info["modern_share"] = g.get("share")
            info["overlaps"] = g.get("all")

    out_groups = []
    for gref, members in sorted(groups.items(), key=lambda kv: kv[0] or ""):
        name = members["modern"][0] if members["modern"] else \
            gref.split("/")[-1].replace("_", " ").title()
        out_groups.append({
            "ref": gref,
            "name": name,
            "modern": sorted(members["modern"]),
            "old": sorted(members["old"]),
        })

    rings = {ref: geom_rings(g) for ref, g in geoms.items()}

    payload = {
        "generated": "2026-07-15",
        "note": "built by build_timeline_data.py from master_dole_v2.csv",
        "locations": locations,
        "groups": out_groups,
        "rings": rings,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    n_charts = sum(len(v["charts"]) for v in locations.values())
    n_gaps = sum(len(v["gaps"]) for v in locations.values())
    n_missing = sum(len(v["missing_eds"]) for v in locations.values())
    print(f"{OUT.name}: {len(locations)} locations, {n_charts} charts, "
          f"{n_gaps} gap spans, {n_missing} missing editions, "
          f"{len(out_groups)} groups, {OUT.stat().st_size/1e6:.2f} MB")
    for g in out_groups:
        if g["old"] and not g["modern"]:
            print(f"  group with no modern lane: {g['ref']} <- {g['old']}")
        if g["ref"] == "__unmatched__":
            print(f"  unmatched old charts: {g['old']}")


if __name__ == "__main__":
    main()
