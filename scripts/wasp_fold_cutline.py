#!/usr/bin/env python3
"""Trace a folded-sheet scan's printed-face boundary into a per-row cutline_wkt.

A WASP-style scan of one face of a folded sectional carries, along the
fold edge, an unprinted margin and the dark scanner background beyond the
paper. The shared sectional outline (a graticule rectangle) cannot remove
it — the paper edge sits inside the outline and bows ~0.08 deg in latitude
(LCC curvature), so no straight cut works either (worklist 04, 2026-08-26).

Along the fold edge this walks inward column by column past the scanner background and the blank fold margin to the first
block with line work (clamped per column to the face's p20-p80 margin
band, so pale tint cannot push a column deep into the map), 20 px on
into the ink, maps the traced ring through the
row's own GCP affine (pixels -> chart CRS -> NAD83 lon/lat) and intersects
it with the row's referenced outline shapefile so the neat-line clip is
kept. The result is written to the row's `cutline_wkt` (cutline_wkt wins
over cutline in dole_v2.row_cutline) with a provenance note.

    ~/venv/bin/python scripts/wasp_fold_cutline.py WASP_02-2007-02-416_01.tif ... [--dry-run]

Rows must already be GCP-ready with `half` set. The catalog is written
through dole_v2.write_rows (backup first).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dole_v2  # noqa: E402

gdal.UseExceptions()
REPO = Path(__file__).resolve().parent.parent
RAWTIFFS = Path("/Volumes/projects/rawtiffs")

STEP = 64          # sample spacing along an edge, px
WIN = 20           # window length for the percentile test, px
DARK_P50 = 100     # window median below this = scanner background (~57)
INK_P10 = 170      # block p10 below this = printed chart (line work pulls
                   # p10 to ~135-155); blank fold margin sits at ~207
BLOCK = 64         # strip width sampled per edge point, px (2-D percentiles
                   # are what make the print edge robust to pale tint)
MAX_MARGIN = 200   # widest blank margin we'd believe past the paper edge, px
CLAMP_HI = 50      # upper clamp percentile: faces with a thin printed overlap gap past p50
BIAS = 20          # px pushed into the ink past the print edge
SMOOTH = 9         # running-median width over edge samples (kills spikes)
MAX_WALK = 1500    # deepest margin we'd believe, px


def find_source(filename: str) -> Path:
    hits = glob.glob(str(RAWTIFFS / "**" / filename), recursive=True)
    if not hits:
        raise SystemExit(f"{filename}: not under {RAWTIFFS}")
    return Path(hits[0])


def luminance_strip(ds, x0, y0, w, h):
    a = ds.ReadAsArray(x0, y0, w, h)[:3].astype(np.float32)
    if ds.GetRasterBand(1).DataType != gdal.GDT_Byte:
        a /= 257.0
    return 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]


def edges(strip):
    """(paper, print) row indices walking inward along axis 0 of a
    h x BLOCK luminance strip: where the scanner background ends (block
    median > DARK_P50) and where line work begins (first block clear of
    the ragged paper edge whose p10 < INK_P10). Either is None when not
    found within the strip."""
    n = strip.shape[0]
    paper = None
    for i in range(0, n - WIN, 4):
        blk = strip[i:i + WIN]
        if paper is None:
            if np.percentile(blk, 50) > DARK_P50:
                paper = i
            continue
        if np.percentile(blk, 2) < DARK_P50:
            continue        # block still straddles the ragged paper edge
        if np.percentile(blk, 10) < INK_P10:
            return paper, i
    return paper, None


def boundary(pairs, clamp_hi=None):
    """Per-sample boundary from (paper, print) pairs along the fold edge.

    The print edge is what must be cut at: the blank margin of whichever
    face the mosaic composites on top otherwise lies over the other
    face's print as a pale band. Per-column detection is right where line
    work reaches the margin and wrong where pale tint does (it walks deep
    into the map), so each column's measured margin is clamped to the
    face's own p20..p50 width band (CLAMP_HI) and the boundary is median-smoothed.
    Over-cutting inside that band is harmless: the faces' paper overlaps
    ~300-500 px at the fold, so the other face's print is underneath.
    Columns with no paper edge borrow the median of the rest."""
    papers = [p for p, _ in pairs if p is not None]
    widths = [q - p for p, q in pairs if p is not None and q is not None]
    fill = float(np.median(papers)) if papers else 0.0
    lo, mid, hi = (np.percentile(widths, (20, 50, clamp_hi or CLAMP_HI)) if widths else (0.0, 0.0, 0.0))
    print(f"    margin widths px: p20 {lo:.0f} p50 {mid:.0f} p80 {hi:.0f}")
    out = []
    for p, q in pairs:
        p = p if p is not None else fill
        wdt = (q - p) if q is not None else mid
        out.append(p + min(max(wdt, lo), hi) + BIAS)
    return out, mid


def smooth(vals):
    k = SMOOTH // 2
    return [float(np.median(vals[max(0, i - k):i + k + 1])) for i in range(len(vals))]


def trace(ds, fold_edge, clamp_hi=None):
    """Ring of raw-frame pixel points, clockwise from the top-left: the
    fold edge traced, the other three left at the image bounds (their
    neat lines are the outline's job; tracing them too cut 20 px inside
    the neat line and lost a sliver the outline would have kept)."""
    w, h = ds.RasterXSize, ds.RasterYSize
    half = BLOCK // 2
    if fold_edge in ("top", "bottom"):
        xs = list(range(half, w - half, STEP)); pairs = []
        for x in xs:
            strip = luminance_strip(ds, x - half, 0, BLOCK, h)
            pairs.append(edges((strip if fold_edge == "top" else strip[::-1])[:MAX_WALK]))
        vals, margin = boundary(pairs, clamp_hi); vals = smooth(vals)
        print(f"    fold edge {fold_edge}: blank margin {margin:.0f} px")
        if fold_edge == "top":
            return list(zip(xs, vals)) + [(w, 0), (w, h), (0, h)]
        return [(0, 0), (w, 0)] + list(zip(xs, [h - v for v in vals]))[::-1]
    ys = list(range(half, h - half, STEP)); pairs = []
    for y in ys:
        strip = luminance_strip(ds, 0, y - half, w, BLOCK).T
        pairs.append(edges((strip if fold_edge == "left" else strip[::-1])[:MAX_WALK]))
    vals, margin = boundary(pairs, clamp_hi); vals = smooth(vals)
    print(f"    fold edge {fold_edge}: blank margin {margin:.0f} px")
    if fold_edge == "left":
        return [(0, 0), (w, 0), (w, h)] + list(zip(vals, ys))[::-1]
    return [(0, 0), (w, 0)] + list(zip([w - v for v in vals], ys)) + [(0, h)]


def fold_edge_for(row, ds):
    """Which raw-frame image edge carries the fold: for an unrotated scan
    the south face folds along its top and the north face along its
    bottom (east/west faces likewise). Rotated scans need --fold-edge."""
    if dole_v2.row_rotation(row) != 0:
        raise SystemExit(f"{row['filename']}: rotation set; pass --fold-edge explicitly")
    return {"south": "top", "north": "bottom", "east": "left", "west": "right"}[dole_v2.row_half(row)]


def pixel_to_lonlat(row, pts):
    px = np.array(dole_v2.row_gcp_pixels(row))
    ll = dole_v2.row_gcp_lonlat(row)
    crs = dole_v2.row_lcc_crs(row)
    src = osr.SpatialReference(); src.ImportFromEPSG(4269)
    dst = osr.SpatialReference(); dst.SetFromUserInput(crs)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    fwd = osr.CoordinateTransformation(src, dst)
    inv = osr.CoordinateTransformation(dst, src)
    tgt = np.array([fwd.TransformPoint(lon, lat)[:2] for lon, lat in ll])
    A = np.c_[px, np.ones(len(px))]
    sol, *_ = np.linalg.lstsq(A, tgt, rcond=None)
    xy = np.c_[np.array(pts), np.ones(len(pts))] @ sol
    return [inv.TransformPoint(float(x), float(y))[:2] for x, y in xy]


def outline_geometry(row):
    """The row's `cutline` shapefile ring — read from that column directly,
    not through row_cutline, which prefers an existing cutline_wkt (a re-run
    would otherwise skip the neat-line clip)."""
    ref = str(row.get("cutline") or "").strip()
    if not ref or dole_v2.cutline_is_none(row):
        return None
    path = REPO / "shapefiles" / f"{ref}.shp"
    if not path.exists():
        raise SystemExit(f"{row.get('filename')}: cutline shapefile missing: {path}")
    ds = ogr.Open(str(path))
    feat = ds.GetLayer(0).GetNextFeature()
    return feat.GetGeometryRef().Clone()


def build_wkt(row, ds, fold_edge, clamp_hi=None):
    ring_px = trace(ds, fold_edge, clamp_hi)
    # GCP pixels are stored in the display frame; trace ran in the raw frame.
    rot = dole_v2.row_rotation(row)
    W, H = ds.RasterXSize, ds.RasterYSize
    ring_disp = [dole_v2.px_raw_to_display(x, y, rot, W, H) for x, y in ring_px]
    ring_ll = pixel_to_lonlat(row, ring_disp)
    poly = ogr.Geometry(ogr.wkbPolygon)
    r = ogr.Geometry(ogr.wkbLinearRing)
    for lon, lat in ring_ll:
        r.AddPoint_2D(lon, lat)
    r.AddPoint_2D(*ring_ll[0])
    poly.AddGeometry(r)
    if not poly.IsValid():
        poly = poly.Buffer(0)
    outline = outline_geometry(row)
    if outline is not None:
        poly = poly.Intersection(outline)
    if poly.GetGeometryType() == ogr.wkbMultiPolygon:
        poly = max((poly.GetGeometryRef(i).Clone() for i in range(poly.GetGeometryCount())),
                   key=lambda g: g.GetArea())
    poly = poly.SimplifyPreserveTopology(0.002)   # ~200 m; keeps the fold bow
    return poly.ExportToWkt(), poly.GetEnvelope()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("filenames", nargs="+")
    ap.add_argument("--csv", default=str(REPO / "master_dole_v2.csv"))
    ap.add_argument("--fold-edge", choices=["auto", "top", "bottom", "left", "right"], default="auto",
                    help="raw-frame image edge along the fold (default: from `half`, unrotated scans)")
    ap.add_argument("--clamp-hi", type=int, default=CLAMP_HI,
                    help="upper clamp percentile of the margin widths (default %(default)s; "
                         "use 20 for a pair whose georefs disagree at the seam — shallower cut, "
                         "a pale band beats a gap)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = dole_v2.load_rows(args.csv)
    by = {r["filename"]: r for r in rows}
    changed = 0
    for fn in args.filenames:
        row = by.get(fn)
        if row is None:
            raise SystemExit(f"{fn}: not in catalog")
        if not dole_v2.is_gcp_ready(row):
            raise SystemExit(f"{fn}: row is not GCP-ready")
        ds = gdal.Open(str(find_source(fn)))
        edge = fold_edge_for(row, ds) if args.fold_edge == "auto" else args.fold_edge
        wkt, env = build_wkt(row, ds, edge, args.clamp_hi)
        print(f"{fn}: half={dole_v2.row_half(row) or '-'} lon {env[0]:.3f}..{env[1]:.3f} "
              f"lat {env[2]:.3f}..{env[3]:.3f} ({len(wkt)} chars)")
        if args.dry_run:
            continue
        row["cutline_wkt"] = wkt
        stamp = (f"FOLD-CUTLINE {__import__('datetime').date.today().isoformat()}: cutline_wkt traced "
                 f"along the {edge} (fold) edge, clamp p20-p{args.clamp_hi} ({BLOCK}px-strip luminance walk to the paper edge + the print edge, clamped to the face's p20-p50 margin band, + {BIAS}px into ink) "
                 f"∩ {row.get('cutline')} outline; scripts/wasp_fold_cutline.py")
        # Re-runs replace the previous stamp instead of stacking them.
        note = " | ".join(part for part in (row.get("note") or "").split(" | ")
                          if not part.startswith("FOLD-CUTLINE "))
        row["note"] = (note + " | " if note else "") + stamp
        changed += 1
    if changed:
        print("backup:", dole_v2.write_rows(args.csv, rows, "fold_cutline"))


if __name__ == "__main__":
    main()
