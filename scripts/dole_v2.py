#!/usr/bin/env python3
"""
Canonical loader for the v2 dole schema (master_dole_v2.csv).

Schema (one row per map):
    identity:   filename, download_link, location, date, end_date, edition, note
    gcps:       gcp1..gcp4 (TL, TR, BR, BL) x (px, py, lat, lon)
                blank for already-georeferenced maps
    cutline:    cutline      - shapefile ref relative to shapefiles/
                               ("extents/aberdeen_sd", "sectional/new_york")
                cutline_wkt  - inline POLYGON((lon lat, ...)), lon/lat NAD83;
                               overrides `cutline` when present
    projection: lcc_lat1, lcc_lat2, lcc_lat0, lcc_lon0
                blank for already-georeferenced maps
    rotation:   degrees CLOCKWISE (0/90/180/270) the raw scan must be turned
                to read upright. OPTIONAL column (old CSVs load without it).
                gcp*_px/_py are stored in this rotated display frame; pipeline
                consumers map them back to the raw frame with px_display_to_raw
                before warping, so the raster itself is never resampled.
    src_crs:    CRS of a pre-georeferenced source whose file carries a
                geotransform but no (or wrong) CRS — e.g. jpg + world file,
                where GDAL reads the .JGW but never the .prj sidecar
                ("EPSG:4269", or any gdal-parseable SRS string). OPTIONAL
                column. rawtiffs holds sources exactly as downloaded; CRS
                knowledge belongs here, not in injected .aux.xml sidecars.
    half:       explicit half-sheet side ('north'/'south'/'east'/'west') for
                scans whose filename stem carries no trailing cardinal token
                (WASP _01/_02 recto/verso numbering, bare "-S" suffixes).
                Overrides the slicer's candidate-group split and chart-URI
                suffix. OPTIONAL column — and it MUST be listed in V2_FIELDS:
                writers sanitize rows to that header, so an optional column
                missing from the list is silently dropped on every rewrite
                (the 2026-08-27 sdcards import lost it exactly that way).

This module is GDAL-light: only cutline geometry reading needs osgeo.ogr,
imported lazily so metadata-only consumers can run without GDAL.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

V2_FIELDS = [
    "filename", "download_link", "location", "date", "end_date", "edition", "note",
    "gcp1_px", "gcp1_py", "gcp1_lat", "gcp1_lon",
    "gcp2_px", "gcp2_py", "gcp2_lat", "gcp2_lon",
    "gcp3_px", "gcp3_py", "gcp3_lat", "gcp3_lon",
    "gcp4_px", "gcp4_py", "gcp4_lat", "gcp4_lon",
    "cutline", "cutline_wkt",
    "lcc_lat1", "lcc_lat2", "lcc_lat0", "lcc_lon0",
    "rotation", "src_crs", "half",
]

# Columns that may be absent from a CSV on disk (added after the v2 freeze).
# Writers always emit the full V2_FIELDS header; readers treat these as "".
V2_OPTIONAL_FIELDS = {"rotation", "src_crs", "half"}

LCC_TEMPLATE = (
    "+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} "
    "+lon_0={lon0} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)

# Cutline geometry is authored in NAD83 lon/lat.
CUTLINE_SRS = "EPSG:4269"


def _fnum(value) -> Optional[float]:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def missing_required_fields(fieldnames) -> List[str]:
    """Required v2 columns absent from a CSV header (optionals excluded)."""
    present = set(fieldnames or [])
    return [k for k in V2_FIELDS
            if k not in present and k not in V2_OPTIONAL_FIELDS]


def load_rows(csv_path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = missing_required_fields(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path}: not a v2 dole CSV, missing {missing}")
        return list(reader)


def row_gcp_pixels(row) -> Optional[List[Tuple[float, float]]]:
    """4 pixel-space corners (TL, TR, BR, BL) or None if incomplete."""
    pixels = []
    for corner in range(1, 5):
        px = _fnum(row.get(f"gcp{corner}_px"))
        py = _fnum(row.get(f"gcp{corner}_py"))
        if px is None or py is None:
            return None
        pixels.append((px, py))
    return pixels


def row_gcp_lonlat(row) -> Optional[List[Tuple[float, float]]]:
    """4 geographic corners as (lon, lat), NAD83, or None if incomplete."""
    coords = []
    for corner in range(1, 5):
        lat = _fnum(row.get(f"gcp{corner}_lat"))
        lon = _fnum(row.get(f"gcp{corner}_lon"))
        if lat is None or lon is None:
            return None
        coords.append((lon, lat))
    return coords


def row_lcc_crs(row) -> str:
    """Proj4 LCC string synthesized from the lcc_* columns, or ''."""
    parts = {
        "lat1": str(row.get("lcc_lat1") or "").strip(),
        "lat2": str(row.get("lcc_lat2") or "").strip(),
        "lat0": str(row.get("lcc_lat0") or "").strip(),
        "lon0": str(row.get("lcc_lon0") or "").strip(),
    }
    if not all(parts.values()):
        return ""
    return LCC_TEMPLATE.format(**parts)


def row_cutline(row, shape_dir) -> Optional[Dict[str, object]]:
    """
    Resolve the row's cutline. Returns:
        {"kind": "wkt", "wkt": str}                        - inline override
        {"kind": "shapefile", "path": Path, "ref": str}    - shapefile ref
        None                                               - no cutline
    cutline_wkt wins over cutline when both are present.
    """
    wkt = str(row.get("cutline_wkt") or "").strip()
    if wkt:
        return {"kind": "wkt", "wkt": wkt}
    ref = str(row.get("cutline") or "").strip()
    if ref:
        path = Path(shape_dir) / f"{ref}.shp"
        return {"kind": "shapefile", "path": path, "ref": ref}
    return None


def cutline_ring(cutline, shape_dir=None) -> Optional[List[Tuple[float, float]]]:
    """
    Outer ring of a cutline as a closed list of (lon, lat). Works for the
    inline-WKT kind and for extent shapefiles (single polygon). Requires GDAL.
    """
    if cutline is None:
        return None
    from osgeo import ogr

    if cutline["kind"] == "wkt":
        geom = ogr.CreateGeometryFromWkt(cutline["wkt"])
        if geom is None:
            raise ValueError(f"unparseable cutline_wkt: {cutline['wkt']!r}")
    else:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.Open(str(cutline["path"]), 0)
        if ds is None:
            raise FileNotFoundError(f"cutline shapefile missing: {cutline['path']}")
        layer = ds.GetLayer(0)
        feature = layer.GetNextFeature()
        if feature is None:
            raise ValueError(f"cutline shapefile empty: {cutline['path']}")
        geom = feature.GetGeometryRef().Clone()
        ds = None

    if geom.GetGeometryName() != "POLYGON":
        raise ValueError(f"cutline is not a polygon: {geom.GetGeometryName()}")
    ring = geom.GetGeometryRef(0)
    return [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]


# --- ROTATION -----------------------------------------------------------
# A scanned chart may be stored sideways/upside-down. `rotation` records the
# clockwise turn (90/180/270) that makes it upright. All pixel-space GCPs are
# authored against the rotated ("display") image; these helpers convert
# between that frame and the raw file's frame. Coordinates are continuous
# GDAL pixel/line values with the origin at the image's top-left corner.

def row_src_crs(row) -> str:
    """Declared CRS of a pre-georeferenced source file, or '' (trust the
    file's own embedded CRS)."""
    return str(row.get("src_crs") or "").strip()


def row_half(row) -> str:
    """Explicit half-sheet side ('north'/'south'/'east'/'west') or ''.

    For half-sheet scans whose filenames carry no cardinal token (e.g. the
    WASP _01/_02 recto/verso numbering — scan order, not sides), this column
    is the slicer's grouping/URI override. Leave empty on multi-file zip rows
    whose members name their own halves."""
    value = str(row.get("half") or "").strip().lower()
    return value if value in ("north", "south", "east", "west") else ""


def row_rotation(row) -> int:
    """Row's display rotation in degrees clockwise: 0, 90, 180 or 270."""
    value = str(row.get("rotation") or "").strip()
    if not value:
        return 0
    try:
        rot = int(float(value)) % 360
    except ValueError:
        return 0
    return rot if rot in (90, 180, 270) else 0


def rotated_dims(raw_w: float, raw_h: float, rotation: int) -> Tuple[float, float]:
    """(width, height) of the raw image after rotating `rotation` degrees CW."""
    if rotation in (90, 270):
        return raw_h, raw_w
    return raw_w, raw_h


def px_raw_to_display(x: float, y: float, rotation: int,
                      raw_w: float, raw_h: float) -> Tuple[float, float]:
    """Map a raw-frame pixel into the frame rotated `rotation` degrees CW."""
    if rotation == 90:
        return raw_h - y, x
    if rotation == 180:
        return raw_w - x, raw_h - y
    if rotation == 270:
        return y, raw_w - x
    return x, y


def px_display_to_raw(dx: float, dy: float, rotation: int,
                      raw_w: float, raw_h: float) -> Tuple[float, float]:
    """Inverse of px_raw_to_display: display-frame pixel -> raw-frame pixel.

    raw_w/raw_h are always the RAW file's dimensions (pre-rotation)."""
    if rotation == 90:
        return dy, raw_h - dx
    if rotation == 180:
        return raw_w - dx, raw_h - dy
    if rotation == 270:
        return raw_w - dy, dx
    return dx, dy


def is_gcp_ready(row) -> bool:
    """True when the row has everything needed for a GCP warp."""
    return (
        row_gcp_pixels(row) is not None
        and row_gcp_lonlat(row) is not None
        and bool(row_lcc_crs(row))
        and (str(row.get("cutline_wkt") or "").strip()
             or str(row.get("cutline") or "").strip()) != ""
    )


# ---------------------------------------------------------------------------
# Raw-frame image access (the Pillow TIFF orientation trap)
#
# Pillow applies TIFF Orientation (tag 274) when it decodes; GDAL does not.
# Finder's rotate is tag-only. The catalog `rotation` column is the sole
# truth for how a scan is turned: the DISPLAY frame is the raw pixel grid
# (what GDAL sees) rotated `rotation` degrees clockwise, gcp*_px/py are
# authored and stored in that display frame, and the slicer maps them back
# to the raw frame with px_display_to_raw. Any tool that shows a scan must
# therefore open it raw (undoing Pillow's auto-transpose) and then apply the
# row's rotation, or its points live in a frame the slicer never constructs.
# Shared here so the georef tool and the timeline preview GUI cannot drift
# apart. PIL is imported lazily: metadata-only consumers never need it.

TIFF_ORIENTATION_TAG = 274


def _pil_image():
    from PIL import Image, TiffImagePlugin
    # Pillow 12's internal TIFF decoder scrambles UNCOMPRESSED files carrying
    # orientation 5-8 (verified 2026-09-08); libtiff reads them correctly.
    TiffImagePlugin.READ_LIBTIFF = True
    return Image


def tiff_orientation(img) -> Optional[int]:
    tags = getattr(img, "tag_v2", None)
    try:
        return tags.get(TIFF_ORIENTATION_TAG) if tags is not None else None
    except Exception:
        return None


def orientation_undo_op(orientation: Optional[int]):
    """PIL transpose that undoes a decoded TIFF orientation, or None."""
    Image = _pil_image()
    table = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_90,    # inverse of exif_transpose's ROTATE_270
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_270,   # inverse of exif_transpose's ROTATE_90
    }
    return table.get(orientation)


def open_raw(path):
    """Image.open in the RAW pixel frame (TIFF Orientation tag undone)."""
    Image = _pil_image()
    img = Image.open(path)
    op = orientation_undo_op(tiff_orientation(img))
    # `is not None`, not truthiness: FLIP_LEFT_RIGHT is enum value 0, and an
    # `if op:` test silently left orientation 2 (mirror) applied.
    return img.transpose(op) if op is not None else img


def raw_size(path) -> Tuple[int, int]:
    """(w, h) of the raw pixel grid, without decoding pixels."""
    Image = _pil_image()
    with Image.open(path) as img:
        w, h = img.size
        if tiff_orientation(img) in (5, 6, 7, 8):
            return h, w
    return w, h


def rotate_for_display(img, rotation: int):
    """Rotate a raw-frame PIL image `rotation` degrees clockwise (the frame
    the slicer builds when it applies the row's `rotation` to its GCPs)."""
    Image = _pil_image()
    op = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
          270: Image.Transpose.ROTATE_90}.get(rotation)
    return img.transpose(op) if op is not None else img


# ---------------------------------------------------------------------------
# Catalog identity for importers
#
# Filename equality proves nothing about whether a chart is already held:
# the same edition lives under many names (NARA ca#####r.tif, FAA
# "Washington SEC 97.tif", a salvage stem). Importers diff on
# (location, edition) — edition numbers are per chart type — and, when the
# edition is unknown, on (location, date). Same date|location rows are
# alternates of one chart by design, so "already present" means: a row for
# this location with the same numeric edition, or the same date.

def norm_location(location) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(location or "").lower()).strip()


def _numeric_edition(value) -> Optional[str]:
    v = str(value or "").strip()
    return v if re.fullmatch(r"\d+", v) else None


def find_same_chart(rows: List[Dict[str, str]], location: str, date: str = "",
                    edition: str = "") -> List[Dict[str, str]]:
    """Rows already cataloguing this chart: same location and (same numeric
    edition, or same date). Empty list when the chart is new."""
    loc = norm_location(location)
    ed = _numeric_edition(edition)
    date = str(date or "").strip()
    hits = []
    for r in rows:
        if norm_location(r.get("location")) != loc:
            continue
        if ed and _numeric_edition(r.get("edition")) == ed:
            hits.append(r)
        elif date and str(r.get("date") or "").strip() == date:
            hits.append(r)
    return hits


def rechain_predecessor(rows: List[Dict[str, str]], location: str, new_date: str) -> List[Dict[str, str]]:
    """Close the era of the edition that precedes `new_date` at `location`.

    Every row of the immediately preceding date (all its alternates) whose
    end_date is empty or later than new_date gets end_date = new_date, so
    inserting an edition never leaves two overlapping eras (the slicer keys
    mosaics on date_to_end_date). Returns the rows changed.
    """
    loc = norm_location(location)
    new_date = str(new_date or "").strip()
    prior_dates = sorted({str(r.get("date") or "").strip()
                          for r in rows
                          if norm_location(r.get("location")) == loc
                          and str(r.get("date") or "").strip()
                          and str(r.get("date") or "").strip() < new_date})
    if not prior_dates:
        return []
    prev = prior_dates[-1]
    changed = []
    for r in rows:
        if norm_location(r.get("location")) != loc or str(r.get("date") or "").strip() != prev:
            continue
        end = str(r.get("end_date") or "").strip()
        if not end or end > new_date:
            r["end_date"] = new_date
            changed.append(r)
    return changed


# ---------------------------------------------------------------------------
# Catalog writer (atomic, backed up, row-count guarded)

def write_rows(csv_path, rows: List[Dict[str, str]], backup_tag: str,
               backup_dir=None) -> str:
    """Write `rows` to `csv_path` with the canonical V2_FIELDS header.

    Backs the current file up to ~/archive.aero-attic/csv-backups/
    pre_<backup_tag>_<YYYY-MM-DD>.csv first (the required practice), writes
    a sibling temp file, verifies the row count, then renames it into place.
    Never truncates the live file. Returns the backup path.
    """
    import os
    import shutil
    from datetime import date
    csv_path = str(csv_path)
    backup_dir = str(backup_dir or os.path.expanduser("~/archive.aero-attic/csv-backups"))
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, f"pre_{backup_tag}_{date.today().isoformat()}.csv")
    if os.path.exists(backup):
        n = 2
        while os.path.exists(f"{backup[:-4]}_{n}.csv"):
            n += 1
        backup = f"{backup[:-4]}_{n}.csv"
    if os.path.exists(csv_path):
        shutil.copy2(csv_path, backup)
    tmp = f"{csv_path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=V2_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in V2_FIELDS})
    with open(tmp, "r", newline="", encoding="utf-8-sig") as f:
        written = sum(1 for _ in csv.DictReader(f))
    if written != len(rows):
        os.remove(tmp)
        raise RuntimeError(f"refusing to save: wrote {written} rows, expected {len(rows)}")
    os.replace(tmp, csv_path)
    return backup
