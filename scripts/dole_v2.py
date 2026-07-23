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
    "rotation",
]

# Columns that may be absent from a CSV on disk (added after the v2 freeze).
# Writers always emit the full V2_FIELDS header; readers treat these as "".
V2_OPTIONAL_FIELDS = {"rotation"}

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
