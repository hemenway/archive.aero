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
]

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


def load_rows(csv_path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [k for k in V2_FIELDS if k not in (reader.fieldnames or [])]
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


def is_gcp_ready(row) -> bool:
    """True when the row has everything needed for a GCP warp."""
    return (
        row_gcp_pixels(row) is not None
        and row_gcp_lonlat(row) is not None
        and bool(row_lcc_crs(row))
        and (str(row.get("cutline_wkt") or "").strip()
             or str(row.get("cutline") or "").strip()) != ""
    )
