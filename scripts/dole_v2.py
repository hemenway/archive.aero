#!/usr/bin/env python3
"""
Canonical loader for the v2 dole schema (master_dole_v2.csv).

Schema (one row per map):
    identity:   filename, download_link, location, date, end_date, edition, note
    gcps:       gcp1..gcp4 (TL, TR, BR, BL) x (px, py, lat, lon)
                blank for already-georeferenced maps
    cutline:    cutline      - shapefile ref relative to shapefiles/
                               ("extents/aberdeen_sd", "sectional/new_york"),
                               or the sentinel "none" (CUTLINE_NONE): the row
                               has NO cutline by decision and the whole warped
                               sheet, collar included, goes into the mosaic.
                               Blank means "not decided yet" (row incomplete).
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
    crop:       "x0,y0,x1,y1" - the part of the scan that is the sheet, in
                the same full-resolution rotated display frame as gcp*_px/_py
                (origin top-left, x1/y1 exclusive). Everything outside it is
                overscan (scanner bed, lid, a neighbouring sheet) and never
                reaches a warp: the slicer reads only this window, for the
                mosaic and for the full-sheet chart artifact alike, and the
                cutline then trims inside it. GCP pixels stay in whole-image
                coordinates, so setting or clearing a crop never moves them.
                OPTIONAL column; blank = the whole image. GCP rows only.

This module is GDAL-light: only cutline geometry reading needs osgeo.ogr,
imported lazily so metadata-only consumers can run without GDAL.
"""

import csv
import math
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
    "rotation", "src_crs", "half", "proj", "crop",
]

# Columns that may be absent from a CSV on disk (added after the v2 freeze).
# Writers always emit the full V2_FIELDS header; readers treat these as "".
V2_OPTIONAL_FIELDS = {"rotation", "src_crs", "half", "proj", "crop"}

LCC_TEMPLATE = (
    "+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} "
    "+lon_0={lon0} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)

# `proj` (optional, 2026-09-15): the chart's own projection family, which
# is what the affine GCP fit must be computed in. Blank = Lambert conformal
# conic from the lcc_* columns (every FAA sectional). `merc` = Mercator
# about lcc_lon0 (the 1928-35 Key West Navy strips: their four corners fit
# Mercator to 30-80 m and LCC 45/33 to ~1 km; lcc_lat1/lat2/lat0 are
# ignored). The georef tool's fit check still assumes LCC, so it warns on
# these rows — "Save anyway" is right.
PROJ_MERCATOR = "merc"
MERC_TEMPLATE = (
    "+proj=merc +lon_0={lon0} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
)

# Cutline geometry is authored in NAD83 lon/lat.
CUTLINE_SRS = "EPSG:4269"

# `cutline` sentinel: no cutline by decision. The slicer warps the whole
# sheet (collar included) into the mosaic, exactly like the per-chart
# full-sheet artifacts, for sheets no rectangle or sectional outline fits
# (the 1928-35 Key West Navy strip charts, GlidePlan's collar-free mosaics).
# Distinct from a BLANK cutline, which means "not decided yet" and keeps the
# row incomplete. Set from the georef tool's Cutline panel.
CUTLINE_NONE = "none"


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
    """Proj4 string for the row's chart projection, or ''.

    LCC from the lcc_* columns unless `proj` names another family
    (PROJ_MERCATOR: Mercator about lcc_lon0)."""
    proj = str(row.get("proj") or "").strip().lower()
    if proj == PROJ_MERCATOR:
        lon0 = str(row.get("lcc_lon0") or "").strip()
        return MERC_TEMPLATE.format(lon0=lon0) if lon0 else ""
    if proj:
        raise ValueError(f"{row.get('filename')}: unknown proj {proj!r}")
    parts = {
        "lat1": str(row.get("lcc_lat1") or "").strip(),
        "lat2": str(row.get("lcc_lat2") or "").strip(),
        "lat0": str(row.get("lcc_lat0") or "").strip(),
        "lon0": str(row.get("lcc_lon0") or "").strip(),
    }
    if not all(parts.values()):
        return ""
    return LCC_TEMPLATE.format(**parts)


# --- Corner-GCP plausibility -------------------------------------------------
# The slicer fits an affine from the four pixel corners to the corners
# projected into the row's chart projection (polynomialOrder=1, LCC metres).
# A scan has square pixels, so that affine should map a pixel square to a
# ground square: its two singular values should be equal, whatever the scan's
# rotation. The Boston 1953-70 rows (44-40N sheet, corners copied from the
# older 44-41N one) fit at 0.76 and squeezed the sheet north by up to 0.7 deg
# for 17 years; residual alone could not see it (the four points still fit an
# affine). Thresholds: good hand-GCP rows fit to 31-270 m RMS, a wrong
# projection to 0.8-1.3 km (worklist 04, 2026-09-15); old paper shrinks
# unevenly by about a percent.
GCP_FIT_WARN = {"aspect": 0.03, "rms_m": 500.0}
GCP_FIT_ERROR = {"aspect": 0.10, "rms_m": 5000.0}

# NAD83 = GRS80 ellipsoid (the +datum=NAD83 of the templates above).
_GRS80_A = 6378137.0
_GRS80_E = math.sqrt((1 / 298.257222101) * (2 - 1 / 298.257222101))


def _lon_delta(lon, lon0) -> float:
    """lon - lon0 in radians, wrapped to [-pi, pi) (Aleutian sheets cross 180)."""
    d = (lon - lon0 + 180.0) % 360.0 - 180.0
    return math.radians(d)


def _iso_t(phi: float) -> float:
    s = _GRS80_E * math.sin(phi)
    return math.tan(math.pi / 4 - phi / 2) / ((1 - s) / (1 + s)) ** (_GRS80_E / 2)


def row_projector(row):
    """(lon, lat) -> (x, y) metres in the row's chart projection, or None.

    The projection row_lcc_crs names, evaluated in pure Python (Snyder's
    ellipsoidal LCC 2SP and Mercator formulas, matching PROJ to well under a
    millimetre) so the check runs where GDAL is not installed."""
    proj = str(row.get("proj") or "").strip().lower()
    lon0 = _fnum(row.get("lcc_lon0"))
    if lon0 is None:
        return None
    a, e = _GRS80_A, _GRS80_E
    if proj == PROJ_MERCATOR:
        def merc(lon, lat):
            phi = math.radians(lat)
            s = e * math.sin(phi)
            y = a * math.log(math.tan(math.pi / 4 + phi / 2) * ((1 - s) / (1 + s)) ** (e / 2))
            return a * _lon_delta(lon, lon0), y
        return merc
    if proj:
        raise ValueError(f"{row.get('filename')}: unknown proj {proj!r}")
    lat1, lat2, lat0 = (_fnum(row.get(k)) for k in ("lcc_lat1", "lcc_lat2", "lcc_lat0"))
    if None in (lat1, lat2, lat0):
        return None
    p1, p2, p0 = (math.radians(v) for v in (lat1, lat2, lat0))

    def m(phi):
        return math.cos(phi) / math.sqrt(1 - (e * math.sin(phi)) ** 2)

    if abs(p1 - p2) < 1e-12:
        n = math.sin(p1)
    else:
        n = (math.log(m(p1)) - math.log(m(p2))) / (math.log(_iso_t(p1)) - math.log(_iso_t(p2)))
    big_f = m(p1) / (n * _iso_t(p1) ** n)
    rho0 = a * big_f * _iso_t(p0) ** n

    def lcc(lon, lat):
        rho = a * big_f * _iso_t(math.radians(lat)) ** n
        theta = n * _lon_delta(lon, lon0)
        return rho * math.sin(theta), rho0 - rho * math.cos(theta)
    return lcc


def gcp_fit(row) -> Optional[Dict[str, float]]:
    """Least-squares affine of the four corner GCPs, pixel -> projected metres.

    Returns None when the row has no complete GCPs or projection. Keys:
      aspect       smaller / larger singular value of the fit (1 = a pixel
                   square lands as a ground square); rotation-invariant
      scale_ratio  metres per pixel down / across the display frame, and
      shear_deg    how far the pixel axes are from perpendicular on the
                   ground: where the distortion lies (diagnostic only)
      mirrored     True when the corners are in mirror order (det > 0: pixel y
                   runs down while northing runs up, so a correct fit has det < 0)
      rms_m/max_m  corner residuals of the fit, metres
      m_per_px     mean ground metres per pixel
    """
    pixels = row_gcp_pixels(row)
    lonlat = row_gcp_lonlat(row)
    if not (pixels and lonlat):
        return None
    project = row_projector(row)
    if project is None:
        return None
    ground = [project(lon, lat) for lon, lat in lonlat]
    # Centred pixels make the constant term the ground centroid and leave a
    # 2x2 system for the linear part: X = cx0 + a1*u + a2*v (and Y with b).
    cx = sum(p[0] for p in pixels) / 4
    cy = sum(p[1] for p in pixels) / 4
    pts = [(px - cx, py - cy) for px, py in pixels]
    suu = sum(u * u for u, _ in pts)
    svv = sum(v * v for _, v in pts)
    suv = sum(u * v for u, v in pts)
    den = suu * svv - suv * suv
    if suu <= 0 or svv <= 0 or den <= 1e-9 * suu * svv:
        return {"degenerate": True}
    coef = []
    for k in range(2):
        g0 = sum(g[k] for g in ground) / 4
        sug = sum(u * (g[k] - g0) for (u, _), g in zip(pts, ground))
        svg = sum(v * (g[k] - g0) for (_, v), g in zip(pts, ground))
        coef.append((g0, (sug * svv - svg * suv) / den, (svg * suu - sug * suv) / den))
    (cx0, a1, a2), (cy0, b1, b2) = coef
    sx, sy = math.hypot(a1, b1), math.hypot(a2, b2)
    det = a1 * b2 - a2 * b1
    if sx == 0 or sy == 0:
        return {"degenerate": True}
    cos_axes = max(-1.0, min(1.0, (a1 * a2 + b1 * b2) / (sx * sy)))
    resid = [math.hypot(cx0 + a1 * u + a2 * v - gx, cy0 + b1 * u + b2 * v - gy)
             for (u, v), (gx, gy) in zip(pts, ground)]
    frob = a1 * a1 + a2 * a2 + b1 * b1 + b2 * b2
    root = math.sqrt(max(0.0, frob * frob - 4 * det * det))
    s_big, s_small = math.sqrt((frob + root) / 2), math.sqrt(max(0.0, (frob - root) / 2))
    return {
        "aspect": s_small / s_big,
        "scale_ratio": sy / sx,
        "shear_deg": math.degrees(math.asin(abs(cos_axes))),
        "mirrored": det > 0,
        "rms_m": math.sqrt(sum(r * r for r in resid) / 4),
        "max_m": max(resid),
        "m_per_px": math.sqrt(abs(det)),
    }


def gcp_fit_problems(row) -> Tuple[str, List[str]]:
    """('ok' | 'warn' | 'error', reasons) for a row's corner GCPs.

    'ok' with no reasons also covers rows with nothing to check (no GCPs or
    no projection: is_gcp_ready reports those)."""
    try:
        fit = gcp_fit(row)
    except ValueError as e:
        return "error", [str(e)]
    if fit is None:
        return "ok", []
    if fit.get("degenerate"):
        return "error", ["corner GCPs are degenerate (collinear or repeated pixels)"]
    errors, warns = [], []
    if fit["mirrored"]:
        errors.append("corners are in mirror order (check TL/TR/BR/BL and rotation)")
    off = 1 - fit["aspect"]
    msg = (f"aspect {fit['aspect']:.3f} (y/x {fit['scale_ratio']:.3f}, shear {fit['shear_deg']:.1f} deg): "
           f"the corners' ground span does not match their pixel span "
           f"(a wrong corner lat/lon, or GCPs from another sheet format)")
    (errors if off > GCP_FIT_ERROR["aspect"] else warns if off > GCP_FIT_WARN["aspect"] else []).append(msg)
    msg = (f"corner residual {fit['rms_m']:.0f} m RMS (max {fit['max_m']:.0f} m): "
           f"a misplaced corner, or the wrong projection/standard parallels")
    (errors if fit["rms_m"] > GCP_FIT_ERROR["rms_m"]
     else warns if fit["rms_m"] > GCP_FIT_WARN["rms_m"] else []).append(msg)
    if errors:
        return "error", errors + warns
    return ("warn", warns) if warns else ("ok", [])


def row_cutline(row, shape_dir) -> Optional[Dict[str, object]]:
    """
    Resolve the row's cutline. Returns:
        {"kind": "wkt", "wkt": str}                        - inline override
        {"kind": "shapefile", "path": Path, "ref": str}    - shapefile ref
        {"kind": "none", "ref": "none"}                    - CUTLINE_NONE:
                                                             full sheet, no mask
        None                                               - undecided (blank)
    cutline_wkt wins over cutline when both are present.
    """
    wkt = str(row.get("cutline_wkt") or "").strip()
    if wkt:
        return {"kind": "wkt", "wkt": wkt}
    ref = str(row.get("cutline") or "").strip()
    if ref == CUTLINE_NONE:
        return {"kind": "none", "ref": ref}
    if ref:
        path = Path(shape_dir) / f"{ref}.shp"
        return {"kind": "shapefile", "path": path, "ref": ref}
    return None


def cutline_is_none(row) -> bool:
    """True when the row opts out of a cutline (CUTLINE_NONE, no WKT override)."""
    return (str(row.get("cutline") or "").strip() == CUTLINE_NONE
            and not str(row.get("cutline_wkt") or "").strip())


def cutline_ring(cutline, shape_dir=None) -> Optional[List[Tuple[float, float]]]:
    """
    Outer ring of a cutline as a closed list of (lon, lat). Works for the
    inline-WKT kind and for extent shapefiles (single polygon). Requires GDAL.
    None for an undecided (blank) cutline and for the "none" kind alike: a
    full-sheet row has no ring to clip or sanity-check against.
    """
    if cutline is None or cutline["kind"] == "none":
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


# Chart type (2026-10-04): TACs, WACs and planning charts are catalogued
# beside the sectionals and sliced into the same era mosaics. The type is
# carried by the LOCATION suffix ("New Orleans TAC", "CF-16 WAC", "Flight
# Case Planning Chart") rather than a column: the suffix already has to be
# there so a TAC and the sectional of the same city and date get different
# candidate groups, cutline votes and chart URIs, and a location-derived type
# cannot be dropped by a writer that sanitizes rows to an older header.
# Edition numbers are per chart type - never match on (city, edition) alone.
CHART_TYPE_SECTIONAL = "sectional"
_CHART_TYPE_SUFFIXES = (
    (" planning chart", "planning"),
    (" wac", "wac"),
    (" tac", "tac"),
)
# Mosaic stacking, bottom to top. gdalwarp composites last-source-wins, so
# where charts of one era overlap the WAC lies under the sectional and the
# TAC on top (smaller scale below larger scale).
CHART_TYPE_LAYER = {"planning": 0, "wac": 1, CHART_TYPE_SECTIONAL: 2, "tac": 3}


def location_chart_type(location) -> str:
    """'tac' / 'wac' / 'planning' from a location's suffix, else 'sectional'."""
    loc = str(location or "").strip().lower()
    for suffix, kind in _CHART_TYPE_SUFFIXES:
        if loc.endswith(suffix):
            return kind
    return CHART_TYPE_SECTIONAL


def row_chart_type(row) -> str:
    return location_chart_type(row.get("location"))


def row_layer(row) -> int:
    """Mosaic stacking rank of the row's chart type (higher = drawn later)."""
    return CHART_TYPE_LAYER[row_chart_type(row)]


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


# --- CROP ---------------------------------------------------------------
# An overscanned sheet carries scanner bed around the paper. `crop` boxes the
# paper in the display frame (the frame the GCPs are authored in); consumers
# take the raw-frame window from row_crop_raw_window and shift their raw GCP
# pixels by its offset. This is the one place the storage format is known.

def format_crop(x0: float, y0: float, x1: float, y1: float) -> str:
    """Catalog text for a display-frame crop box (whole pixels)."""
    return ",".join(str(int(round(v))) for v in (x0, y0, x1, y1))


def row_crop(row) -> Optional[Tuple[float, float, float, float]]:
    """Row's crop box (x0, y0, x1, y1) in display-frame pixels, or None.

    Raises ValueError on a value that is present but unusable: a crop that
    silently fell back to the whole image would put the overscan back."""
    value = str(row.get("crop") or "").strip()
    if not value:
        return None
    parts = [_fnum(p) for p in value.split(",")]
    if len(parts) != 4 or any(p is None for p in parts):
        raise ValueError(f"{row.get('filename')}: crop must be 'x0,y0,x1,y1', got {value!r}")
    x0, y0, x1, y1 = parts
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"{row.get('filename')}: crop is empty or inverted: {value!r}")
    return x0, y0, x1, y1


def crop_display_to_raw(crop, rotation: int, raw_w: float, raw_h: float
                        ) -> Tuple[float, float, float, float]:
    """A display-frame box as (x0, y0, x1, y1) in the raw file's frame."""
    ax, ay = px_display_to_raw(crop[0], crop[1], rotation, raw_w, raw_h)
    bx, by = px_display_to_raw(crop[2], crop[3], rotation, raw_w, raw_h)
    return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)


def crop_raw_to_display(box, rotation: int, raw_w: float, raw_h: float
                        ) -> Tuple[float, float, float, float]:
    """Inverse of crop_display_to_raw."""
    ax, ay = px_raw_to_display(box[0], box[1], rotation, raw_w, raw_h)
    bx, by = px_raw_to_display(box[2], box[3], rotation, raw_w, raw_h)
    return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)


def row_crop_raw_window(row, raw_w: int, raw_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Row's crop as a GDAL source window (xoff, yoff, xsize, ysize) in the
    RAW file's frame, clamped to the raster, or None when the row has no crop
    or the crop covers the whole image. Raises ValueError when the crop
    misses the raster entirely."""
    crop = row_crop(row)
    if crop is None:
        return None
    x0, y0, x1, y1 = crop_display_to_raw(crop, row_rotation(row), raw_w, raw_h)
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(int(raw_w), int(round(x1))), min(int(raw_h), int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{row.get('filename')}: crop {row.get('crop')!r} lies outside "
                         f"the {raw_w}x{raw_h} image")
    if (x0, y0, x1, y1) == (0, 0, int(raw_w), int(raw_h)):
        return None
    return x0, y0, x1 - x0, y1 - y0


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
    """Image.open in the RAW pixel frame (TIFF Orientation tag undone).

    Pixels are decoded here, not lazily, so a file whose strips run past
    EOF surfaces now instead of inside a resize. Such files fall back to
    `open_truncated`, which returns the rows that exist over a white sheet
    (ca000815, Chicago 1939: LOC's own master stops at row 4,548 of 7,069)."""
    Image = _pil_image()
    img = Image.open(path)
    orientation = tiff_orientation(img)
    try:
        img.load()
    except OSError as exc:
        img.close()
        img = open_truncated(path, exc)
    op = orientation_undo_op(orientation)
    # `is not None`, not truthiness: FLIP_LEFT_RIGHT is enum value 0, and an
    # `if op:` test silently left orientation 2 (mirror) applied.
    return img.transpose(op) if op is not None else img


def open_truncated(path, cause=None):
    """The decodable top of a TIFF whose pixel data ends early, as a PIL
    image of the full declared size (missing rows white). GDAL reads the
    strips that exist; the last good row is found by bisection. Raises the
    original error when not even the first row decodes."""
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()
    Image = _pil_image()
    ds = gdal.Open(str(path))
    if ds is None:
        raise cause or OSError(f"GDAL cannot open {path}")
    w, h, bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount

    def readable(y0, n):
        try:
            ds.ReadRaster(0, y0, w, n, band_list=[1])
            return True
        except RuntimeError:
            return False

    # Coarse walk down the sheet, then bisect inside the failing chunk.
    step, good = 512, 0
    while good < h and readable(good, min(step, h - good)):
        good += step
    good = min(good, h)
    lo, hi = good, min(good + step, h)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if readable(lo, mid - lo):
            lo = mid
        else:
            hi = mid - 1
    rows = lo
    if rows <= 0:
        raise cause or OSError(f"{path}: no decodable rows")
    data = ds.ReadAsArray(0, 0, w, rows)
    if bands == 1:
        arr = data
    else:
        arr = np.moveaxis(data, 0, -1)
    if ds.GetRasterBand(1).GetColorTable() is not None:
        lut = ds.GetRasterBand(1).GetColorTable()
        pal = np.array([lut.GetColorEntry(i)[:3] for i in range(256)], dtype=np.uint8)
        arr = pal[arr]
    ds = None
    print(f"[dole_v2] {Path(path).name}: pixel data ends at row {rows} of {h}; "
          f"padding the rest white ({cause})")
    sheet = np.full((h, w) + arr.shape[2:], 255, dtype=arr.dtype)
    sheet[:rows] = arr
    return Image.fromarray(sheet)


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


def check_gcps(rows: List[Dict[str, str]]) -> List[Tuple[str, Dict[str, str], Optional[Dict], List[str]]]:
    """(level, row, fit, reasons) for every row with corner GCPs, worst first."""
    order = {"error": 0, "warn": 1, "ok": 2}
    out = []
    for row in rows:
        if not (row_gcp_pixels(row) and row_gcp_lonlat(row)):
            continue
        level, reasons = gcp_fit_problems(row)
        try:
            fit = gcp_fit(row)
        except ValueError:
            fit = None
        out.append((level, row, fit, reasons))
    out.sort(key=lambda r: (order[r[0]], norm_location(r[1].get("location")), r[1].get("date") or ""))
    return out


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Checks on a v2 dole CSV.",
        epilog="check-gcps exits 1 when any row is an error (the slicer refuses those rows).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-gcps", help="corner-GCP plausibility of every GCP row: aspect, mirror order, residual")
    c.add_argument("csv", type=Path, help="master_dole_v2.csv")
    c.add_argument("--all", action="store_true", help="list passing rows too")
    c.add_argument("--out", type=Path, help="also write every checked row to this CSV")
    args = ap.parse_args(argv)

    results = check_gcps(load_rows(args.csv))
    counts = {k: sum(1 for r in results if r[0] == k) for k in ("error", "warn", "ok")}

    def num(fit, key, fmt):
        return format(fit[key], fmt) if fit and key in fit else "-"

    shown = [r for r in results if args.all or r[0] != "ok"]
    if shown:
        print(f"{'level':5}  {'aspect':>6}  {'y/x':>5}  {'shear':>5}  {'rms_m':>6}  "
              f"{'date':10}  {'location':24}  filename / reason")
    for level, row, fit, reasons in shown:
        print(f"{level:5}  {num(fit, 'aspect', '.3f'):>6}  {num(fit, 'scale_ratio', '.3f'):>5}  "
              f"{num(fit, 'shear_deg', '.1f'):>5}  {num(fit, 'rms_m', '.0f'):>6}  "
              f"{(row.get('date') or '')[:10]:10}  {(row.get('location') or '')[:24]:24}  {row.get('filename')}")
        for reason in reasons:
            print(f"{'':58}- {reason}")
    print(f"\n{len(results)} GCP rows: {counts['error']} error, {counts['warn']} warn, {counts['ok']} ok "
          f"(thresholds: aspect off by >{GCP_FIT_WARN['aspect']:.0%} warn / >{GCP_FIT_ERROR['aspect']:.0%} error, "
          f"RMS >{GCP_FIT_WARN['rms_m']:.0f} m warn / >{GCP_FIT_ERROR['rms_m']:.0f} m error, mirror order error)")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["level", "aspect", "scale_ratio", "shear_deg", "rms_m", "max_m", "m_per_px",
                        "mirrored", "date", "location", "filename", "reasons"])
            for level, row, fit, reasons in results:
                fit = fit or {}
                w.writerow([level] + [("" if fit.get(k) is None else
                                       (f"{fit[k]:.4f}" if isinstance(fit[k], float) else fit[k]))
                                      for k in ("aspect", "scale_ratio", "shear_deg", "rms_m", "max_m",
                                                "m_per_px", "mirrored")]
                           + [row.get("date"), row.get("location"), row.get("filename"), " | ".join(reasons)])
        print(f"wrote {args.out}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
