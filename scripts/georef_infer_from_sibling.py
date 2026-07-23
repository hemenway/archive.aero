#!/usr/bin/env python3
"""
Georef transfer v2 - robust to false graticule locks, small scale drift and
layout offsets.

Per target/sibling pair:
  1. 9 sibling windows (3x3 grid), each seeded independently by NCC of its
     1/8-scale template over the whole 1/8-scale target
  2. full-res NCC refine (1024px template, +-160px search)
  3. RANSAC similarity fit  target_px = s * sib_px + T  (no rotation)
  4. corner GCPs from the fit via the sibling's LCC grid
Emits georef_transfer2.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/ryanhemenway/archive.aero/scripts")
from osgeo import gdal, osr, ogr
gdal.UseExceptions()
import dole_v2
from slicer import ChartSlicer

RAW = Path("/Volumes/projects/rawtiffs")
SCRATCH = Path("/private/tmp/claude-501/-Users-ryanhemenway-archive-aero/979396e4-d26f-44e2-a132-01bc48223900/scratchpad")
CSV = Path("/Users/ryanhemenway/archive.aero/master_dole_v2.csv")
TARGET_DATES = {"2025-11-27", "2026-01-22", "2026-03-19", "2015-10-15"}
NARA_TARGETS = {
    "s3.amazonaws.com_NARAprodstorage_lz_electronic-records_rg-237_VFR_2019-0060_Los_Angeles_SEC_101.tif",
    "s3.amazonaws.com_NARAprodstorage_lz_electronic-records_rg-237_VFR_2019-0060_Washington_SEC_101.tif",
}
ZIP_COVERED_0319 = {"Albuquerque", "Bethel", "Billings", "Brownsville", "Cheyenne", "Dallas-Ft_Worth",
                    "Dawson", "Denver", "Detroit", "Fairbanks", "Great_Falls", "Houston", "Jacksonville",
                    "Juneau", "Los_Angeles", "McGrath", "Montreal", "San_Francisco", "Seattle", "Wichita"}

DEC = 8
s = ChartSlicer(RAW, SCRATCH / "out", CSV, Path("/Users/ryanhemenway/archive.aero/shapefiles"))
s._build_source_index()
s._build_shapefile_index()
zip_covered_norm = {s.normalize_name(n.replace("_", " ")) for n in ZIP_COVERED_0319}


_gray_cache = {}


def _gray_ds(ds):
    """Dataset for luminance reads: palette-indexed rasters (the FAA
    sectional-files GeoTIFFs) get their color table expanded to RGB first -
    raw palette indices don't correlate with an RGB sibling under NCC."""
    key = ds.GetDescription()
    if key not in _gray_cache:
        if ds.GetRasterBand(1).GetRasterColorTable() is not None:
            _gray_cache[key] = gdal.Translate("", ds, options=gdal.TranslateOptions(
                format="VRT", rgbExpand="rgb"))
        else:
            _gray_cache[key] = ds
    return _gray_cache[key]


def gray(ds, x, y, w, h, bw=None, bh=None):
    g = _gray_ds(ds)
    kw = dict(buf_xsize=bw, buf_ysize=bh) if bw else {}
    n = min(3, g.RasterCount)
    acc = None
    for i in range(1, n + 1):
        a = g.GetRasterBand(i).ReadAsArray(x, y, w, h, **kw).astype(np.float32)
        acc = a if acc is None else acc + a
    return acc / n


def ncc_match(img, tpl):
    """Peak of normalized cross-correlation of tpl over img. -> (x, y, score)"""
    th, tw = tpl.shape
    ih, iw = img.shape
    if th > ih or tw > iw:
        return None
    t0 = tpl - tpl.mean()
    tnorm = np.sqrt((t0 * t0).sum())
    if tnorm < 1e-6:
        return None
    H, W = ih + th - 1, iw + tw - 1
    F = np.fft.rfft2(img, s=(H, W))
    Tc = np.fft.rfft2(t0[::-1, ::-1], s=(H, W))
    corr = np.fft.irfft2(F * Tc, s=(H, W))[th - 1:ih, tw - 1:iw]
    ii = np.cumsum(np.cumsum(np.pad(img, ((1, 0), (1, 0))), 0), 1)
    ii2 = np.cumsum(np.cumsum(np.pad(img * img, ((1, 0), (1, 0))), 0), 1)
    n = th * tw
    S = ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw]
    S2 = ii2[th:, tw:] - ii2[:-th, tw:] - ii2[th:, :-tw] + ii2[:-th, :-tw]
    # Variance floor: a near-flat region (blank margin) must not explode the
    # quotient - require the window std to be >= 20% of the template's.
    var = np.maximum(S2 - S * S / n, 0.04 * tnorm * tnorm)
    ncc = corr / (np.sqrt(var) * tnorm)
    y, x = np.unravel_index(np.argmax(ncc), ncc.shape)
    return int(x), int(y), float(ncc[y, x])


def lcc_params(srs):
    if srs.GetAttrValue("PROJECTION") not in ("Lambert_Conformal_Conic_2SP", "Lambert Conformal Conic (2SP)"):
        return None
    if abs(srs.GetProjParm(osr.SRS_PP_FALSE_EASTING, 0.0)) > 1e-6:
        return None
    if abs(srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING, 0.0)) > 1e-6:
        return None
    return (srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1),
            srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2),
            srs.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN),
            srs.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN))


# sibling index (largest georeferenced tif per location, 05-14 preferred)
siblings = {}
for cycle in ("05-14-2026", "07-09-2026"):
    base = RAW / cycle
    if not base.exists():
        continue
    for d in sorted(base.glob(f"aeronav.faa.gov_visual_{cycle}_sectional-files_*")):
        if not d.is_dir():
            continue
        key = s.normalize_name(d.name.split("sectional-files_", 1)[1].replace("_", " "))
        if key in siblings:
            continue
        best, area = None, -1
        for t in d.glob("*.tif"):
            try:
                ds = gdal.Open(str(t))
            except Exception:
                continue
            if ds.GetGeoTransform(can_return_null=True) is None:
                continue
            if ds.RasterXSize * ds.RasterYSize > area:
                best, area = t, ds.RasterXSize * ds.RasterYSize
            ds = None
        if best:
            siblings[key] = best
print(f"siblings: {len(siblings)}")

sib_cache = {}


def solve_pair(tgt_path, sib_path):
    tgt_ds = gdal.Open(str(tgt_path))
    sib_ds = gdal.Open(str(sib_path))
    gt = sib_ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=sib_ds.GetProjection())
    params = lcc_params(srs)
    if params is None or abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
        return {"status": "sibling-not-plain-lcc"}
    sw, sh = sib_ds.RasterXSize, sib_ds.RasterYSize
    tw, th = tgt_ds.RasterXSize, tgt_ds.RasterYSize
    tdec = gray(tgt_ds, 0, 0, tw, th, tw // DEC, th // DEC)

    key = str(sib_path)
    if key not in sib_cache:
        sib_cache[key] = gray(sib_ds, 0, 0, sw, sh, sw // DEC, sh // DEC)
    sdec = sib_cache[key]

    # Window grid spans only the plausible overlap: a sibling can be much
    # taller/wider than the target (Hawaiian strip print vs composite tif) -
    # windows outside the target's reach can never match.
    ov_w8 = min(sdec.shape[1], tdec.shape[1] + 100)
    ov_h8 = min(sdec.shape[0], tdec.shape[0] + 100)
    matches = []
    for fy in (0.2, 0.5, 0.8):
        for fx in (0.2, 0.5, 0.8):
            twin = 192  # 1/8-scale template = 1536 full-res px
            sx8 = int(min(max(fx * ov_w8 - twin / 2, 0), sdec.shape[1] - twin))
            sy8 = int(min(max(fy * ov_h8 - twin / 2, 0), sdec.shape[0] - twin))
            m = ncc_match(tdec, sdec[sy8:sy8 + twin, sx8:sx8 + twin])
            if m is None or m[2] < 0.2:
                continue
            # full-res refine
            seed_x, seed_y = m[0] * DEC, m[1] * DEC
            fsx, fsy = sx8 * DEC, sy8 * DEC
            tw_full = 1024
            pad = 160
            cx = fsx + (twin * DEC - tw_full) // 2
            cy = fsy + (twin * DEC - tw_full) // 2
            rx = seed_x + (twin * DEC - tw_full) // 2 - pad
            ry = seed_y + (twin * DEC - tw_full) // 2 - pad
            rx = max(0, min(rx, tw - tw_full - 2 * pad))
            ry = max(0, min(ry, th - tw_full - 2 * pad))
            if cx < 0 or cy < 0 or cx + tw_full > sw or cy + tw_full > sh:
                continue
            tpl = gray(sib_ds, cx, cy, tw_full, tw_full)
            reg = gray(tgt_ds, rx, ry, tw_full + 2 * pad, tw_full + 2 * pad)
            r = ncc_match(reg, tpl)
            if r is None or r[2] < 0.25:
                continue
            matches.append(((cx + tw_full / 2, cy + tw_full / 2),
                            (rx + r[0] + tw_full / 2, ry + r[1] + tw_full / 2), r[2]))

    if len(matches) < 4:
        return {"status": f"too-few-matches ({len(matches)})"}

    # RANSAC similarity fit t = s*c + T
    pts_s = np.array([m[0] for m in matches])
    pts_t = np.array([m[1] for m in matches])
    best_fit = None
    n = len(matches)
    for i in range(n):
        for j in range(i + 1, n):
            d_s = np.linalg.norm(pts_s[j] - pts_s[i])
            d_t = np.linalg.norm(pts_t[j] - pts_t[i])
            if d_s < 500:
                continue
            sc = d_t / d_s
            if not 0.97 <= sc <= 1.03:
                continue
            T = pts_t[i] - sc * pts_s[i]
            resid = np.linalg.norm(pts_t - (sc * pts_s + T), axis=1)
            inliers = resid < 3.0
            if best_fit is None or inliers.sum() > best_fit[0]:
                best_fit = (int(inliers.sum()), sc, T, inliers)
    if best_fit is None or best_fit[0] < 4:
        return {"status": f"ransac-failed (matches={n}, best={best_fit[0] if best_fit else 0})",
                "matches": n}
    n_in, sc, T, inliers = best_fit
    # least-squares polish on inliers (solve s and T jointly, isotropic s)
    A_s = pts_s[inliers]
    A_t = pts_t[inliers]
    cs = A_s.mean(axis=0)
    ct = A_t.mean(axis=0)
    sc = float((((A_t - ct) * (A_s - cs)).sum()) / (((A_s - cs) ** 2).sum()))
    T = ct - sc * cs
    resid = np.linalg.norm(A_t - (sc * A_s + T), axis=1)

    # target corner px -> sibling px -> geo -> NAD83
    proj4 = dole_v2.LCC_TEMPLATE.format(lat1=params[0], lat2=params[1], lat0=params[2], lon0=params[3])
    lcc = osr.SpatialReference(); lcc.SetFromUserInput(proj4)
    nad83 = osr.SpatialReference(); nad83.ImportFromEPSG(4269)
    for sr in (lcc, nad83):
        sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    inv = osr.CoordinateTransformation(lcc, nad83)
    gcps = []
    for px, py in [(0, 0), (tw, 0), (tw, th), (0, th)]:
        sx_f = (px - T[0]) / sc
        sy_f = (py - T[1]) / sc
        mx = gt[0] + sx_f * gt[1]
        my = gt[3] + sy_f * gt[5]
        lon, lat, _ = inv.TransformPoint(mx, my)
        gcps.append({"px": px, "py": py, "lat": round(lat, 7), "lon": round(lon, 7)})
    return {
        "status": "ok", "sibling": str(sib_path), "scale": round(sc, 6),
        "T": [round(float(T[0]), 1), round(float(T[1]), 1)],
        "inliers": n_in, "windows": n, "resid_max": round(float(resid.max()), 2),
        "dims": [tw, th], "lcc": params, "gcps": gcps,
        "lcc_srs_proj4": proj4,
    }


def validate_cutline(entry, norm_loc, tgt_dims):
    shp = s.shapefile_index.get(norm_loc)
    if not shp or entry["status"] != "ok":
        return
    lcc = osr.SpatialReference(); lcc.SetFromUserInput(entry["lcc_srs_proj4"])
    ssrs = osr.SpatialReference()
    p4 = s.get_shapefile_srs(shp)
    ssrs.SetFromUserInput(p4 if p4 else "EPSG:4269")
    for sr in (lcc, ssrs):
        sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(ssrs, lcc)
    shp_ds = ogr.Open(str(shp))
    ext = shp_ds.GetLayer(0).GetExtent()
    # target px from lcc meters via corner gcp fit: rebuild affine from gcp1/gcp3
    g1, g3 = entry["gcps"][0], entry["gcps"][2]
    # affine: meters = m0 + px*mx ; solve from two corners
    to_lcc = osr.CoordinateTransformation(_nad83(), lcc)
    m1 = to_lcc.TransformPoint(g1["lon"], g1["lat"])[:2]
    m3 = to_lcc.TransformPoint(g3["lon"], g3["lat"])[:2]
    W, H = tgt_dims
    mx = (m3[0] - m1[0]) / W
    my = (m3[1] - m1[1]) / H
    pxs, pys = [], []
    for cx in (ext[0], ext[1]):
        for cy in (ext[2], ext[3]):
            X, Y = tr.TransformPoint(cx, cy)[:2]
            pxs.append((X - m1[0]) / mx)
            pys.append((Y - m1[1]) / my)
    inside = (min(pxs) > -0.1 * W and max(pxs) < 1.1 * W and min(pys) > -0.1 * H and max(pys) < 1.1 * H)
    entry["cutline_check"] = {"inside": bool(inside),
                              "px_range": [round(min(pxs)), round(max(pxs))],
                              "py_range": [round(min(pys)), round(max(pys))]}
    if not inside:
        entry["status"] = "cutline-outside-raster"


def _nad83():
    n = osr.SpatialReference()
    n.ImportFromEPSG(4269)
    n.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return n


rows = dole_v2.load_rows(CSV)
results = {}
for row in rows:
    fn = row["filename"]
    is_nara = fn in NARA_TARGETS
    if str(row.get("date")) not in TARGET_DATES and not is_nara:
        continue
    if dole_v2.row_gcp_pixels(row):
        continue
    norm_loc = s.normalize_name(row.get("location", ""))
    entry = {"location": row.get("location"), "date": row.get("date")}
    if not is_nara and row.get("date") == "2026-03-19" and norm_loc in zip_covered_norm:
        continue
    local = s.files_by_name.get(fn)
    if local is None or not local.exists():
        entry["status"] = "no-local-file"
        results[fn] = entry
        continue
    try:
        tds = gdal.Open(str(local))
        has_geo = tds.GetGeoTransform(can_return_null=True) is not None or tds.GetGCPCount() > 0
        dims = (tds.RasterXSize, tds.RasterYSize)
        tds = None
    except Exception as e:
        entry["status"] = f"unreadable: {str(e)[:60]}"
        results[fn] = entry
        continue
    if has_geo and not is_nara:
        continue
    sib = siblings.get(norm_loc)
    if sib is None:
        entry["status"] = "no-sibling"
        results[fn] = entry
        continue
    try:
        r = solve_pair(local, sib)
    except Exception as e:
        r = {"status": f"error: {str(e)[:80]}"}
    entry.update(r)
    if entry["status"] == "ok":
        validate_cutline(entry, norm_loc, dims)
    results[fn] = entry
    tag = f"s={entry.get('scale')} in={entry.get('inliers')}/{entry.get('windows')} rmax={entry.get('resid_max')}"
    print(f"{entry['status']:<24} {row['location']:<26} {tag if entry['status'] == 'ok' else ''} {fn[:55]}", flush=True)

with open(SCRATCH / "georef_transfer2.json", "w") as f:
    json.dump(results, f, indent=1)
counts = {}
for e in results.values():
    k = e["status"].split()[0].split("(")[0]
    counts[k] = counts.get(k, 0) + 1
print("SUMMARY:", json.dumps(counts))
