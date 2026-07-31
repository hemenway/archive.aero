#!/usr/bin/env python3
"""
Sibling georef transfer for the ungeoreferenced catalogued .jpg scans
(dole_gap_2026-07/archive2004 halves, NARA rg-370, NOAA jpgs, ...).

Derived from georef_infer_from_sibling.py (the 2026-07-22 FAA-PDF run) with
three changes for this target set:

  1. Targets = every catalogued *.jpg row with no GCPs whose on-disk file
     carries no georeferencing (skips the AVSIM world-file jpgs on its own).
  2. Donor choice is per (location, half): prefer a georeferenced FAA tif of
     the SAME half (wayback old-layout "Charlotte 90 South.tif" style),
     lowest edition number first (closest era to 2004); fall back to the
     current-cycle full chart.
  3. Scale normalization: the 2004 scans are ~half the FAA tif resolution,
     and NCC is not scale-invariant. The donor is wrapped in a resampled VRT
     at the target's estimated pixel scale (sc_est = tgt_w / sib_w — both are
     full-chart-width images), so the existing ~1:1 pipeline applies; the
     VRT's geotransform stays valid for the corner-GCP math. The RANSAC scale
     band widens to 0.93-1.07 to absorb margin-crop differences.

Window grid spans the WHOLE donor (not the top-left overlap) so south halves
can match full-chart fallback donors. Emits georef_transfer_2004.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/ryanhemenway/archive.aero/scripts")
from osgeo import gdal, osr, ogr
gdal.UseExceptions()
import dole_v2
from slicer import ChartSlicer

RAW = Path("/Volumes/projects/rawtiffs")
SCRATCH = Path("/private/tmp/claude-501/-Users-ryanhemenway-archive-aero/f7ec1c59-1503-421f-87f9-bf71cfb5dec6/scratchpad")
CSV = Path("/Users/ryanhemenway/archive.aero/master_dole_v2.csv")

DEC = 8
HALF_TOKENS = ("north", "south", "east", "west")

s = ChartSlicer(RAW, SCRATCH / "out", CSV, Path("/Users/ryanhemenway/archive.aero/shapefiles"))
s._build_source_index()
s._build_shapefile_index()

_gray_cache = {}


def _gray_ds(ds):
    key = ds.GetDescription() or str(id(ds))
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


def stem_half_and_edition(stem_norm, loc_norm):
    """Half token + edition number from the stem AFTER removing the location
    match, so 'Key West'-style location words never read as a half."""
    rest = stem_norm.replace(loc_norm, " ", 1)
    toks = re.split(r"[^a-z0-9]+", rest)
    half = next((t for t in toks if t in HALF_TOKENS), None)
    ed = next((int(t) for t in toks if t.isdigit() and len(t) <= 3), None)
    return half, ed


# ---- donor index: georeferenced FAA tifs by (location, half) --------------
def build_donor_pool(loc_norms):
    """{loc_norm: [(priority_tuple, path, half)]} over aeronav.faa.gov tifs."""
    pool = {}
    for fname, path in s.files_by_name.items():
        if not fname.lower().endswith(".tif"):
            continue
        if "aeronav.faa.gov" not in str(path.parent.name):
            continue
        stem_norm = s.normalize_name(Path(fname).stem)
        for loc in loc_norms:
            if loc in stem_norm:
                half, ed = stem_half_and_edition(stem_norm, loc)
                pool.setdefault(loc, []).append((ed if ed is not None else 999, path, half))
                break
    return pool


def pick_donor(pool, loc_norm, tgt_half):
    """Ordered candidates: same-half lowest-edition first, then full-chart,
    then anything; first one that opens with a plain-LCC georef wins."""
    cands = sorted(pool.get(loc_norm, []), key=lambda c: (
        0 if (tgt_half and c[2] == tgt_half) else (1 if c[2] is None else 2),
        c[0]))
    for _ed, path, half in cands:
        try:
            ds = gdal.Open(str(path))
            gt = ds.GetGeoTransform(can_return_null=True)
            if gt is None or abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
                continue
            if lcc_params(osr.SpatialReference(wkt=ds.GetProjection())) is None:
                continue
            ds = None
            return path, half
        except Exception:
            continue
    return None, None


_scaled_cache = {}


def scaled_sibling(sib_path, sc_est):
    """Donor resampled to the target's pixel scale (bilinear VRT). Paletted
    donors are expanded to RGB FIRST — resampling raw palette indices (which
    forces nearest) leaves NCC nothing to correlate. The VRT keeps a
    consistent geotransform, so downstream geo math is unchanged."""
    key = (str(sib_path), round(sc_est, 4))
    if key not in _scaled_cache:
        ds = gdal.Open(str(sib_path))
        if ds.GetRasterBand(1).GetRasterColorTable() is not None:
            ds = gdal.Translate("", ds, options=gdal.TranslateOptions(
                format="VRT", rgbExpand="rgb"))
        w = max(1, int(round(ds.RasterXSize * sc_est)))
        h = max(1, int(round(ds.RasterYSize * sc_est)))
        _scaled_cache.clear()          # one donor at a time; keep memory flat
        _gray_cache.clear()
        _scaled_cache[key] = gdal.Translate("", ds, options=gdal.TranslateOptions(
            format="VRT", width=w, height=h, resampleAlg="bilinear"))
    return _scaled_cache[key]


def solve_pair(tgt_path, sib_ds):
    tgt_ds = gdal.Open(str(tgt_path))
    gt = sib_ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=sib_ds.GetProjection())
    params = lcc_params(srs)
    if params is None or abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
        return {"status": "sibling-not-plain-lcc"}
    sw, sh = sib_ds.RasterXSize, sib_ds.RasterYSize
    tw, th = tgt_ds.RasterXSize, tgt_ds.RasterYSize
    tdec = gray(tgt_ds, 0, 0, tw, th, tw // DEC, th // DEC)

    sdec = gray(sib_ds, 0, 0, sw, sh, sw // DEC, sh // DEC)

    # Grid over the WHOLE donor: with a full-chart fallback donor only some
    # rows overlap a half target; NCC score thresholds discard the rest.
    matches = []
    fys = (0.15, 0.38, 0.62, 0.85) if sh <= 1.5 * th else (0.12, 0.3, 0.5, 0.7, 0.88)
    for fy in fys:
        for fx in (0.12, 0.35, 0.6, 0.85):
            twin = 192
            sx8 = int(min(max(fx * sdec.shape[1] - twin / 2, 0), sdec.shape[1] - twin))
            sy8 = int(min(max(fy * sdec.shape[0] - twin / 2, 0), sdec.shape[0] - twin))
            if sx8 < 0 or sy8 < 0:
                continue
            m = ncc_match(tdec, sdec[sy8:sy8 + twin, sx8:sx8 + twin])
            if m is None or m[2] < 0.15:
                continue
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
            if r is None or r[2] < 0.2:
                continue
            matches.append(((cx + tw_full / 2, cy + tw_full / 2),
                            (rx + r[0] + tw_full / 2, ry + r[1] + tw_full / 2), r[2]))

    if len(matches) < 4:
        return {"status": f"too-few-matches ({len(matches)})"}

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
            # wider than the FAA-PDF run: sc_est absorbs resolution, this
            # band only has to absorb margin-crop error in the estimate
            if not 0.93 <= sc <= 1.07:
                continue
            T = pts_t[i] - sc * pts_s[i]
            resid = np.linalg.norm(pts_t - (sc * pts_s + T), axis=1)
            # 4px inlier band (vs 3 in the FAA-PDF run): cross-era scans +
            # 2x downsampling of the donor blur the correlation peaks
            inliers = resid < 4.0
            if best_fit is None or inliers.sum() > best_fit[0]:
                best_fit = (int(inliers.sum()), sc, T, inliers)
    if best_fit is None or best_fit[0] < 4:
        return {"status": f"ransac-failed (matches={n}, best={best_fit[0] if best_fit else 0})",
                "matches": n}
    n_in, sc, T, inliers = best_fit
    A_s = pts_s[inliers]
    A_t = pts_t[inliers]
    cs = A_s.mean(axis=0)
    ct = A_t.mean(axis=0)
    sc = float((((A_t - ct) * (A_s - cs)).sum()) / (((A_s - cs) ** 2).sum()))
    T = ct - sc * cs
    resid = np.linalg.norm(A_t - (sc * A_s + T), axis=1)

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
        "status": "ok", "scale": round(sc, 6),
        "T": [round(float(T[0]), 1), round(float(T[1]), 1)],
        "inliers": n_in, "windows": n, "resid_max": round(float(resid.max()), 2),
        "dims": [tw, th], "lcc": params, "gcps": gcps,
        "lcc_srs_proj4": proj4,
    }


def _nad83():
    n = osr.SpatialReference()
    n.ImportFromEPSG(4269)
    n.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return n


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
    g1, g3 = entry["gcps"][0], entry["gcps"][2]
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
    # A HALF target only covers ~half the sectional cutline: require overlap,
    # not containment — the cutline rectangle must INTERSECT the raster with
    # sane orientation, and the raster must sit inside the cutline's px box.
    inter_w = min(max(pxs), W) - max(min(pxs), 0)
    inter_h = min(max(pys), H) - max(min(pys), 0)
    overlaps = inter_w > 0.5 * W and inter_h > 0.35 * H
    entry["cutline_check"] = {"overlaps": bool(overlaps),
                             "px_range": [round(min(pxs)), round(max(pxs))],
                             "py_range": [round(min(pys)), round(max(pys))]}
    if not overlaps:
        entry["status"] = "cutline-mismatch"


# ---- main -----------------------------------------------------------------
def main():
    only = set(sys.argv[1:])   # optional filename filter for testing
    rows = dole_v2.load_rows(CSV)
    targets = []
    for row in rows:
        fn = row["filename"]
        if not fn.lower().endswith(".jpg"):
            continue
        if only and fn not in only:
            continue
        if dole_v2.row_gcp_pixels(row):
            continue
        local = s.files_by_name.get(fn)
        entry = {"location": row.get("location"), "date": str(row.get("date") or "")}
        if local is None or not local.exists():
            entry["status"] = "no-local-file"
            targets.append((fn, row, None, entry))
            continue
        try:
            tds = gdal.Open(str(local))
            has_geo = tds.GetGeoTransform(can_return_null=True) is not None or tds.GetGCPCount() > 0
            entry["dims"] = [tds.RasterXSize, tds.RasterYSize]
            tds = None
        except Exception as e:
            entry["status"] = f"unreadable: {str(e)[:60]}"
            targets.append((fn, row, None, entry))
            continue
        if has_geo:
            continue    # AVSIM world-file jpgs etc. — already georeferenced
        targets.append((fn, row, local, entry))

    loc_norms = {s.normalize_name(r.get("location", "")) for _fn, r, _l, _e in targets if r.get("location")}
    pool = build_donor_pool(loc_norms)
    print(f"targets: {len(targets)}  donor locations: {len(pool)}", flush=True)

    results = {}
    for fn, row, local, entry in sorted(targets, key=lambda t: t[0]):
        if entry.get("status"):
            results[fn] = entry
            print(f"{entry['status']:<28} {entry['location']:<26} {fn[:55]}", flush=True)
            continue
        norm_loc = s.normalize_name(row.get("location", ""))
        tgt_half, _ = stem_half_and_edition(s.normalize_name(Path(fn).stem), norm_loc)
        donor, donor_half = pick_donor(pool, norm_loc, tgt_half)
        if donor is None:
            entry["status"] = "no-sibling"
            results[fn] = entry
            print(f"{entry['status']:<28} {entry['location']:<26} {fn[:55]}", flush=True)
            continue
        entry["sibling"] = str(donor)
        entry["sibling_half"] = donor_half
        entry["target_half"] = tgt_half
        try:
            sds = gdal.Open(str(donor))
            sc_est = entry["dims"][0] / sds.RasterXSize
            sds = None
            entry["sc_est"] = round(sc_est, 4)
            sib = scaled_sibling(donor, sc_est)
            r = solve_pair(local, sib)
        except Exception as e:
            r = {"status": f"error: {str(e)[:80]}"}
        entry.update(r)
        if entry["status"] == "ok":
            validate_cutline(entry, norm_loc, entry["dims"])
        results[fn] = entry
        tag = f"s={entry.get('scale')} in={entry.get('inliers')}/{entry.get('windows')} rmax={entry.get('resid_max')} sib={Path(entry.get('sibling','')).stem[:28]}"
        print(f"{entry['status']:<28} {entry['location']:<26} {tag if entry['status'] == 'ok' else Path(entry.get('sibling','?')).stem[:30]} {fn[:48]}", flush=True)

    out = SCRATCH / "georef_transfer_2004.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    counts = {}
    for e in results.values():
        k = e["status"].split()[0].split("(")[0]
        counts[k] = counts.get(k, 0) + 1
    print("SUMMARY:", json.dumps(counts), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
