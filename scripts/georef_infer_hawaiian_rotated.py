#!/usr/bin/env python3
"""
Rotation-aware georef transfer for the Hawaiian Islands strip prints.

The FAA composite GeoTIFF draws the island chain geographically (diagonal,
north-up raster); the print/PDF product is an OBLIQUE strip - the chart is
rotated on the sheet. A similarity fit without rotation can never match, but
the slicer's order-1 GCP warp is a full affine, so 4 corner GCPs express the
rotation exactly.

Method: search the rotation angle by NCC of a central sibling window against
rotated copies of the target, then match windows on the best-angle rotation
and fit a full 6-parameter affine sib_px -> target_px.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/ryanhemenway/archive.aero/scripts")
from osgeo import gdal, osr
gdal.UseExceptions()
import dole_v2

RAW = Path("/Volumes/projects/rawtiffs")
SCRATCH = Path("/private/tmp/claude-501/-Users-ryanhemenway-archive-aero/979396e4-d26f-44e2-a132-01bc48223900/scratchpad")
SIB = RAW / "05-14-2026/aeronav.faa.gov_visual_05-14-2026_sectional-files_Hawaiian_Islands/Hawaiian Islands SEC.tif"
TARGETS = [
    "11-27-2025/web.archive.org_web_20251223180751_aeronav.faa.gov_visual_11-27-2025_PDFs_Hawaiian_Islands.tif",
    "01-22-2026/web.archive.org_web_20260109103044_aeronav.faa.gov_visual_01-22-2026_PDFs_Hawaiian_Islands.tif",
    "03-19-2026/web.archive.org_web_20260329031545_aeronav.faa.gov_visual_03-19-2026_PDFs_Hawaiian_Islands.tif",
]
DEC = 8


def gray_ds(ds):
    if ds.GetRasterBand(1).GetRasterColorTable() is not None:
        return gdal.Translate("", ds, options=gdal.TranslateOptions(format="VRT", rgbExpand="rgb"))
    return ds


def gray(ds, x, y, w, h, bw=None, bh=None):
    g = gray_ds(ds)
    kw = dict(buf_xsize=bw, buf_ysize=bh) if bw else {}
    n = min(3, g.RasterCount)
    return sum(g.GetRasterBand(i).ReadAsArray(x, y, w, h, **kw).astype(np.float32)
               for i in range(1, n + 1)) / n


def ncc_match(img, tpl):
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


def rotate_img(img, deg):
    """Rotate CCW-positive (math convention, y down so visually CW) about the
    center, nearest neighbor, output sized to hold the whole rotation."""
    th = np.deg2rad(deg)
    h, w = img.shape
    c, s = np.cos(th), np.sin(th)
    W = int(abs(w * c) + abs(h * s)) + 2
    H = int(abs(w * s) + abs(h * c)) + 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xc, yc = xx - W / 2, yy - H / 2
    # inverse map (rotate output coords by -deg back into source)
    xs = c * xc + s * yc + w / 2
    ys = -s * xc + c * yc + h / 2
    xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1)
    out = img[yi, xi]
    out[(xs < 0) | (xs > w - 1) | (ys < 0) | (ys > h - 1)] = img.mean()
    return out


def rot_fwd(pt, deg, w, h, W, H):
    """Map source px -> rotated-image px for rotate_img."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    x0, y0 = pt[0] - w / 2, pt[1] - h / 2
    return (c * x0 - s * y0 + W / 2, s * x0 + c * y0 + H / 2)


sib_ds = gdal.Open(str(SIB))
gt = sib_ds.GetGeoTransform()
srs = osr.SpatialReference(wkt=sib_ds.GetProjection())
lat1 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1)
lat2 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2)
lat0 = srs.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN)
lon0 = srs.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN)
sw, sh = sib_ds.RasterXSize, sib_ds.RasterYSize
sdec = gray(sib_ds, 0, 0, sw, sh, sw // DEC, sh // DEC)

results = {}
for rel in TARGETS:
    tgt_path = RAW / rel
    fn = tgt_path.name
    if not tgt_path.exists():
        results[fn] = {"status": "no-local-file"}
        continue
    tds = gdal.Open(str(tgt_path))
    tw_, th_ = tds.RasterXSize, tds.RasterYSize
    tdec = gray(tds, 0, 0, tw_, th_, tw_ // DEC, th_ // DEC)

    # angle search: rotate the TARGET so its strip matches the sibling's
    # diagonal chain. Central sibling window as probe.
    twin = 224
    sx8 = sdec.shape[1] // 2 - twin // 2
    sy8 = sdec.shape[0] // 3 - twin // 2  # chain sits in the upper part
    probe = sdec[sy8:sy8 + twin, sx8:sx8 + twin]
    best = None
    for deg in np.arange(-45, 46, 2.5):
        r = rotate_img(tdec, deg)
        m = ncc_match(r, probe)
        if m and (best is None or m[2] > best[2]):
            best = (deg, m, m[2])
    if best is None or best[2] < 0.2:
        results[fn] = {"status": f"angle-search-failed best={best[2] if best else None}"}
        print(fn, results[fn]["status"])
        continue
    deg0 = best[0]
    for deg in np.arange(deg0 - 2.0, deg0 + 2.01, 0.5):
        r = rotate_img(tdec, deg)
        m = ncc_match(r, probe)
        if m and m[2] > best[2]:
            best = (deg, m, m[2])
    deg = float(best[0])
    print(f"{fn[:60]} angle={deg} score={best[2]:.3f}")

    # match windows on the rotated decimated target, then express matches in
    # ORIGINAL target coords and fit a full affine sib -> target
    rdec = rotate_img(tdec, deg)
    RH, RW = rdec.shape
    pairs = []
    for fy in (0.15, 0.3, 0.45):
        for fx in (0.15, 0.35, 0.55, 0.75):
            w8 = 160
            ax = int(min(max(fx * sdec.shape[1] - w8 / 2, 0), sdec.shape[1] - w8))
            ay = int(min(max(fy * sdec.shape[0] - w8 / 2, 0), sdec.shape[0] - w8))
            m = ncc_match(rdec, sdec[ay:ay + w8, ax:ax + w8])
            if m is None or m[2] < 0.3:
                continue
            # rotated target px (decimated) -> original target px
            rx, ry = m[0] + w8 / 2, m[1] + w8 / 2
            th_r = np.deg2rad(deg)
            c, s = np.cos(th_r), np.sin(th_r)
            xc, yc = rx - RW / 2, ry - RH / 2
            ox = (c * xc + s * yc + tdec.shape[1] / 2) * DEC
            oy = (-s * xc + c * yc + tdec.shape[0] / 2) * DEC
            pairs.append(((ax + w8 / 2) * DEC, (ay + w8 / 2) * DEC, ox, oy, m[2]))
    if len(pairs) < 4:
        results[fn] = {"status": f"too-few-rot-matches ({len(pairs)})"}
        print(fn, results[fn]["status"])
        continue

    # least-squares affine with RANSAC (target = A @ sib + b)
    P = np.array(pairs)
    best_fit = None
    n = len(pairs)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len({i, j, k}) < 3:
                    continue
                src = np.array([[P[m_][0], P[m_][1], 1] for m_ in (i, j, k)])
                if abs(np.linalg.det(src)) < 1e5:
                    continue
                try:
                    coefx = np.linalg.solve(src, P[[i, j, k], 2])
                    coefy = np.linalg.solve(src, P[[i, j, k], 3])
                except np.linalg.LinAlgError:
                    continue
                pred_x = P[:, 0] * coefx[0] + P[:, 1] * coefx[1] + coefx[2]
                pred_y = P[:, 0] * coefy[0] + P[:, 1] * coefy[1] + coefy[2]
                resid = np.hypot(pred_x - P[:, 2], pred_y - P[:, 3])
                inl = resid < DEC * 1.5  # decimated matching -> ~1.5 dec px
                if best_fit is None or inl.sum() > best_fit[0]:
                    best_fit = (int(inl.sum()), inl)
    if best_fit is None or best_fit[0] < 4:
        results[fn] = {"status": f"rot-ransac-failed ({n} matches)"}
        print(fn, results[fn]["status"])
        continue
    inl = best_fit[1]
    A = np.column_stack([P[inl, 0], P[inl, 1], np.ones(inl.sum())])
    coefx, *_ = np.linalg.lstsq(A, P[inl, 2], rcond=None)
    coefy, *_ = np.linalg.lstsq(A, P[inl, 3], rcond=None)
    pred_x = P[inl, 0] * coefx[0] + P[inl, 1] * coefx[1] + coefx[2]
    pred_y = P[inl, 0] * coefy[0] + P[inl, 1] * coefy[1] + coefy[2]
    rmax = float(np.hypot(pred_x - P[inl, 2], pred_y - P[inl, 3]).max())

    # invert affine: sib_px = M^-1 (target_px - b)
    M = np.array([[coefx[0], coefx[1]], [coefy[0], coefy[1]]])
    b = np.array([coefx[2], coefy[2]])
    Minv = np.linalg.inv(M)
    proj4 = dole_v2.LCC_TEMPLATE.format(lat1=lat1, lat2=lat2, lat0=lat0, lon0=lon0)
    lcc = osr.SpatialReference(); lcc.SetFromUserInput(proj4)
    nad83 = osr.SpatialReference(); nad83.ImportFromEPSG(4269)
    for sr in (lcc, nad83):
        sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    inv = osr.CoordinateTransformation(lcc, nad83)
    gcps = []
    for px, py in [(0, 0), (tw_, 0), (tw_, th_), (0, th_)]:
        sp = Minv @ (np.array([px, py]) - b)
        mx = gt[0] + sp[0] * gt[1]
        my = gt[3] + sp[1] * gt[5]
        lon, lat, _ = inv.TransformPoint(mx, my)
        gcps.append({"px": px, "py": py, "lat": round(lat, 7), "lon": round(lon, 7)})
    results[fn] = {
        "status": "ok", "sibling": str(SIB), "scale": round(float(np.sqrt(abs(np.linalg.det(M)))), 6),
        "T": [round(float(b[0]), 1), round(float(b[1]), 1)],
        "inliers": int(inl.sum()), "windows": n, "resid_max": round(rmax, 2),
        "rotation_deg": deg, "dims": [tw_, th_],
        "lcc": (lat1, lat2, lat0, lon0), "gcps": gcps,
    }
    print(f"OK {fn[:60]} angle={deg} inl={inl.sum()}/{n} rmax={rmax:.1f}px scale={results[fn]['scale']}")

with open(SCRATCH / "georef_transfer_hi.json", "w") as f:
    json.dump(results, f, indent=1)
print("done")
