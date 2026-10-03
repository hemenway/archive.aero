#!/usr/bin/env python3
"""Alignment check: PMTiles tiles vs a GDAL resample of the mosaic they came from.

For each probe point, stitch a 5x5 block of z12 tiles out of the .pmtiles and
warp the same Web-Mercator window straight out of the mosaic GeoTIFF with GDAL
(which honours non-square pixels), then phase-correlate the two. A correct
conversion lands within ~0.3 px on both axes; the pre-aa74aa0 geotiff2pmtiles
put these eras 12-17 px north at z12.

  verify_align.py MOSAIC.tif ARCHIVE.pmtiles [--zoom 12] [--tol 1.0] [--auto-probes]
Exit status 0 when every probe that found chart content is within --tol px.
--auto-probes adds five probes inside the mosaic's own footprint (centre and
the four quarter points), so single-sheet and regional eras far from the
twelve city probes still get checked (archive-slicer, 2026-10-02).
"""
import argparse, math, subprocess, sys, tempfile, os
import numpy as np
from osgeo import gdal
from PIL import Image
gdal.UseExceptions()

ORIG = 20037508.342789244
PROBES = [("Albuquerque", 35.04, -106.61), ("Atlanta", 33.64, -84.43), ("Chicago", 41.98, -87.90),
          ("Seattle", 47.45, -122.31), ("Dallas", 32.90, -97.04), ("Miami", 25.79, -80.29),
          ("New York", 40.64, -73.78), ("Los Angeles", 33.94, -118.41), ("Anchorage", 61.17, -149.99),
          ("Honolulu", 21.32, -157.92), ("Minneapolis", 44.88, -93.22), ("Salt Lake", 40.79, -111.98)]


def tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def pm_tile(archive, z, x, y):
    out = subprocess.run(["pmtiles", "tile", archive, str(z), str(x), str(y)], capture_output=True)
    if out.returncode != 0 or not out.stdout or out.stdout.startswith(b"Tile not found"):
        return None
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        f.write(out.stdout)
    try:
        return Image.open(f.name).convert("RGBA")
    finally:
        os.unlink(f.name)


def gray_rgba(a):
    """(4,H,W) or (H,W,4) float -> luminance with alpha-0 pixels set to 0, plus mask."""
    if a.shape[0] == 4:
        a = np.moveaxis(a, 0, -1)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    m = a[..., 3] > 0
    return np.where(m, lum, 0.0), m


def phase(a, b):
    a = a - a.mean(); b = b - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    R = np.fft.fft2(a * w) * np.conj(np.fft.fft2(b * w)); R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    i = np.unravel_index(np.argmax(r), r.shape)
    sy, sx = [v if v <= s // 2 else v - s for v, s in zip(i, r.shape)]
    def sub(c, m, p):
        d = m - 2 * c + p
        return (m - p) / (2 * d) if d else 0.0
    H, W = r.shape
    sx += sub(r[i], r[i[0], (i[1] - 1) % W], r[i[0], (i[1] + 1) % W])
    sy += sub(r[i], r[(i[0] - 1) % H, i[1]], r[(i[0] + 1) % H, i[1]])
    return sx, sy, float(r[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mosaic"); ap.add_argument("archive")
    ap.add_argument("--zoom", type=int, default=12); ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--auto-probes", action="store_true")
    a = ap.parse_args()
    z, r = a.zoom, a.radius
    src = gdal.Open(a.mosaic)
    gt = src.GetGeoTransform()
    print(f"mosaic {os.path.basename(a.mosaic)}: {src.RasterXSize}x{src.RasterYSize} "
          f"xres={gt[1]!r} yres={-gt[5]!r} (y/x-1 = {(-gt[5] / gt[1] - 1):+.3e})")
    probes = list(PROBES)
    if a.auto_probes:
        def lonlat(x, y):
            return math.degrees(math.atan(math.sinh(y / ORIG * math.pi))), x / ORIG * 180
        x0, y0 = gt[0], gt[3]
        x1, y1 = x0 + gt[1] * src.RasterXSize, y0 + gt[5] * src.RasterYSize
        for label, fx, fy in (("centre", .5, .5), ("NW quarter", .25, .25), ("NE quarter", .75, .25),
                              ("SW quarter", .25, .75), ("SE quarter", .75, .75)):
            probes.append((label, *lonlat(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)))
    worst, n = 0.0, 0
    for name, lat, lon in probes:
        fx, fy = tile_xy(lat, lon, z); cx, cy = int(fx), int(fy)
        ts = 2 * ORIG / 2 ** z
        bounds = (-ORIG + (cx - r) * ts, ORIG - (cy + r + 1) * ts, -ORIG + (cx + r + 1) * ts, ORIG - (cy - r) * ts)
        size = 256 * (2 * r + 1)
        truth = gdal.Warp("", src, format="MEM", outputBounds=bounds, width=size, height=size,
                          resampleAlg="bilinear").ReadAsArray().astype(np.float64)
        t, tm = gray_rgba(truth)
        im = Image.new("RGBA", (size, size))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                tile = pm_tile(a.archive, z, cx + dx, cy + dy)
                if tile is not None:
                    im.paste(tile, ((dx + r) * 256, (dy + r) * 256))
        e, em = gray_rgba(np.asarray(im, dtype=np.float64))
        cover = min(tm.mean(), em.mean())
        if cover < 0.25:
            print(f"  {name:12s} skipped (chart coverage {cover:.0%})")
            continue
        sx, sy, pk = phase(t, e)
        d = max(abs(sx), abs(sy)); worst = max(worst, d); n += 1
        flag = "ok " if d <= a.tol else "BAD"
        print(f"  {flag} {name:12s} dx={sx:+.2f} dy={sy:+.2f} px z{z} peak={pk:.3f}")
    print(f"  worst |shift| {worst:.2f} px over {n} probes (tol {a.tol})")
    sys.exit(0 if n and worst <= a.tol else 1)


if __name__ == "__main__":
    main()
