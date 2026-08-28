#!/usr/bin/env python3
"""Extract georeferenced GeoTIFFs from iFly EFB legacy chart bundles.

The old iFly 700/720 card format stores each chart as a trio:
    <name>.dat  concatenated JPEG tiles
    <name>.fil  plaintext index, CRLF lines of "L-CC-RR.JPG,offset,length"
                (L = pyramid level, CC = column, RR = row; level 1 is finest;
                256x256 tiles, edge tiles partial)
    <name>.gti  ASCII georef: an exact copy of the source FAA GeoTIFF's
                LCC/NAD83 parameters (angles in RADIANS), geotransform origin
                (EastInsertion/NorthInsertion = top-left pixel edge), meters
                per pixel, nominal image dims, edition and effective dates.

iFly resampled the source image (~0.9x) before tiling, so the stitched
level-1 mosaic is smaller than the .gti's nominal dims but covers the same
projected bounding box: the output geotransform keeps the origin and scales
meters/pixel by nominal/stitched.  Verified against the FAA original of
Atlanta 91 North (rawtiffs) 2026-08-19.

Usage:
    ifly_extract.py --card-dir /Volumes/Untitled/Sectionals \
                    --out /Volumes/projects/ifly_extract \
                    [--charts Albuquerque_North ...] [--list names.txt]

Outputs <out>/<Location>_<edition>/<ImageName> plus manifest.jsonl.
Never write these into rawtiffs: they are derived, not as-downloaded.

Companion: scripts/ifly_card_rescue.py ddrescue-images the card and recovers
DELETED chart bundles from unallocated exFAT space; point --card-dir at its
recover output to extract those too.

Also handles the NEWER iFly bundle format (vfrsec/vfrtac/vfrwac era):
    <name>.dat   concatenated JPEG tiles (same idea as legacy)
    <name>.fil2  binary index: u32 tile_count, then 8-byte records
                 {u16 seq, u16 level, u32 offset} where seq = col*256 + row
                 within that pyramid level's grid (level 1 = full res,
                 256 px tiles); a final {0xFFFF, 0xFFFF, dat_size} sentinel
                 closes the last tile. Reverse-engineered 2026-08-19,
                 verified against the FAA original of Atlanta SEC 104.
    <name>.gtj   ASCII header: name, height, width, scale, then the
                 geographic bbox (maxlat, minlon, minlat, maxlon, center
                 lat/lon), effective + expiry dates, edition, pyramid scales.
                 The imagery is on an ellipsoidal Mercator (EPSG:3395) grid
                 (2026-08-24 height-test finding — NOT spherical 3857); the
                 bbox corners, Mercator-projected, define the georef.
Format is auto-detected per chart: .gti/.fil = legacy LCC, .gtj/.fil2 = new.
"""

import argparse
import glob
import io
import json
import math
import os
import re
import struct
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from osgeo import gdal, osr

gdal.UseExceptions()

TILE = 256


def parse_gti(path):
    g = {}
    with open(path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                g.setdefault(k, v)  # keep first (Corner1/2 dupes don't matter)
    name = g["ImageName"].strip()
    m = re.match(r"(.+?)\s+(\d+)(?:\s+(North|South|East|West))?\.tif$", name, re.I)
    if not m:
        raise ValueError(f"unparseable ImageName {name!r} in {path}")
    return {
        "image_name": name,
        "location": m.group(1),
        "edition": m.group(2),
        "half": m.group(3) or "",
        "width": int(g["ImageWidth"]),
        "height": int(g["ImageHeight"]),
        "mppx": float(g["MetersPerPixelX"]),
        "mppy": float(g["MetersPerPixelY"]),
        "east": float(g["EastInsertion"]),
        "north": float(g["NorthInsertion"]),
        "lon0": math.degrees(float(g["CentralMeridian"])),
        "lat0": math.degrees(float(g["OriginLat"])),
        "sp1": math.degrees(float(g["StandardParallel1"])),
        "sp2": math.degrees(float(g["StandardParallel2"])),
        "start_date": g.get("StartDate", "").strip(),
        "expire_date": g.get("ExpireDate", "").strip(),
    }


def parse_fil(path):
    levels = defaultdict(dict)
    with open(path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            name, off, ln = line.split(",")
            m = re.match(r"(\d+)-(\d+)-(\d+)\.JPG$", name, re.I)
            if not m:
                raise ValueError(f"unparseable tile name {name!r} in {path}")
            lvl, col, row = int(m.group(1)), int(m.group(2)), int(m.group(3))
            levels[lvl][(col, row)] = (int(off), int(ln))
    return levels


def stitch_finest(dat_path, levels):
    """Stitch the finest pyramid level (largest stitched width) to RGB array."""
    with open(dat_path, "rb") as dat:
        def tile_img(entry):
            off, ln = entry
            dat.seek(off)
            return Image.open(io.BytesIO(dat.read(ln)))

        # Finest level = largest stitched width, probing each level's edge tile.
        best = None
        for lvl, tiles in levels.items():
            ncol = max(c for c, _ in tiles) + 1
            nrow = max(r for _, r in tiles) + 1
            edge = tiles.get((ncol - 1, 0))
            w = (ncol - 1) * TILE + (tile_img(edge).size[0] if edge else TILE)
            if best is None or w > best[1]:
                best = (lvl, w, ncol, nrow)
        lvl, _, ncol, nrow = best
        tiles = levels[lvl]

        # Actual dims from the far edge tiles.
        last_col = tiles.get((ncol - 1, 0))
        last_row = tiles.get((0, nrow - 1))
        width = (ncol - 1) * TILE + (tile_img(last_col).size[0] if last_col else TILE)
        height = (nrow - 1) * TILE + (tile_img(last_row).size[1] if last_row else TILE)

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        missing = []
        for row in range(nrow):
            for col in range(ncol):
                entry = tiles.get((col, row))
                if entry is None:
                    missing.append((col, row))
                    continue
                canvas.paste(tile_img(entry).convert("RGB"), (col * TILE, row * TILE))
    return lvl, canvas, missing


def write_geotiff(out_path, canvas, gti, level, missing, card_dir):
    w_st, h_st = canvas.size
    sx = gti["width"] / w_st
    sy = gti["height"] / h_st
    gt = (gti["east"], gti["mppx"] * sx, 0.0, gti["north"], 0.0, gti["mppy"] * sy)

    srs = osr.SpatialReference()
    srs.ImportFromProj4(
        f"+proj=lcc +lat_1={gti['sp1']:.13f} +lat_2={gti['sp2']:.13f} "
        f"+lat_0={gti['lat0']:.13f} +lon_0={gti['lon0']:.13f} "
        f"+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
    )

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        out_path, w_st, h_st, 3, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadata({
        "SOURCE": f"iFly EFB card {card_dir}",
        "METHOD": "scripts/ifly_extract.py: stitched level-%d JPEG tiles; "
                  "georef from .gti (copy of FAA GeoTIFF), m/px scaled by "
                  "nominal/stitched dims (iFly ~0.9x resample)" % level,
        "NOMINAL_DIMS": f"{gti['width']}x{gti['height']}",
        "EDITION": gti["edition"],
        "EFFECTIVE": gti["start_date"],
        "EXPIRES": gti["expire_date"],
        "MISSING_TILES": str(len(missing)),
    })
    arr = np.asarray(canvas)
    for b in range(3):
        ds.GetRasterBand(b + 1).WriteArray(arr[:, :, b])
    ds.FlushCache()
    ds = None
    return gt


def parse_gtj(path):
    parts = [p.strip() for p in open(path, encoding="ascii", errors="replace")
             .read().replace("\r", "\n").split("\n") if p.strip()]
    name = parts[0]
    height, width = int(parts[1]), int(parts[2])
    maxlat, minlon, minlat, maxlon = (float(parts[i]) for i in range(4, 8))
    dates = [p for p in parts if re.match(r"\d{1,2}/\d{1,2}/\d{4}$", p)]
    ed = ""
    if len(dates) == 2:
        di = parts.index(dates[1])
        if di + 1 < len(parts) and re.match(r"^\d{1,4}$", parts[di + 1]):
            ed = parts[di + 1]
    m = re.match(r"(?:Sec_|Tac_|Wac_|ENR_)?(?:AK)?(.+)", name)
    return {
        "image_name": f"{name}.tif", "location": m.group(1).replace("_", " "),
        "edition": ed or "0", "half": "",
        "width": width, "height": height,
        "maxlat": maxlat, "minlon": minlon, "minlat": minlat, "maxlon": maxlon,
        "start_date": dates[0] if dates else "", "expire_date": dates[1] if len(dates) > 1 else "",
    }


def parse_fil2(path, dat_size):
    raw = open(path, "rb").read()
    (count,) = struct.unpack_from("<I", raw, 0)
    recs = [struct.unpack_from("<HHI", raw, 4 + 8 * i) for i in range(count)]
    levels = defaultdict(dict)
    for i, (seq, level, off) in enumerate(recs):
        if level == 0xFFFF:  # end sentinel
            continue
        end = recs[i + 1][2] if i + 1 < count else dat_size
        levels[level][(seq >> 8, seq & 0xFF)] = (off, end - off)
    return levels


def fil2_offsets_virtual(path, dat_size):
    """2014-15 era .fil2 (e.g. the 2015 SD card's sectionals): same record
    layout, but offsets index a pre-compression layout ~2.5x the .dat and the
    end sentinel disagrees with the file size. The .dat is the same tiles as
    back-to-back JPEGs in record order (257 px, 1 px overlap), so record order
    alone recovers the grid. Detect by the sentinel mismatch."""
    raw = open(path, "rb").read()
    (count,) = struct.unpack_from("<I", raw, 0)
    sentinel = struct.unpack_from("<HHI", raw, 4 + 8 * (count - 1))
    return sentinel[1] == 0xFFFF and sentinel[2] != dat_size


def parse_fil2_sequential(fil2_path, dat_path):
    """Map i-th record to i-th sequential JPEG in the .dat (virtual-offset
    era). Returns {level: {(col, row): jpeg_bytes}}; raises if the JPEG count
    does not equal the record count exactly."""
    raw = open(fil2_path, "rb").read()
    (count,) = struct.unpack_from("<I", raw, 0)
    recs = [struct.unpack_from("<HHI", raw, 4 + 8 * i) for i in range(count)]
    real = [r for r in recs if r[1] != 0xFFFF]

    dat = open(dat_path, "rb").read()
    blobs, pos = [], 0
    while pos + 3 <= len(dat) and dat[pos:pos + 3] == b"\xff\xd8\xff":
        p = pos + 2
        while p < len(dat) - 1:
            if dat[p] != 0xFF:
                break
            m = dat[p + 1]
            if m == 0xD9:
                p += 2
                break
            elif 0xD0 <= m <= 0xD8 or m == 0x01:
                p += 2
            elif m == 0xDA:
                q = p + 2 + struct.unpack_from(">H", dat, p + 2)[0]
                while q < len(dat) - 1:
                    if dat[q] == 0xFF and dat[q + 1] != 0x00 \
                            and not (0xD0 <= dat[q + 1] <= 0xD7):
                        break
                    q += 1
                p = q
            else:
                p += 2 + struct.unpack_from(">H", dat, p + 2)[0]
        blobs.append(dat[pos:p])
        pos = p
    if len(blobs) != len(real) or pos != len(dat):
        raise ValueError(
            f"sequential-JPEG walk mismatch: {len(blobs)} JPEGs vs "
            f"{len(real)} records, {len(dat) - pos} trailing bytes")
    levels = defaultdict(dict)
    for (seq, level, _), blob in zip(real, blobs):
        levels[level][(seq >> 8, seq & 0xFF)] = blob
    return levels


def write_geotiff_mercator(out_path, canvas, gtj, missing, card_dir):
    """New-format imagery sits on an ELLIPSOIDAL Mercator (EPSG:3395) grid,
    not spherical 3857 — corrected 2026-08-24: a height test on 14 .gtj files
    of both generations fits ellipsoidal within ±0.6 px, vs +7..+39 px for
    spherical (the earlier "sub-pixel 3857" check was a whole-chart shift
    test, blind to the corner-pinned mid-chart bow). Linear-latitude is far
    worse still (~2.9 km bow)."""
    w, h = canvas.size
    a = 6378137.0
    e = 0.0818191908426215  # WGS84 first eccentricity
    minlon, maxlon = gtj["minlon"], gtj["maxlon"]
    if maxlon < minlon:  # antimeridian crossing: run past 180
        maxlon += 360.0
    def mx(lon): return a * math.radians(lon)
    def my(lat):
        phi = math.radians(lat)
        es = e * math.sin(phi)
        return a * math.log(math.tan(math.pi / 4 + phi / 2)
                            * ((1 - es) / (1 + es)) ** (e / 2))
    x0, x1 = mx(minlon), mx(maxlon)
    y0, y1 = my(gtj["maxlat"]), my(gtj["minlat"])
    gt = (x0, (x1 - x0) / w, 0.0, y0, 0.0, -(y0 - y1) / h)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3395)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out_path, w, h, 3, gdal.GDT_Byte,
                    options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                             "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadata({
        "SOURCE": f"iFly EFB card {card_dir}",
        "METHOD": "scripts/ifly_extract.py: stitched level-1 JPEG tiles from "
                  ".fil2 index; EPSG:3395 ellipsoidal-Mercator georef from "
                  ".gtj bbox (2026-08-24 height-test finding, ±0.6 px); "
                  "iFly-resampled imagery",
        "EDITION": gtj["edition"], "EFFECTIVE": gtj["start_date"],
        "EXPIRES": gtj["expire_date"], "MISSING_TILES": str(len(missing)),
    })
    arr = np.asarray(canvas)
    for b in range(3):
        ds.GetRasterBand(b + 1).WriteArray(arr[:, :, b])
    ds.FlushCache()
    return gt


def stitch_fil2_level1(dat_path, levels, width, height):
    tiles = levels[min(levels)]
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    missing = []
    # grid shape from the nominal dims: blank (ocean/margin) tiles are simply
    # omitted from the index, so max-present coordinates can undercount
    ncol = -(-width // TILE)
    nrow = -(-height // TILE)
    with open(dat_path, "rb") as dat:
        for row in range(nrow):
            for col in range(ncol):
                ent = tiles.get((col, row))
                if ent is None:
                    missing.append((col, row))
                    continue
                if isinstance(ent, bytes):  # sequential (virtual-offset) era
                    im = Image.open(io.BytesIO(ent))
                else:
                    dat.seek(ent[0])
                    im = Image.open(io.BytesIO(dat.read(ent[1])))
                # 2014-15 tiles are 257 px (1 px overlap): 256 pitch + clip
                canvas.paste(im.convert("RGB"), (col * TILE, row * TILE))
    return canvas, missing, (ncol, nrow)


def extract_one_new(card_dir, base, out_root):
    dat_path = os.path.join(card_dir, base + ".dat")
    gtj = parse_gtj(os.path.join(card_dir, base + ".gtj"))
    fil2_path = os.path.join(card_dir, base + ".fil2")
    virtual = fil2_offsets_virtual(fil2_path, os.path.getsize(dat_path))
    levels = (parse_fil2_sequential(fil2_path, dat_path) if virtual
              else parse_fil2(fil2_path, os.path.getsize(dat_path)))
    canvas, missing, (ncol, nrow) = stitch_fil2_level1(
        dat_path, levels, gtj["width"], gtj["height"])
    present = levels[min(levels)]
    stray = [k for k in present if k[0] >= ncol or k[1] >= nrow]
    if stray:
        raise ValueError(f"{base}: {len(stray)} level-1 tiles outside the "
                         f"{ncol}x{nrow} grid for dims "
                         f"{gtj['width']}x{gtj['height']}, e.g. {stray[:4]}")
    out_dir = os.path.join(
        out_root, f"{gtj['location'].replace(' ', '_')}_{gtj['edition']}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, gtj["image_name"])
    gt = write_geotiff_mercator(out_path, canvas, gtj, missing, card_dir)
    rec = {"base": base, "format": "fil2-sequential" if virtual else "fil2",
           "image_name": gtj["image_name"],
           "location": gtj["location"], "edition": gtj["edition"], "half": "",
           "start_date": gtj["start_date"], "expire_date": gtj["expire_date"],
           "level": min(levels), "stitched": list(canvas.size),
           "nominal": [gtj["width"], gtj["height"]],
           "missing_tiles": len(missing), "geotransform": list(gt),
           "out": out_path}
    with open(os.path.join(out_root, "manifest.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def extract_one(card_dir, base, out_root):
    if not os.path.exists(os.path.join(card_dir, base + ".gti")) \
            and os.path.exists(os.path.join(card_dir, base + ".gtj")):
        return extract_one_new(card_dir, base, out_root)
    gti = parse_gti(os.path.join(card_dir, base + ".gti"))
    levels = parse_fil(os.path.join(card_dir, base + ".fil"))
    level, canvas, missing = stitch_finest(os.path.join(card_dir, base + ".dat"), levels)

    out_dir = os.path.join(
        out_root, f"{gti['location'].replace(' ', '_')}_{gti['edition']}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, gti["image_name"])
    gt = write_geotiff(out_path, canvas, gti, level, missing, card_dir)

    rec = {
        "base": base,
        "image_name": gti["image_name"],
        "location": gti["location"],
        "edition": gti["edition"],
        "half": gti["half"],
        "start_date": gti["start_date"],
        "expire_date": gti["expire_date"],
        "level": level,
        "stitched": list(canvas.size),
        "nominal": [gti["width"], gti["height"]],
        "missing_tiles": len(missing),
        "geotransform": list(gt),
        "out": out_path,
    }
    with open(os.path.join(out_root, "manifest.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--card-dir", default="/Volumes/Untitled/Sectionals")
    ap.add_argument("--out", required=True)
    ap.add_argument("--charts", nargs="*", default=[],
                    help="basenames (no extension), e.g. Albuquerque_North")
    ap.add_argument("--list", help="file with one basename per line")
    ap.add_argument("--all", action="store_true",
                    help="extract every chart bundle found in --card-dir")
    args = ap.parse_args()

    bases = list(args.charts)
    if args.list:
        with open(args.list) as fh:
            bases += [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    if args.all:
        bases += sorted(os.path.splitext(os.path.basename(p))[0]
                        for x in ("*.gti", "*.gtj")
                        for p in glob.glob(os.path.join(args.card_dir, x)))
    if not bases:
        ap.error("nothing to do: pass --charts, --list, or --all")

    os.makedirs(args.out, exist_ok=True)
    failures = []
    for i, base in enumerate(bases, 1):
        try:
            rec = extract_one(args.card_dir, base, args.out)
            tag = f" ({rec['missing_tiles']} missing tiles!)" if rec["missing_tiles"] else ""
            print(f"[{i}/{len(bases)}] {rec['image_name']}  "
                  f"{rec['stitched'][0]}x{rec['stitched'][1]} level {rec['level']}{tag}")
        except Exception as e:  # keep batch going, report at end
            failures.append((base, repr(e)))
            print(f"[{i}/{len(bases)}] {base}  FAILED: {e!r}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} failures:", file=sys.stderr)
        for base, err in failures:
            print(f"  {base}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
