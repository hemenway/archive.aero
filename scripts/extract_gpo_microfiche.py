#!/usr/bin/env python3
"""Extract chart images from the HUNT26 GPO depository microfiche zips.

Each archive.org ``*_micro_jp2.zip`` in rawtiffs/hunt_2026-07/gpo_microfiche/
holds, per fiche frame, 16 jp2 camera tiles, a hugin .pto, and the finished
stitched frame ``NNNN.jp2`` (dimensions match the .pto output crop exactly),
so hugin never needs to run. The stitched frame is the WHOLE fiche sheet:
a GPO/READEX header bar, mostly blank fiche, and 1-2 small dark cells that
are the actual chart exposures (negatives). One chart half per cell — same
data model as the archive.org N/S half-sheet JPG pairs, one dole row each.

Stages (run in order):
  detect  — downsampled read of every frame, find cell bounding boxes with
            cv2 connected components; writes boxes.json + per-frame overview
            PNGs (cells outlined) for visual verification.
  crop    — full-res crop of each verified box straight from /vsizip/, with
            negative->positive inversion, to <zipstem>_f{frame}_c{cell}.tif.
            Compression stays LZW (slicer pipeline requirement: never ZSTD).
  apply   — rewrite the matching master_dole_v2.csv rows: the .zip filename
            becomes the first cell tif; extra cells get a sibling row
            inserted right after (same dates/edition/cutline); notes updated.

Usage:
    ~/venv/bin/python scripts/extract_gpo_microfiche.py detect [--only S] [--workdir D]
    ~/venv/bin/python scripts/extract_gpo_microfiche.py crop   [--only S] [--workdir D]
    ~/venv/bin/python scripts/extract_gpo_microfiche.py apply  [--workdir D] [--dry-run]
"""
import argparse
import json
import os
import re
import sys
import zipfile

import cv2
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dole_v2

gdal.UseExceptions()

GPO_DIR = "/Volumes/projects/rawtiffs/hunt_2026-07/gpo_microfiche"
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "master_dole_v2.csv")
FRAME_RE = re.compile(r"^[^/]+/(\d{4})\.jp2$")
DETECT_WIDTH = 1600          # downsample width for cell detection
DARK_THRESH = 128            # fiche background is light, cells/header dark
MARGIN = 40                  # full-res pixels of margin around each cell
CREATION_OPTS = [
    "COMPRESS=LZW",
    "PREDICTOR=2",
    "TILED=YES",
    "BIGTIFF=IF_SAFER",
    "NUM_THREADS=ALL_CPUS",
]
OLD_NOTE = ("zip of jp2 camera tiles + hugin .pto; must be stitched to a "
            "single tif before use")


def list_zips(only=None):
    return sorted(
        f for f in os.listdir(GPO_DIR)
        if f.endswith("_micro_jp2.zip") and (not only or only in f)
    )


def frames_in_zip(zip_path):
    """(member path, frame number) for each stitched frame, sorted."""
    with zipfile.ZipFile(zip_path) as zf:
        found = [(n, int(m.group(1))) for n in zf.namelist()
                 if (m := FRAME_RE.match(n))]
    return sorted(found, key=lambda t: t[1])


def vsi_path(zip_name, member):
    return f"/vsizip/{os.path.join(GPO_DIR, zip_name)}/{member}"


def split_wide(mask, box):
    """Fiche exposures are step-and-repeat, roughly square. A box much wider
    than tall is two touching cells: split it at internal light seams
    (column runs where the dark fraction dips)."""
    x, y, w, h = box
    if w / h <= 1.6:
        return [box]
    sub = mask[y:y + h, x:x + w]
    dark_frac = sub.mean(axis=0)
    interior = slice(int(0.15 * w), int(0.85 * w))
    valley_cols = [i for i in range(*interior.indices(w))
                   if dark_frac[i] < 0.35]
    if not valley_cols:
        return [box]
    # contiguous valley runs -> split at each run's center
    runs, start = [], valley_cols[0]
    for a, b in zip(valley_cols, valley_cols[1:] + [None]):
        if b is None or b != a + 1:
            runs.append((start + a) // 2)
            start = b
    out, x0 = [], x
    for cut in runs:
        out.append((x0, y, x + cut - x0, h))
        x0 = x + cut
    out.append((x0, y, x + w - x0, h))
    return [b for b in out if b[2] > 0.05 * mask.shape[1]]


def detect_cells(src):
    """Cell bounding boxes in full-res pixels, left-to-right, top-to-bottom."""
    ds = gdal.Open(src)
    fw, fh = ds.RasterXSize, ds.RasterYSize
    scale = fw / DETECT_WIDTH
    small = ds.GetRasterBand(1).ReadAsArray(
        buf_xsize=DETECT_WIDTH, buf_ysize=round(fh / scale))
    mask = (small < DARK_THRESH).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    sh, sw = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 0.10 * sw or h < 0.15 * sh:
            continue          # specks, edge shadows
        if w / h > 6:
            continue          # full-width READEX header bar
        if area < 0.5 * w * h:
            continue          # not a solid block
        boxes.extend(split_wide(mask, (x, y, w, h)))
    # reading order: row bands (by vertical overlap), then left-to-right
    boxes.sort(key=lambda b: b[1])
    rows_of = []
    for b in boxes:
        for band in rows_of:
            if b[1] < band[0][1] + band[0][3] * 0.5:
                band.append(b)
                break
        else:
            rows_of.append([b])
    boxes = [b for band in rows_of for b in sorted(band, key=lambda b: b[0])]
    full = []
    for x, y, w, h in boxes:
        x0 = max(0, int(x * scale) - MARGIN)
        y0 = max(0, int(y * scale) - MARGIN)
        x1 = min(fw, int((x + w) * scale) + MARGIN)
        y1 = min(fh, int((y + h) * scale) + MARGIN)
        full.append([x0, y0, x1 - x0, y1 - y0])
    return full, small, scale


def overview_png(small, scale, boxes, out_path):
    vis = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    for x0, y0, w, h in boxes:
        p0 = (int(x0 / scale), int(y0 / scale))
        p1 = (int((x0 + w) / scale), int((y0 + h) / scale))
        cv2.rectangle(vis, p0, p1, (0, 0, 255), 3)
    cv2.imwrite(out_path, vis)


def cell_tif_name(zip_name, frame_i, cell_i):
    stem = os.path.splitext(zip_name)[0]
    return f"{stem}_f{frame_i}_c{cell_i}.tif"


def stage_detect(args):
    os.makedirs(args.workdir, exist_ok=True)
    results = {}
    zips = list_zips(args.only)
    for n, z in enumerate(zips, 1):
        print(f"[{n}/{len(zips)}] {z}")
        frames = frames_in_zip(os.path.join(GPO_DIR, z))
        results[z] = []
        for frame_i, (member, _num) in enumerate(frames):
            boxes, small, scale = detect_cells(vsi_path(z, member))
            results[z].append({"member": member, "boxes": boxes})
            ov = os.path.join(args.workdir, f"{os.path.splitext(z)[0]}_f{frame_i}.png")
            overview_png(small, scale, boxes, ov)
            print(f"    frame {frame_i}: {len(boxes)} cell(s)")
    out = os.path.join(args.workdir, "boxes.json")
    # merge with any earlier partial run
    if os.path.exists(out) and args.only:
        with open(out) as f:
            merged = json.load(f)
        merged.update(results)
        results = merged
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}")


def load_decisions(workdir):
    """Optional curation file mapping zip -> frame index (str) -> list of
    per-cell roles ('chart' = crop + dole row, 'skip' = cover/blank cell).
    Cells without an entry default to 'chart'."""
    path = os.path.join(workdir, "decisions.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def cell_role(decisions, zip_name, frame_i, cell_i):
    try:
        return decisions[zip_name][str(frame_i)][cell_i]
    except (KeyError, IndexError):
        return "chart"


def stage_crop(args):
    with open(os.path.join(args.workdir, "boxes.json")) as f:
        results = json.load(f)
    decisions = load_decisions(args.workdir)
    todo = [(z, fr) for z, frames in sorted(results.items())
            if not args.only or args.only in z for fr in frames]
    for n, (z, frame) in enumerate(todo, 1):
        frame_i = int(FRAME_RE.match(frame["member"]).group(1))
        for cell_i, box in enumerate(frame["boxes"]):
            if cell_role(decisions, z, frame_i, cell_i) != "chart":
                continue
            out_name = cell_tif_name(z, frame_i, cell_i)
            out_path = os.path.join(GPO_DIR, out_name)
            if os.path.exists(out_path):
                print(f"[{n}/{len(todo)}] -- exists {out_name}")
                continue
            print(f"[{n}/{len(todo)}] -> {out_name}")
            tmp = out_path + ".part.tif"
            try:
                gdal.Translate(
                    tmp, vsi_path(z, frame["member"]), srcWin=box,
                    scaleParams=[[0, 255, 255, 0]],   # negative -> positive
                    creationOptions=CREATION_OPTS)
                os.replace(tmp, out_path)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            prev = os.path.join(args.workdir, out_name.replace(".tif", "_preview.png"))
            gdal.Translate(prev, out_path, width=1100, format="PNG")


def build_cell_note(base_note, seq, total):
    note = base_note.replace(
        OLD_NOTE,
        "chart exposure cropped from the pre-stitched fiche frame, negative "
        "inverted to positive (scripts/extract_gpo_microfiche.py)")
    if total > 1:
        note += (f"; OVERLAP-PAN exposure {seq}/{total} across the sheet - "
                 "siblings overlap, GCP enough of them to cover the chart")
    return note


def stage_apply(args):
    with open(os.path.join(args.workdir, "boxes.json")) as f:
        results = json.load(f)
    decisions = load_decisions(args.workdir)
    rows = dole_v2.load_rows(CSV_PATH)
    by_fname = {r.get("filename", "").strip(): r for r in rows}

    # chart cells per zip, in reading order across frames
    chart_cells = {}
    for z, frames in results.items():
        cells = []
        for fr in frames:
            frame_i = int(FRAME_RE.match(fr["member"]).group(1))
            for ci in range(len(fr["boxes"])):
                if cell_role(decisions, z, frame_i, ci) == "chart":
                    cells.append((frame_i, ci))
        chart_cells[z] = cells

    missing_tifs = [
        cell_tif_name(z, fi, ci)
        for z, cells in chart_cells.items() for fi, ci in cells
        if not os.path.exists(os.path.join(GPO_DIR, cell_tif_name(z, fi, ci)))
    ]
    if missing_tifs:
        print("Refusing to apply - cell tifs not on disk yet:")
        for t in missing_tifs:
            print("  ", t)
        return 1

    out_rows, changed, added = [], 0, 0
    for row in rows:
        fname = row.get("filename", "").strip()
        if fname not in chart_cells:
            out_rows.append(row)
            continue
        cells = chart_cells[fname]
        base_note = row.get("note", "") or ""
        if not cells:
            print(f"  !! no chart cells recorded for {fname}; row left untouched")
            out_rows.append(row)
            continue
        for k, (frame_i, cell_i) in enumerate(cells):
            new_name = cell_tif_name(fname, frame_i, cell_i)
            if new_name in by_fname:
                print(f"  !! {new_name} already a dole row; skipping")
                continue
            r = row if k == 0 else dict(row)
            r["filename"] = new_name
            r["note"] = build_cell_note(base_note, k + 1, len(cells))
            out_rows.append(r)
            if k == 0:
                changed += 1
            else:
                added += 1
    print(f"rows rewritten: {changed}, sibling rows added: {added}, "
          f"total {len(rows)} -> {len(out_rows)}")
    if args.dry_run:
        for r in out_rows:
            if "_micro_jp2_f" in r.get("filename", ""):
                print("   ", r["filename"])
        return 0
    if len(out_rows) < len(rows):
        print("Refusing to shrink row count.")
        return 1
    bak = CSV_PATH + ".gpo_apply.bak"
    with open(CSV_PATH, "rb") as f, open(bak, "wb") as g:
        g.write(f.read())
    import csv
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: ("" if r.get(k) is None else r.get(k, ""))
                      for k in dole_v2.V2_FIELDS} for r in out_rows])
    os.replace(tmp, CSV_PATH)
    print(f"wrote {CSV_PATH} (backup: {bak})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["detect", "crop", "apply"])
    ap.add_argument("--only", help="process only zips whose name contains S")
    ap.add_argument("--workdir", default=os.path.join(GPO_DIR, "_extract_work"),
                    help="where boxes.json + preview PNGs go")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.stage == "detect":
        return stage_detect(args)
    if args.stage == "crop":
        return stage_crop(args)
    return stage_apply(args)


if __name__ == "__main__":
    sys.exit(main())
