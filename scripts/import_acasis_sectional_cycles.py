#!/usr/bin/env python3
"""Append dole rows for the charts staged by stage_acasis_sectional_cycles.py.

Reads rawtiffs/acasis_sectional_cycles/manifest.jsonl and writes one row per
staged (location, cycle).  The row filename is the container directory name plus
'.zip' -- slicer.py's resolve_filename and audit_disk_vs_dole.py both address an
extracted container that way, and it is what lets the FAA filenames inside stay
verbatim across five cycles that all use the same basenames.

Run with ~/venv/bin/python from the repo root; --write to actually append.
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dole_v2

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "master_dole_v2.csv"
STAGE = Path("/Volumes/projects/rawtiffs/acasis_sectional_cycles")
BACKUP_DIR = Path.home() / "archive.aero-attic" / "csv-backups"

CUTLINE = {
    "Cincinnati": "sectional/cincinnati",
    "Los Angeles": "sectional/los_angeles",
    "Las Vegas": "sectional/las_vegas",
    "Honolulu Inset": "sectional/honolulu_inset",
    "Mariana Islands Inset": "sectional/mariana_islands_inset",
    "Samoan Islands Inset": "sectional/samoan_islands_inset",
    "Western Aleutian Islands East": "sectional/western_aleutian_islands_east",
    "Western Aleutian Islands West": "sectional/western_aleutian_islands_west",
}

# Why each chart is the only copy ever seen, per CLAUDE.md's admission test.
BUNDLED = ("this sheet ships only inside the cycle's Hawaiian_Islands.zip / "
           "Western_Aleutian_Islands.zip, and Wayback captured no sectional-files "
           "zip at all for this cycle (CDX for aeronav.faa.gov/visual/%s/"
           "sectional-files* returns only the directory listing%s), so no separately "
           "georeferenced copy of it exists anywhere else we have looked")
TRUNCATED = ("this is the cycle gap the 2026-07 harvest could not close -- Wayback holds "
             "no sectional-files zip for this sheet and its only PDF capture is a "
             "5,242,880-byte truncation (no %%%%EOF), the archive.org "
             "faa-visual-flight-rules-charts item carries a different cycle, and "
             "aeronav no longer serves %s")

NOTE_HEAD = (
    "recovered 2026-08-28 from /Volumes/ACASIS root set '%s' -- a whole-cycle FAA "
    "sectional-files download (all 57 charts, tif+tfw+htm) kept by the iFly GPS/EFB "
    "build machine's operator. FAA original georeferenced GeoTIFF, LZW palette, "
    "LCC/NAD83, with its .tfw and FAA .htm alongside; date/end_date taken from the "
    ".htm Beginning_Date/Ending_Date (end_date = FAA Ending_Date + 1 day, the "
    "catalog's next-cycle-start convention). Embedded georef, no GCP work. "
    "Only copy ever seen: %s; absent from the catalog on (location, date) and from "
    "rawtiffs and rawtiffs_attic by exact byte size across all 13,377 tif/zip files. "
    "Staged rawtiffs/acasis_sectional_cycles/%s/ with FAA filenames verbatim; "
    "the row filename is that container, which holds no .zip -- the source was a "
    "whole-cycle archive, not a per-chart one."
)

NAMED = {"Cincinnati", "Los Angeles", "Las Vegas"}


def cycle_slug(date):
    return "%s-%s-%s" % (date[5:7], date[8:10], date[:4])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows = dole_v2.load_rows(CSV_PATH)
    existing_files = {r["filename"] for r in rows}
    existing_keys = {(r["location"], r["date"]) for r in rows}

    planned, problems = [], []
    for line in open(STAGE / "manifest.jsonl"):
        m = json.loads(line)
        c, cont = m["chart"], m["container"]
        loc, date = c["location"], c["date"]
        cut = CUTLINE[loc]
        if not (REPO / "shapefiles" / (cut + ".shp")).exists():
            problems.append("%s: no cutline shapefile %s" % (cont, cut))
        if not (STAGE / cont).is_dir():
            problems.append("%s: staged directory missing" % cont)
        if (loc, date) in existing_keys:
            problems.append("%s: (%s, %s) already in the catalog" % (cont, loc, date))

        slug = cycle_slug(date)
        if loc in NAMED:
            why = TRUNCATED % ("aeronav.faa.gov/visual/%s/sectional-files/%s.zip"
                               % (slug, loc.replace(" ", "_")))
        else:
            extra = (" plus the 20 per-chart zips the archive already holds"
                     if date == "2026-03-19" else "")
            why = BUNDLED % (slug, extra)

        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({
            "filename": cont + ".zip",
            "download_link": "",
            "location": loc,
            "date": date,
            "end_date": c["end_date"],
            "edition": "Unknown",
            "note": NOTE_HEAD % (m["files"][0]["source_container"], why, cont),
            "cutline": cut,
        })
        planned.append(row)

    planned.sort(key=lambda r: (r["date"], r["location"]))
    print("%d rows planned:" % len(planned))
    for r in planned:
        dup = "  (ALREADY IN CSV - will skip)" if r["filename"] in existing_files else ""
        print("  %-30s %s -> %s  cutline=%-40s %s%s"
              % (r["location"], r["date"], r["end_date"], r["cutline"],
                 r["filename"], dup))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1

    if not a.write:
        print("\n(dry run; pass --write to append)")
        return 0

    new_rows = [r for r in planned if r["filename"] not in existing_files]
    if not new_rows:
        print("no new rows to write")
        return 0
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = BACKUP_DIR / ("master_dole_v2.csv.pre_acasis_sectional_cycles_%s.bak" % stamp)
    shutil.copy2(CSV_PATH, backup)
    tmp = str(CSV_PATH) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in dole_v2.V2_FIELDS})
        w.writerows(new_rows)
    os.replace(tmp, CSV_PATH)
    print("\nwrote %d rows (backup: %s); dole now %d rows"
          % (len(new_rows), backup, len(rows) + len(new_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
