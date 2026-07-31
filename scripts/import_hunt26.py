#!/usr/bin/env python3
"""Import the HUNT26 batch (2026-07 download-queue hunt, 235 new charts) into
master_dole_v2.csv.

Reads worklists/data/hunt26_plan.csv — the reviewed move+import plan produced
in the 2026-07-25 session (rid → filename/location/date/end_date/edition/note/
cutline/lcc, one row per file moved to /Volumes/projects/rawtiffs/hunt_2026-07/).
Rows with import=no (chart backs, misidentified files, reference pages,
separation plates) are skipped.

Idempotent: rows whose filename already exists in the dole are skipped, so the
script can be re-run after a partial import. A timestamped backup of the CSV is
written to ~/archive.aero-attic/csv-backups/ first.

GCPs are left blank on every imported row: the FAA-era wayback zips carry tfw
georeferencing (status 'georef', sliceable as-is); everything else is hand-GCP
work surfaced in the georef tool's HUNT26 worklist items.
"""
import csv
import datetime
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dole_v2

CSV_PATH = os.path.join(REPO, "master_dole_v2.csv")
PLAN_PATH = os.path.join(REPO, "worklists", "data", "hunt26_plan.csv")
BACKUP_DIR = os.path.expanduser("~/archive.aero-attic/csv-backups")


def main():
    plan = [r for r in csv.DictReader(open(PLAN_PATH)) if r["import"] == "yes"]
    rows = dole_v2.load_rows(CSV_PATH)
    existing = {r["filename"] for r in rows}

    new_rows = []
    for p in plan:
        if p["filename"] in existing:
            continue
        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({
            "filename": p["filename"],
            "download_link": p["download_link"],
            "location": p["location"],
            "date": p["date"],
            "end_date": p["end_date"],
            "edition": p["edition"],
            "note": p["note"],
            "cutline": p["cutline"],
            "lcc_lat1": p["lcc_lat1"],
            "lcc_lat2": p["lcc_lat2"],
            "lcc_lat0": p["lcc_lat0"],
            "lcc_lon0": p["lcc_lon0"],
        })
        new_rows.append(row)

    if not new_rows:
        print("nothing to import (all plan rows already present)")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"master_dole_v2.csv.pre_hunt26_{stamp}.bak")
    shutil.copy2(CSV_PATH, backup)
    print(f"backup: {backup}")

    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in dole_v2.V2_FIELDS})
        w.writerows(new_rows)
    os.replace(tmp, CSV_PATH)
    print(f"imported {len(new_rows)} rows "
          f"({len(plan) - len(new_rows)} already present); "
          f"dole now {len(rows) + len(new_rows)} rows")


if __name__ == "__main__":
    main()
