#!/usr/bin/env python3
"""Import the iFly-card 2013 sectional extractions into rawtiffs + master dole.

Source: /Volumes/projects/ifly_extract/ (built by scripts/ifly_extract.py from
the iFly EFB data card, 2026-08-19). These 34 editions (2012-13 cycle) exist
nowhere else we could find: absent from the catalog, from NARA rg-237 (fully
enumerated 2026-07, reaches back only to 2016), from the 2026-07 hunt's 1,490
online findings, and from Wayback (live CDX sweep of the old FAA
sectional_files/ layout, 2026-08-19: zero captures of any target edition).

Layout mirrors the wayback-zip convention: one dole row per edition whose
filename is `ifly_efb_card_2013_<Chart>_<ed>.zip`; the TIFs live in the
same-stem folder under rawtiffs/ifly_efb_card_2013/ (no actual zip — the
slicer's resolve_filename finds the folder via its index). Hawaiian insets
ride inside the Hawaiian_Islands folder; Western Aleutian E+W share one row.

Usage:
    import_ifly_card.py            # dry-run: print planned rows/copies
    import_ifly_card.py --only albuquerque --copy --write   # one group
    import_ifly_card.py --copy --write                      # everything
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
from collections import Counter, defaultdict

import dole_v2

STAGING = "/Volumes/projects/ifly_extract"
RAWTIFFS = "/Volumes/projects/rawtiffs"
PARENT = "ifly_efb_card_2013"
CSV_PATH = os.path.expanduser("~/archive.aero/master_dole_v2.csv")
BACKUP_DIR = os.path.expanduser("~/archive.aero-attic/csv-backups")

# inset sheets ride in the Hawaiian Islands folder, like the FAA zips
FOLD_INTO_HAWAII = {"Honolulu Inset", "Mariana Islands Inset", "Samoan Islands Inset"}

NOTE = ("extracted 2026-08-19 from iFly EFB data card via scripts/ifly_extract.py "
        "(level-1 JPEG tiles stitched; georef copied from FAA GeoTIFF via .gti; "
        "iFly ~0.9x resample, ~70.5 m/px vs 63.5 original); registration verified "
        "vs rawtiffs FAA original Atlanta 91 N (bbox exact, <1 px shift); no other "
        "copy found (wayback CDX, NARA rg-237, 2026-07 hunt)")

README = """iFly EFB card extractions (2012-13 cycle sectionals)
=====================================================
34 chart editions missing from every other source we could find (wayback,
NARA rg-237, the 2026-07 hunt). Extracted 2026-08-19 by scripts/ifly_extract.py
from the iFly 700-era data card; see the dole rows' note column and
worklists for provenance. These are DERIVED files (stitched JPEG tiles,
~70.5 m/px), the one deliberate exception to the "rawtiffs holds sources
exactly as downloaded" rule. If a full-res FAA original of any edition ever
surfaces, prefer it and retire the folder here.
Staging + manifest: /Volumes/projects/ifly_extract/ (keep until superseded).
"""


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_groups():
    """manifest.jsonl -> {(loc, ed): [recs]} with insets folded into Hawaii."""
    groups = defaultdict(list)
    with open(os.path.join(STAGING, "manifest.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            loc = rec["location"]
            if loc in FOLD_INTO_HAWAII:
                loc = "Hawaiian Islands"
            groups[(loc, rec["edition"])].append(rec)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="restrict to one location (normalized substring)")
    ap.add_argument("--copy", action="store_true", help="copy TIFs into rawtiffs")
    ap.add_argument("--write", action="store_true", help="append rows to master dole")
    args = ap.parse_args()

    rows = dole_v2.load_rows(CSV_PATH)
    by_loc = defaultdict(list)
    for r in rows:
        by_loc[norm(r["location"])].append(r)
    existing_filenames = {r["filename"] for r in rows}

    groups = load_groups()
    planned = []
    for (loc, ed), recs in sorted(groups.items()):
        if args.only and args.only.lower() not in norm(loc):
            continue
        sibs = by_loc.get(norm(loc))
        if not sibs:
            raise SystemExit(f"no catalog rows for location {loc!r} - naming mismatch?")
        loc_exact = sibs[0]["location"]

        dates = {datetime.datetime.strptime(r["start_date"], "%A, %B %d, %Y").date()
                 for r in recs}
        if len(dates) != 1:
            raise SystemExit(f"{loc} {ed}: halves disagree on start date: {dates}")
        date = dates.pop().isoformat()

        later = sorted(d for d in {r["date"] for r in sibs} if d > date)
        if not later:
            raise SystemExit(f"{loc} {ed}: no later catalog edition to derive end_date")
        end_date = later[0]
        expire = recs[0]["expire_date"]

        cutline = Counter(r["cutline"] for r in sibs if r["cutline"]).most_common(1)
        cutline = cutline[0][0] if cutline else ""

        stem = f"{PARENT}_{loc_exact.replace(' ', '_')}_{ed}"
        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({
            "filename": stem + ".zip",
            "location": loc_exact,
            "date": date,
            "end_date": end_date,
            "edition": ed,
            "note": NOTE,
            "cutline": cutline,
        })
        planned.append((row, stem, recs, expire))

    print(f"{len(planned)} rows planned:")
    for row, stem, recs, expire in planned:
        dupe = "  (ALREADY IN CSV - will skip)" if row["filename"] in existing_filenames else ""
        print(f"  {row['location']:<26} ed {row['edition']:<3} {row['date']} -> "
              f"{row['end_date']} (card expire {expire})  "
              f"{len(recs)} tif(s)  cutline={row['cutline'] or '-'}{dupe}")

    if args.copy:
        parent_dir = os.path.join(RAWTIFFS, PARENT)
        os.makedirs(parent_dir, exist_ok=True)
        readme = os.path.join(parent_dir, "README.txt")
        if not os.path.exists(readme):
            with open(readme, "w") as fh:
                fh.write(README)
        copied = skipped = 0
        for row, stem, recs, _ in planned:
            dst_dir = os.path.join(parent_dir, stem)
            os.makedirs(dst_dir, exist_ok=True)
            for rec in recs:
                dst = os.path.join(dst_dir, os.path.basename(rec["out"]))
                if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(rec["out"]):
                    skipped += 1
                    continue
                shutil.copy2(rec["out"], dst)
                copied += 1
        print(f"copied {copied} tifs into {parent_dir} ({skipped} already present)")

    if args.write:
        new_rows = [row for row, *_ in planned
                    if row["filename"] not in existing_filenames]
        if not new_rows:
            print("no new rows to write")
            return
        os.makedirs(BACKUP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup = os.path.join(BACKUP_DIR, f"master_dole_v2.csv.pre_ifly_card_{stamp}.bak")
        shutil.copy2(CSV_PATH, backup)
        tmp = CSV_PATH + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in dole_v2.V2_FIELDS})
            w.writerows(new_rows)
        os.replace(tmp, CSV_PATH)
        print(f"wrote {len(new_rows)} rows (backup: {backup}); "
              f"dole now {len(rows) + len(new_rows)} rows")


if __name__ == "__main__":
    main()
