#!/usr/bin/env python3
"""Import the 2026-08 SD-card iFly sectional extractions into rawtiffs + dole.

RETRACTION 2026-08-27 (same day): the ifly_efb_card_2017 Western Aleutian 50
row + folder were withdrawn — the catalog already held eds 50 AND 51 as NARA
rg-237 full-res rows under per-half locations ("Western Aleutian Islands
East"/"...West"), invisible to this script's combined-name dedupe. Net
import: the 18 ifly_efb_card_2015 rows.

Source: /Volumes/projects/ifly_extract_sdcards/ (built by ifly_extract.py from
the 2026-08 sd_slurp card-image batch on /Volumes/ACASIS/All SD cards/2026-08-20/).
NOTE: ran 2026-08-27, after which the staging dir was folded into its permanent
homes and removed — extraction manifests sit in each rawtiffs parent, the source
bundle trios in rawtiffs_attic/ifly_sdcard_sec_2014-15_2017/ (extracted/derived
files always live in rawtiffs / rawtiffs_attic directly, no side staging).
Rerunning --copy would need the staging regenerated from the card images.
Twenty only-copy editions, verified absent 2026-08-27 from: the catalog, the
attic's acasis_imgcv2_FAA_originals full-res set, NARA rg-237 (2016+ only),
and all 1,517 verified findings of the 2026-07 hunt (incl. its wayback CDX
sweeps). Western Aleutian ed 51 was excluded (wayback capture exists; it
belongs to the worklist-07 queue), as were the nine editions the attic holds
as FAA originals.

Two parents, one per source card era, following the ifly_efb_card_2013
layout (one dole row per edition, filename `<parent>_<Chart>_<ed>.zip`,
TIFs in the same-stem folder; Western Aleutian E+W share one row):

    ifly_efb_card_2015  18 editions, 2014-15 cycle, card 003
                        (fil2 virtual offsets -> sequential-JPEG mapping)
    ifly_efb_card_2017  Western Aleutian Islands ed 50 (E+W), card 001

Usage:
    import_ifly_sdcards.py                   # dry-run: print planned rows
    import_ifly_sdcards.py --copy --write    # copy TIFs + append dole rows
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

STAGING = "/Volumes/projects/ifly_extract_sdcards"
RAWTIFFS = "/Volumes/projects/rawtiffs"
CSV_PATH = os.path.expanduser("~/archive.aero/master_dole_v2.csv")
BACKUP_DIR = os.path.expanduser("~/archive.aero-attic/csv-backups")

CARD_IMG = {
    "out_003": "003_NO_NAME_7.9GB_00000000.img",
    "out_001": "001_NO_NAME_7.9GB_df16946b.img",
}
PARENT = {"out_003": "ifly_efb_card_2015", "out_001": "ifly_efb_card_2017"}

NOTE = {
    "out_003": (
        "extracted 2026-08-27 from iFly EFB SD card (2026-08 sd_slurp batch, image "
        "003_NO_NAME_7.9GB_00000000.img on ACASIS) via scripts/ifly_extract.py; "
        "2015-era fil2 virtual offsets -> sequential-JPEG tile mapping, 257px tiles, "
        "EPSG:3395 ellipsoidal-Mercator bbox georef (2026-08-24 grid finding; KSEA landmark check on Seattle 88; reextracted 2026-08-27); "
        "iFly ~0.9x resample; no other copy found (catalog, attic FAA originals, "
        "NARA rg-237, 2026-07 hunt's 1,517 online findings incl. wayback CDX)"),
    "out_001": (
        "extracted 2026-08-27 from iFly EFB SD card (2026-08 sd_slurp batch, image "
        "001_NO_NAME_7.9GB_df16946b.img on ACASIS) via scripts/ifly_extract.py; "
        "new-format fil2, EPSG:3395 ellipsoidal-Mercator bbox georef (2026-08-24 grid finding; reextracted 2026-08-27); iFly ~0.9x resample; no other copy "
        "found (catalog, attic FAA originals, NARA rg-237, 2026-07 hunt's 1,517 "
        "online findings incl. wayback CDX; ed 51 by contrast HAS a wayback capture "
        "and is queued in worklist 07)"),
}

README = {
    "ifly_efb_card_2015": """iFly EFB SD-card extractions (2014-15 cycle sectionals)
=======================================================
18 chart editions from the 2014-15 gap era, missing from every other source
checked (catalog, attic FAA originals, NARA rg-237, the 2026-07 hunt's 1,517
verified online findings). Extracted 2026-08-27 by scripts/ifly_extract.py
from SD card 003 of the 2026-08 sd_slurp batch (ddrescue image on ACASIS,
complete, 0 bad sectors). This card era's fil2 index carries virtual offsets;
tiles were recovered by sequential-JPEG mapping (see ifly_extract.py).
These are DERIVED files (stitched 257px JPEG tiles, iFly ~0.9x resample) —
the same deliberate exception to the rawtiffs as-found rule as
ifly_efb_card_2013. If a full-res FAA original of any edition surfaces,
prefer it and retire the folder here.
Extraction records: manifest.jsonl here. Source bundle trios:
rawtiffs_attic/ifly_sdcard_sec_2014-15_2017/.
""",
    "ifly_efb_card_2017": """iFly EFB SD-card extractions (2017 Western Aleutian ed 50)
==========================================================
Western Aleutian Islands ed 50 (E+W sheets, one dole row per the 2013-card
convention), missing from every other source checked; ed 51 exists on
wayback and is queued in worklist 07 instead. Extracted 2026-08-27 by
scripts/ifly_extract.py from SD card 001 of the 2026-08 sd_slurp batch
(ddrescue image on ACASIS, complete). DERIVED files (stitched JPEG tiles,
iFly ~0.9x resample); a full-res FAA original supersedes this folder.
Extraction records: manifest.jsonl here. Source bundle trios:
rawtiffs_attic/ifly_sdcard_sec_2014-15_2017/.
""",
}

ALEUTIAN = {"Aleutian Islands E", "Aleutian Islands W"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_groups():
    """manifests -> {(outdir, loc, ed): [recs]}, Aleutian E+W folded."""
    groups = defaultdict(list)
    for outdir in ("out_003", "out_001"):
        with open(os.path.join(STAGING, outdir, "manifest.jsonl")) as fh:
            for line in fh:
                rec = json.loads(line)
                if not rec["base"].startswith("Sec_"):
                    continue
                loc = rec["location"]
                if loc in ALEUTIAN:
                    loc = "Western Aleutian Islands"
                rec["_out"] = rec["out"] if os.path.isabs(rec["out"]) \
                    else os.path.join(STAGING, rec["out"])
                groups[(outdir, loc, rec["edition"])].append(rec)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--copy", action="store_true", help="copy TIFs into rawtiffs")
    ap.add_argument("--write", action="store_true", help="append rows to master dole")
    args = ap.parse_args()

    rows = dole_v2.load_rows(CSV_PATH)
    by_loc = defaultdict(list)
    for r in rows:
        by_loc[norm(r["location"])].append(r)
    existing_filenames = {r["filename"] for r in rows}

    planned = []
    for (outdir, loc, ed), recs in sorted(load_groups().items()):
        sibs = by_loc.get(norm(loc))
        if not sibs:
            raise SystemExit(f"no catalog rows for location {loc!r} - naming mismatch?")
        loc_exact = sibs[0]["location"]

        dates = {datetime.datetime.strptime(r["start_date"], "%m/%d/%Y").date()
                 for r in recs}
        if len(dates) != 1:
            raise SystemExit(f"{loc} {ed}: sheets disagree on start date: {dates}")
        date = dates.pop().isoformat()

        later = sorted(d for d in {r["date"] for r in sibs} if d > date)
        if not later:
            raise SystemExit(f"{loc} {ed}: no later catalog edition to derive end_date")
        end_date = later[0]

        cutline = Counter(r["cutline"] for r in sibs if r["cutline"]).most_common(1)
        cutline = cutline[0][0] if cutline else ""

        stem = f"{PARENT[outdir]}_{loc_exact.replace(' ', '_')}_{ed}"
        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({
            "filename": stem + ".zip",
            "location": loc_exact,
            "date": date,
            "end_date": end_date,
            "edition": ed,
            "note": NOTE[outdir],
            "cutline": cutline,
        })
        planned.append((row, outdir, stem, recs))

    print(f"{len(planned)} rows planned:")
    for row, outdir, stem, recs in planned:
        dupe = "  (ALREADY IN CSV - will skip)" if row["filename"] in existing_filenames else ""
        exp = recs[0]["expire_date"]
        print(f"  {row['location']:<28} ed {row['edition']:<4} {row['date']} -> "
              f"{row['end_date']} (card expire {exp})  {len(recs)} tif(s)  "
              f"cutline={row['cutline'] or '-'}{dupe}")

    if args.copy:
        copied = skipped = 0
        for row, outdir, stem, recs in planned:
            parent_dir = os.path.join(RAWTIFFS, PARENT[outdir])
            os.makedirs(parent_dir, exist_ok=True)
            readme = os.path.join(parent_dir, "README.txt")
            if not os.path.exists(readme):
                with open(readme, "w") as fh:
                    fh.write(README[PARENT[outdir]])
            dst_dir = os.path.join(parent_dir, stem)
            os.makedirs(dst_dir, exist_ok=True)
            for rec in recs:
                dst = os.path.join(dst_dir, os.path.basename(rec["_out"]))
                if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(rec["_out"]):
                    skipped += 1
                    continue
                shutil.copy2(rec["_out"], dst)
                copied += 1
        print(f"copied {copied} tifs into rawtiffs ({skipped} already present)")

    if args.write:
        new_rows = [row for row, *_ in planned
                    if row["filename"] not in existing_filenames]
        if not new_rows:
            print("no new rows to write")
            return
        os.makedirs(BACKUP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup = os.path.join(BACKUP_DIR, f"master_dole_v2.csv.pre_ifly_sdcards_{stamp}.bak")
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
