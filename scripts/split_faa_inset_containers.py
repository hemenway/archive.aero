#!/usr/bin/env python3
"""Give the Hawaiian insets and both Western Aleutian halves their own dole rows.

The FAA ships four sheets inside Hawaiian_Islands.zip (the main sheet plus the
Honolulu, Mariana and Samoan insets) and two inside Western_Aleutian_Islands.zip
(East + West).  A `.zip` row resolves to EVERY tif in its container and applies
that row's single cutline to all of them, so for the cycles catalogued this way:

  * Mariana and Samoa fall outside `sectional/hawaiian_islands` and clip to
    nothing -- the archive has no southern-hemisphere coverage in those eras;
  * the Honolulu inset lands on Oahu at 2x scale, double-layering the main sheet;
  * the West Aleutian sheet clips to 0.1% of its area against
    `sectional/western_aleutian_islands_east` (96.5% against its own cutline).

Fix: move each of those five sheets into its own per-chart container, and give it
its own row and cutline -- the shape the 2016-2024 catalog used and the shape the
2026-08-28 ACASIS batch restored for 2025-11-27 .. 2026-03-19.

The main Hawaiian sheet STAYS in its original container, so that row needs no
change and its zip stays claimed.  The Western Aleutian container empties out and
its zip is recorded in worklists/superseded_sources.csv.

Sheets are MOVED, not copied: the `.zip` beside each container is the as-found
artifact, and the extracted directory is derived from it.

Run with ~/venv/bin/python from the repo root; --write to apply.
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
RAW = Path("/Volumes/projects/rawtiffs")
SPLITS = RAW / "faa_chart_splits"
BACKUP_DIR = Path.home() / "archive.aero-attic" / "csv-backups"
LEDGER = REPO / "worklists" / "superseded_sources.csv"

CYCLES = ["2025-02-20", "2025-04-17", "2025-06-12", "2025-08-07", "2025-10-02",
          "2026-05-14", "2026-07-09", "2026-09-03",
          # The 2013 iFly-card import (import_ifly_card.py, 2026-08-19) folded
          # the three inset sheets into the Hawaiian Islands row the same
          # combined way: ifly_efb_card_2013_Hawaiian_Islands_88.zip holds
          # "Hawaiian Islands 88.tif" plus "* Inset 88.tif" (2026-09-08 audit).
          "2013-05-02"]

# parent row location -> [(member stem PREFIX, new dole location, cutline slug)]
# Members are matched by prefix so both FAA ("Honolulu Inset SEC.tif") and
# card-salvage ("Honolulu Inset 88.tif") names resolve.
MEMBERS = {
    "Hawaiian Islands": [
        ("Honolulu Inset", "Honolulu Inset", "sectional/honolulu_inset"),
        ("Mariana Islands Inset", "Mariana Islands Inset", "sectional/mariana_islands_inset"),
        ("Samoan Islands Inset", "Samoan Islands Inset", "sectional/samoan_islands_inset"),
    ],
    "Western Aleutian Islands": [
        ("Western Aleutian Islands East", "Western Aleutian Islands East",
         "sectional/western_aleutian_islands_east"),
        ("Western Aleutian Islands West", "Western Aleutian Islands West",
         "sectional/western_aleutian_islands_west"),
    ],
}
# The parent row survives for Hawaiian Islands (keeps the main sheet) and is
# dropped for Western Aleutian Islands (both halves move out).
DROP_PARENT = {"Western Aleutian Islands"}

NOTE = (
    "split out {when} from the container {parent} so it can carry its own "
    "cutline. A .zip row resolves to every tif in its container and applies one "
    "cutline to all of them, so this sheet was being clipped against "
    "{parent_cut}: {harm} Same bytes as the container member ({source}), moved into "
    "rawtiffs/faa_chart_splits/{cont}/ with the filename verbatim; the .zip "
    "beside the original container remains the as-found artifact. Restores the "
    "per-inset/per-half row shape the catalog used through 2024-12-26."
)
SOURCE_FAA = "FAA distribution GeoTIFF, LZW palette, LCC/NAD83, georeference embedded plus .tfw"

README_ROOT = """faa_chart_splits
================

Per-chart containers split out of multi-sheet source containers by
scripts/split_faa_inset_containers.py, so each sheet can carry its own
catalog row and cutline (a .zip row applies one cutline to every tif in
its container). Files are MOVED here from the extracted parent directory
with their names verbatim; the parent .zip stays beside the parent
directory as the as-found artifact. Every container here has a README.txt
naming its parent and a manifest.jsonl (one line per file: original path,
parent container, size, TIFF structure check). Provenance of the bytes
themselves is the parent row's note (FAA download, iFly-card extraction...).
"""

README_CONTAINER = """{cont}
{underline}

Split out {when} from parent container {parent} (catalog row
{parent_loc} {cycle}) by scripts/split_faa_inset_containers.py.
Sheet: {location}. Files moved with names verbatim; see manifest.jsonl.
Source of the bytes: {source}. Condition: as extracted from the parent
container; the whole member set of the parent was examined.
"""


def tiff_ok(path):
    """Cheap structural check: TIFF/BigTIFF magic and a readable first IFD."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
        if head[:4] in (b"II*\x00", b"MM\x00*"):
            import struct
            le = head[:2] == b"II"
            off = struct.unpack("<I" if le else ">I", head[4:8])[0]
            with open(path, "rb") as f:
                f.seek(off)
                n = f.read(2)
            return len(n) == 2 and struct.unpack("<H" if le else ">H", n)[0] > 0
        return head[:4] in (b"II+\x00", b"MM\x00+")  # BigTIFF: magic only
    except OSError:
        return False
HARM = {
    "Honolulu Inset": "it landed on Oahu at 2x scale, double-layering the main sheet.",
    "Mariana Islands Inset": "it fell entirely outside and clipped to nothing.",
    "Samoan Islands Inset": "it fell entirely outside and clipped to nothing, "
                            "leaving the era with no southern-hemisphere coverage.",
    "Western Aleutian Islands East": "that is its own cutline, so it was unharmed; "
                                     "it moves for symmetry with the West half.",
    "Western Aleutian Islands West": "only 0.1% of its area survived, against 96.5% "
                                     "under its own cutline.",
}


def container(cycle, location, parent_filename=""):
    prefix = "iflysplit" if parent_filename.lower().startswith("ifly") else "faasplit"
    return "%s_%s-%s-%s_%s" % (prefix, cycle[5:7], cycle[8:10], cycle[:4],
                               location.replace(" ", "_"))


def member_files(src_dir, prefix):
    """The member tif whose stem starts with `prefix`, plus its sidecars."""
    tifs = sorted(p for p in src_dir.iterdir()
                  if p.suffix.lower() == ".tif" and p.stem.startswith(prefix))
    if len(tifs) != 1:
        return None, []
    tif = tifs[0]
    sidecars = [p for p in src_dir.iterdir()
                if p != tif and p.stem == tif.stem and p.suffix.lower() in (".tfw", ".htm", ".jgw", ".prj")]
    return tif, sidecars


def find_dir(stem):
    for root, dirs, _f in os.walk(RAW):
        if os.path.basename(root) == stem:
            return Path(root)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows = dole_v2.load_rows(CSV_PATH)
    by_key = {(r["location"], r["date"]): r for r in rows}

    plan, problems, done = [], [], []
    for cycle in CYCLES:
        for parent_loc, members in MEMBERS.items():
            # Idempotent: a cycle whose member rows all exist was split on an
            # earlier run (2026-08-29 for the 2025-26 FAA cycles); skip it.
            if all((new_loc, cycle) in by_key for _p, new_loc, _c in members):
                done.append("%s %s" % (cycle, parent_loc))
                continue
            prow = by_key.get((parent_loc, cycle))
            if prow is None:
                if any((new_loc, cycle) in by_key for _p, new_loc, _c in members):
                    problems.append("%s %s: parent row gone but only some member rows exist" % (cycle, parent_loc))
                continue  # no such cycle for this location (e.g. no 2013 Western Aleutian card row)
            stem = prow["filename"][:-4] if prow["filename"].endswith(".zip") else None
            src = find_dir(stem) if stem else None
            if src is None:
                problems.append("%s %s: container %s not on disk" % (cycle, parent_loc, stem))
                continue
            for member_prefix, new_loc, cut in members:
                tif, sidecars = member_files(src, member_prefix)
                if tif is None:
                    problems.append("%s %s: no single member starting with %r in %s"
                                    % (cycle, parent_loc, member_prefix, src.name))
                    continue
                if (new_loc, cycle) in by_key:
                    problems.append("%s %s: row already exists" % (cycle, new_loc))
                    continue
                if not (REPO / "shapefiles" / (cut + ".shp")).exists():
                    problems.append("%s: no cutline %s" % (new_loc, cut))
                    continue
                if not tiff_ok(tif):
                    problems.append("%s %s: %s is not a structurally valid TIFF" % (cycle, parent_loc, tif.name))
                    continue
                plan.append({"cycle": cycle, "parent_loc": parent_loc, "parent_row": prow,
                             "src_dir": src, "tif": tif, "sidecars": sidecars,
                             "member_stem": tif.stem,
                             "location": new_loc, "cutline": cut,
                             "container": container(cycle, new_loc, prow["filename"])})

    drops = [by_key[(loc, c)] for c in CYCLES for loc in DROP_PARENT
             if (loc, c) in by_key and any(p["parent_loc"] == loc and p["cycle"] == c for p in plan)]
    print("%d sheets to split out; %d combined Western Aleutian rows to drop; "
          "%d cycle/location pairs already split" % (len(plan), len(drops), len(done)))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1
    if not a.write:
        for p in plan[:6]:
            print("  would move %-34s -> %s" % (p["member_stem"] + ".tif", p["container"]))
        print("  ... (%d more)" % max(0, len(plan) - 6))
        print("\n(pass --write to apply)")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = BACKUP_DIR / ("master_dole_v2.csv.pre_inset_split_%s.bak" % stamp)
    shutil.copy2(CSV_PATH, backup)

    SPLITS.mkdir(parents=True, exist_ok=True)
    if not (SPLITS / "README.txt").exists():
        (SPLITS / "README.txt").write_text(README_ROOT)
    when = datetime.date.today().isoformat()
    new_rows, ledger, moved = [], [], 0
    for p in plan:
        dest = SPLITS / p["container"]
        dest.mkdir(parents=True, exist_ok=True)
        is_ifly = p["parent_row"]["filename"].lower().startswith("ifly")
        source = ("iFly-card extraction, see the parent row's note" if is_ifly else SOURCE_FAA)
        manifest_lines = []
        for src_file in [p["tif"]] + p["sidecars"]:
            original = str(src_file.relative_to(RAW))
            target = dest / src_file.name
            shutil.move(str(src_file), str(target))
            moved += 1
            manifest_lines.append({
                "file": src_file.name,
                "original_path": original,
                "parent_container": p["parent_row"]["filename"],
                "parent_row": {"location": p["parent_loc"], "date": p["cycle"]},
                "live": True,
                "bytes": target.stat().st_size,
                "verification": ("tiff structure ok" if src_file.suffix.lower() == ".tif" and tiff_ok(target)
                                 else "sidecar, not checked" if src_file.suffix.lower() != ".tif"
                                 else "TIFF STRUCTURE FAILED"),
                "moved": when,
            })
        with open(dest / "manifest.jsonl", "a", encoding="utf-8") as mf:
            for line in manifest_lines:
                mf.write(json.dumps(line) + "\n")
        (dest / "README.txt").write_text(README_CONTAINER.format(
            cont=p["container"], underline="=" * len(p["container"]), when=when,
            parent=p["parent_row"]["filename"], parent_loc=p["parent_loc"], cycle=p["cycle"],
            location=p["location"], source=source))
        # Provenance: keep the parent's own note (card extraction, download
        # hunt...) in front of the split note so the chain stays readable.
        parent_note = (p["parent_row"].get("note") or "").strip()
        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({
            "filename": p["container"] + ".zip",
            "download_link": "",
            "location": p["location"],
            "date": p["cycle"],
            "end_date": p["parent_row"]["end_date"],
            "edition": (p["parent_row"].get("edition") or "Unknown").strip() or "Unknown",
            "src_crs": p["parent_row"].get("src_crs", ""),
            "cutline": p["cutline"],
            "note": (parent_note + "; " if parent_note else "") + NOTE.format(
                when=when, parent=p["parent_row"]["filename"],
                parent_cut=p["parent_row"]["cutline"], source=source,
                harm=HARM[p["location"]], cont=p["container"]),
        })
        new_rows.append(row)

    # Emptied Western Aleutian containers: record their zip, drop the empty dir.
    drop_ids = {id(r) for r in drops}
    for r in drops:
        stem = r["filename"][:-4]
        d = find_dir(stem)
        if d is not None and not any(d.iterdir()):
            d.rmdir()
        z = None
        for root, _ds, fs in os.walk(RAW):
            if stem + ".zip" in fs:
                z = Path(root) / (stem + ".zip")
                break
        if z is not None:
            ledger.append((str(z.relative_to(RAW)), r["filename"], r["location"],
                           r["date"], "(row dropped)",
                           "container split into per-chart rows; both halves now "
                           "carry their own cutline"))

    kept = [r for r in rows if id(r) not in drop_ids]
    tmp = str(CSV_PATH) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in dole_v2.V2_FIELDS})
        w.writerows(new_rows)
    os.replace(tmp, CSV_PATH)

    if ledger:
        with open(LEDGER, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(sorted(set(ledger)))

    print("\nmoved %d files into %d split containers; +%d rows, -%d rows; dole %d -> %d"
          % (moved, len(plan), len(new_rows), len(drops), len(rows),
             len(kept) + len(new_rows)))
    print("backup: %s" % backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
