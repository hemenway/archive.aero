#!/usr/bin/env python3
"""Repoint 2025-11-27 / 2026-01-22 / 2026-03-19 dole rows at FAA originals.

Those three cycles are catalogued from 300-dpi conversions of Wayback/archive.org
PDFs (and, for eight 2026-01-22 rows, from lossy iFly-card rebuilds whose own
notes say to retire them if an FAA original surfaces).  The ACASIS root's
whole-cycle sets hold the FAA distribution GeoTIFF of every one.  This is a
source swap, not a new chart: each row keeps its location, date and cutline and
gets a new filename, and nothing new is added to the catalog.

Also drops the combined "Western Aleutian Islands" row for those three cycles.
It is a single 300-dpi PDF conversion cut to `western_aleutian_islands_east`, so
once the FAA's own East and West halves are catalogued (2026-08-28 batch) it
double-covers the east half at half the resolution and adds nothing.

Rows repointed here carry GCPs transferred from the 300-dpi conversion grid,
which is a DIFFERENT raster size from the FAA original.  Leaving them would warp
the new source through the old grid — the null-island trap in a subtler form —
so all 16 GCP columns and src_crs are cleared; the FAA GeoTIFF's georeference is
embedded.

Two charts carry a bad FAA .htm and are dated from their set's cycle instead
(both verified by content diff against a known copy of that cycle, see NOTES).

Run with ~/venv/bin/python from the repo root; --stage copies files, --write
updates the catalog.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dole_v2

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "master_dole_v2.csv"
RAW = Path("/Volumes/projects/rawtiffs")
STAGE = RAW / "acasis_sectional_cycles"
BACKUP_DIR = Path.home() / "archive.aero-attic" / "csv-backups"
LEDGER = REPO / "worklists" / "superseded_sources.csv"

DATE_RE = re.compile(r"(Publication|Beginning|Ending)_Date:\s*(\d{8})")

SETS = {
    "2025-11-27": ("/Volumes/ACASIS/recovery/Sectional",
                   "Sectional (4) nov 2025.zip (extracted at recovery/Sectional)"),
    "2026-01-22": ("/Volumes/ACASIS/Sectional (3) jan", "Sectional (3) jan"),
    "2026-03-19": ("/Volumes/ACASIS/Sectional (2) march", "Sectional (2) march"),
}
NEXT_CYCLE = {"2025-11-27": "2026-01-22", "2026-01-22": "2026-03-19",
              "2026-03-19": "2026-05-14"}

# The only two .htm anomalies in all 285 charts of the five ACASIS root sets.
# Both are metadata-only: each file's cycle identity was confirmed by diffing it
# against a copy of that cycle the archive already held.
HTM_ANOMALY = {
    ("2026-01-22", "Lake Huron SEC"):
        "the FAA .htm on this file is misstamped Publication/Beginning_Date "
        "20260319 with Ending_Date 20260318 (begin after end); its Process_Date "
        "is 20251230 and it is the 2026-01-22 edition, confirmed by content: it "
        "renders Cheboygan County (SLH) as AWOS-3 118.175, matching the "
        "01-22-2026 Wayback capture, where the 2026-03-19 sheet reads AWOS-3P. "
        "date/end_date therefore come from the set's cycle, not the .htm",
    ("2026-03-19", "Atlanta SEC"):
        "the FAA .htm on this file carries a stale Ending_Date 20260318 against "
        "a correct Beginning_Date 20260319; it is the 2026-03-19 edition, "
        "confirmed by content: the Chilton County (02A) label position and the "
        "959 (320) obstruction match the archive.org 03-19-2026 sheet and differ "
        "from the 2026-01-22 one. end_date therefore comes from the next cycle, "
        "not the .htm",
}

WAI_COMBINED = "Western Aleutian Islands"

GCP_FIELDS = [f"gcp{i}_{k}" for i in (1, 2, 3, 4) for k in ("px", "py", "lat", "lon")]

NOTE = (
    "source upgraded 2026-08-28 to the FAA original: replaces {old} with the "
    "distribution GeoTIFF from /Volumes/ACASIS root set '{set}', a whole-cycle "
    "FAA sectional-files download (all 57 charts, tif+tfw+htm) kept by the iFly "
    "GPS/EFB build machine's operator. {gain} LZW palette, LCC/NAD83, "
    "georeference embedded plus a .tfw; {gcps}date from the set's cycle, "
    "end_date = next cycle start (cross-checked against the .htm "
    "Beginning_Date/Ending_Date{anom}). Staged "
    "rawtiffs/acasis_sectional_cycles/{cont}/ with FAA filenames verbatim; the "
    "row filename is that container, which holds no .zip — the source was a "
    "whole-cycle archive, not a per-chart one. The superseded file stays on "
    "disk as the as-found capture it is, listed in worklists/superseded_sources.csv."
)
GAIN = {
    "pdf": "Full distribution resolution in place of a 300-dpi rasterisation of "
           "the FAA print PDF.",
    "ifly": "Full distribution resolution in place of a lossy ~78 m/px iFly-card "
            "rebuild, which that row's own note said to retire if an FAA "
            "original surfaced.",
}


def sha256(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def htm_dates(path):
    txt = re.sub(r"<[^>]*>", " ", open(path, encoding="utf-8", errors="replace").read())
    d = {}
    for k, v in DATE_RE.findall(txt):
        d.setdefault(k, v)
    fmt = lambda s: "%s-%s-%s" % (s[:4], s[4:6], s[6:]) if s else None
    return fmt(d.get("Beginning")), fmt(d.get("Ending"))


def container(cycle, location):
    return "acasis_sec_%s-%s-%s_%s" % (cycle[5:7], cycle[8:10], cycle[:4],
                                       location.replace(" ", "_"))


def disk_sizes():
    """Every .tif/.zip size in rawtiffs -> paths, for the already-held check.

    Skips our own staging tree: once --stage has run, every chart in the plan
    matches a copy we just made, which would empty the plan on the --write pass.
    """
    idx = collections.defaultdict(list)
    for root, dirs, files in os.walk(RAW):
        if Path(root) == STAGE:
            dirs[:] = []
            continue
        for f in files:
            if f.endswith((".tif", ".zip")):
                p = os.path.join(root, f)
                try:
                    idx[os.path.getsize(p)].append(p)
                except OSError:
                    pass
    return idx


def build_plan(rows):
    """(upgrade rows, WAI combined rows to drop). Pure catalog+disk derivation."""
    by_key = {(r["location"], r["date"]): r for r in rows}
    sizes = disk_sizes()
    plan, drops = [], []
    for cycle, (src_dir, as_found) in SETS.items():
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".tif") or fn.startswith("._"):
                continue
            stem = fn[:-4]
            loc = stem[:-4] if stem.endswith(" SEC") else stem
            row = by_key.get((loc, cycle))
            if row is None or row["filename"].startswith("acasis_sec_"):
                continue                      # not catalogued, or already ours
            path = os.path.join(src_dir, fn)
            if sizes.get(os.path.getsize(path)):
                continue                      # FAA original already on disk
            plan.append({"cycle": cycle, "src_dir": src_dir, "as_found": as_found,
                         "stem": stem, "location": loc, "row": row,
                         "container": container(cycle, loc)})
        wai = by_key.get((WAI_COMBINED, cycle))
        if wai is not None:
            drops.append({"cycle": cycle, "row": wai})
    plan.sort(key=lambda p: (p["cycle"], p["location"]))
    return plan, drops


def stage(plan, dry):
    mpath = STAGE / "manifest.jsonl"
    seen = set()
    if mpath.exists():
        for line in open(mpath):
            try:
                seen.add(json.loads(line)["container"])
            except (ValueError, KeyError):
                pass
    mf = None if dry else open(mpath, "a")
    ok = skip = bad = 0
    for p in plan:
        cont, dest_dir = p["container"], STAGE / p["container"]
        if cont in seen and dest_dir.is_dir():
            skip += 1
            continue
        if dry:
            print("  would stage %-50s <- %s/%s.tif" % (cont, p["as_found"], p["stem"]))
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        files, failed = [], False
        for ext in (".tif", ".tfw", ".htm"):
            src = os.path.join(p["src_dir"], p["stem"] + ext)
            dst = dest_dir / (p["stem"] + ext)
            if not os.path.exists(src):
                print("MISSING SOURCE %s" % src)
                failed = True
                break
            digest = sha256(src)
            shutil.copy2(src, dst)
            if sha256(dst) != digest:
                dst.unlink()
                print("HASH MISMATCH after copy: %s" % dst)
                failed = True
                break
            st = os.stat(src)
            files.append({"name": p["stem"] + ext, "source_path": src,
                          "source_container": p["as_found"],
                          "source_mtime": datetime.datetime.fromtimestamp(
                              st.st_mtime, datetime.timezone.utc).isoformat(),
                          "state": "live", "bytes": st.st_size, "sha256": digest,
                          "verified": "copy sha256 match"})
        if failed:
            bad += 1
            continue
        b, e = htm_dates(os.path.join(p["src_dir"], p["stem"] + ".htm"))
        mf.write(json.dumps({
            "container": cont, "batch": "upgrade-2026-08-28",
            "chart": {"location": p["location"], "type": "SEC", "edition": None,
                      "date": p["cycle"], "end_date": NEXT_CYCLE[p["cycle"]],
                      "htm_beginning_date": b, "htm_ending_date": e,
                      "htm_anomaly": (p["cycle"], p["stem"]) in HTM_ANOMALY},
            "supersedes": p["row"]["filename"],
            "source_volume": "/Volumes/ACASIS (exFAT) root per-cycle FAA sectional sets",
            "files": files}) + "\n")
        mf.flush()
        ok += 1
        print("  staged %-50s (%6.1f MB)" % (cont, files[0]["bytes"] / 2 ** 20))
    if mf:
        mf.close()
    print("staged=%d skipped=%d failed=%d" % (ok, skip, bad))
    return bad == 0


def superseded_files(filename):
    """Every on-disk file the old row filename claimed (both .pdf and .tif twins,
    plus the tifs under an extracted zip-stem directory)."""
    stem, _, ext = filename.rpartition(".")
    names = {filename.lower()}
    if ext in ("pdf", "tif"):
        names |= {(stem + ".pdf").lower(), (stem + ".tif").lower()}
    hits = []
    for root, dirs, files in os.walk(RAW):
        if ext == "zip" and Path(root).name.lower() == stem.lower():
            hits += [os.path.join(root, f) for f in files if f.endswith(".tif")]
        for f in files:
            if f.lower() in names:
                hits.append(os.path.join(root, f))
    return sorted(set(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", action="store_true", help="copy the FAA originals in")
    ap.add_argument("--write", action="store_true", help="repoint catalog rows")
    a = ap.parse_args()

    rows = dole_v2.load_rows(CSV_PATH)
    plan, drops = build_plan(rows)
    per_cycle = collections.Counter(p["cycle"] for p in plan)
    kinds = collections.Counter(
        "ifly" if p["row"]["filename"].startswith("ifly_") else "pdf" for p in plan)
    print("%d rows to upgrade %s; %d combined Western Aleutian rows to drop %s"
          % (len(plan), dict(per_cycle), len(drops),
             [d["cycle"] for d in drops]))
    print("  current sources: %s;  rows carrying GCPs to clear: %d"
          % (dict(kinds), sum(1 for p in plan if dole_v2.row_gcp_pixels(p["row"]))))

    if a.stage:
        if not stage(plan, dry=False):
            return 1
    elif not a.write:
        stage(plan, dry=True)

    missing = [p["container"] for p in plan if not (STAGE / p["container"]).is_dir()]
    if a.write and missing:
        print("REFUSING: %d staged directories missing (run --stage first): %s"
              % (len(missing), missing[:3]))
        return 1
    if not a.write:
        print("\n(pass --stage to copy files, --write to repoint the catalog)")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = BACKUP_DIR / ("master_dole_v2.csv.pre_acasis_upgrade_%s.bak" % stamp)
    shutil.copy2(CSV_PATH, backup)

    ledger = []
    for p in plan:
        row, cycle, stem = p["row"], p["cycle"], p["stem"]
        old = row["filename"]
        anom = HTM_ANOMALY.get((cycle, stem))
        kind = "ifly" if old.startswith("ifly_") else "pdf"
        had_gcps = bool(dole_v2.row_gcp_pixels(row))
        for f in GCP_FIELDS:
            row[f] = ""
        row["src_crs"] = ""
        row["filename"] = p["container"] + ".zip"
        row["download_link"] = ""
        row["end_date"] = NEXT_CYCLE[cycle]
        row["note"] = NOTE.format(
            old=old, set=p["as_found"], cont=p["container"], gain=GAIN[kind],
            gcps=("the 16 GCP columns transferred from the 300-dpi conversion "
                  "grid were cleared, they do not apply to this raster; "
                  if had_gcps else ""),
            anom=(" — " + anom) if anom else "; they agree")
        for f in superseded_files(old):
            ledger.append((os.path.relpath(f, RAW), old, row["location"], cycle,
                           row["filename"], "superseded by FAA original"))

    drop_ids = set()
    for d in drops:
        r = d["row"]
        drop_ids.add(id(r))
        for f in superseded_files(r["filename"]):
            ledger.append((os.path.relpath(f, RAW), r["filename"], r["location"],
                           d["cycle"], "(row dropped)",
                           "combined W. Aleutian sheet, east-cut 300-dpi PDF "
                           "conversion, superseded by the FAA East + West halves"))
    kept = [r for r in rows if id(r) not in drop_ids]

    tmp = str(CSV_PATH) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in dole_v2.V2_FIELDS})
    os.replace(tmp, CSV_PATH)

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    fresh = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["path_in_rawtiffs", "old_row_filename", "location",
                        "date", "new_row_filename", "reason"])
        w.writerows(sorted(set(ledger)))

    print("\nrepointed %d rows, dropped %d; dole %d -> %d rows"
          % (len(plan), len(drops), len(rows), len(kept)))
    print("backup: %s" % backup)
    print("ledger: %s (+%d files)" % (LEDGER, len(set(ledger))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
