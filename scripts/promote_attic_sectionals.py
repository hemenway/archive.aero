#!/usr/bin/env python3
"""Clear the remaining sectionals out of rawtiffs_attic and catalogue them.

Three distinct cases fell out of auditing the imgcv2 carve against the catalog:

  A  9 charts that exist in no other copy -- 8 mainland 2014-15 sheets plus
     Dutch Harbor ed 50, which closes a 364-day hole in the Aleutians. Moved out
     of the attic into rawtiffs/acasis_imgcv2_sec/ and catalogued.
  B  27 Hawaiian/Pacific inset editions, 2016-2020, whose FAA GeoTIFFs are
     ALREADY in rawtiffs as standalone NARA files -- they simply never got rows.
     Catalog-only; nothing moves.
  C  the 2015-04-30 inset trio (ed 92), which sits inside a Wayback
     Hawaiian_Islands_92 container. Split into per-chart containers the same way
     scripts/split_faa_inset_containers.py handles the modern cycles.

Inserting an edition STALES ITS PREDECESSOR'S end_date: the catalog chains each
row's end_date to the next edition known at the time it was written. Every
affected location is re-chained here and the changes are printed, because getting
this wrong shifts the era KEY and the new mosaic would never merge.

Run with ~/venv/bin/python from the repo root; --write to apply.
"""
import argparse
import collections
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
ATTIC = Path("/Volumes/projects/rawtiffs_attic/acasis_imgcv2_FAA_originals")
DEST_A = RAW / "acasis_imgcv2_sec"
SPLITS = RAW / "faa_chart_splits"
BACKUP_DIR = Path.home() / "archive.aero-attic" / "csv-backups"
# Plan inputs (the audit lists that drove the 2026-08-29 run). They used to
# be read from a session scratchpad that no longer exists; they live under
# the gitignored worklists/data/ now and are overridable on the command line.
PLAN_DIR = REPO / "worklists" / "data" / "attic_promote"

INSET_CUT = {"Honolulu Inset": "sectional/honolulu_inset",
             "Mariana Islands Inset": "sectional/mariana_islands_inset",
             "Samoan Islands Inset": "sectional/samoan_islands_inset"}

NOTE_A = (
    "recovered 2026-08-29 from rawtiffs_attic/acasis_imgcv2_FAA_originals (the "
    "868-chart carve of the iFly build machine's .imgcv2 disk image, worklist 11) "
    "and promoted into rawtiffs: FAA original distribution GeoTIFF, LZW palette, "
    "LCC/NAD83, .tfw + FAA .htm alongside; date/end_date from the .htm "
    "Beginning_Date/Ending_Date (end_date = Ending_Date + 1 day). Embedded georef, "
    "no GCP work. Only copy ever seen: absent from the catalog on (location, date), "
    "absent from rawtiffs by exact byte size, and absent from Wayback's "
    "sectional_files space (383 retrievable (location, edition) zips checked, none "
    "is this one). Carve integrity verified block-by-block: every strip decodes, "
    "no read errors, image content runs to the sheet margin."
)
NOTE_B = (
    "catalogued 2026-08-29 from a file already held in rawtiffs. The FAA ships this "
    "inset inside the cycle's Hawaiian Islands package, and the catalog carried no "
    "row for it between 2016 and 2020 even though the standalone NARA GeoTIFF was "
    "on disk the whole time — so the sheet was never warped and the era had no "
    "coverage here. Nothing moved; this row just addresses the existing file with "
    "the inset's own cutline."
)
NOTE_C = (
    "split out 2026-08-29 from the Wayback container {parent} so it can carry its "
    "own cutline: a .zip row resolves to every tif in its container and applied "
    "sectional/hawaiian_islands to all four sheets, so this inset was clipped away "
    "(Mariana and Samoa entirely). Same bytes, moved into "
    "rawtiffs/faa_chart_splits/{cont}/ with the FAA filename verbatim; the .zip "
    "beside the original container remains the as-found artifact."
)


def container(cycle, location):
    return "faasplit_%s-%s-%s_%s" % (cycle[5:7], cycle[8:10], cycle[:4],
                                     location.replace(" ", "_"))


def plus_one(iso):
    return (datetime.date.fromisoformat(iso) + datetime.timedelta(days=1)).isoformat()


def tiff_ok(path):
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
    except OSError:
        return False


def write_salvage_metadata(dest, readme_text, entries):
    """README.txt (if absent) + manifest.jsonl lines for a salvage directory.

    Every salvage directory carries both (CLAUDE.md): what source, which tool,
    what date, and one manifest line per file with its original path, its
    identifier in the source, live/deleted, byte size and verification.
    """
    dest.mkdir(parents=True, exist_ok=True)
    readme = dest / "README.txt"
    if not readme.exists():
        readme.write_text(readme_text)
    if entries:
        with open(dest / "manifest.jsonl", "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")


def attic_record(source_filename):
    """The attic manifest line for a file, keyed by its source filename."""
    path = ATTIC / "manifest.jsonl"
    if not path.exists():
        return None
    for line in open(path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("source_filename") == source_filename and rec.get("kind") != "sidecar":
            return rec
    return None


def attic_note_moved(entries):
    """Append `moved_to` records to the attic manifest so it stays truthful
    about files that were promoted out (it kept listing them under `out`)."""
    path = ATTIC / "manifest.jsonl"
    if not path.exists() or not entries:
        return
    with open(path, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


README_DEST_A = """acasis_imgcv2_sec
=================

FAA original distribution GeoTIFFs promoted out of
rawtiffs_attic/acasis_imgcv2_FAA_originals (the 868-chart carve of the iFly
build machine's .imgcv2 disk image, worklist 11) on 2026-08-29 by
scripts/promote_attic_sectionals.py: the editions of which no other copy is
known (absent from the catalog on (location, date), from rawtiffs by byte
size, and from Wayback's sectional_files space). Each .tif keeps its FAA
filename verbatim with its .tfw and FAA .htm alongside. Device, image, tool
and carve conditions: see the attic's README.txt; per-file provenance
(source path on the disk image, MFT record, live/deleted, verification):
manifest.jsonl here, which repeats the attic manifest line for each file.
"""

README_SPLIT = """{cont}
{underline}

Split out {when} from parent container {parent} (catalog row
{parent_loc} {cycle}) by scripts/promote_attic_sectionals.py so the sheet
can carry its own catalog row and cutline. Files moved with names verbatim;
see manifest.jsonl. Source of the bytes: the parent row's note.
"""


def reconcile(rows):
    """Write the README/manifest a previous run of this script did not:
    rawtiffs/acasis_imgcv2_sec/ and the faasplit containers it created, plus
    `moved_to` records in the attic manifest. Idempotent: files already
    listed in a manifest are skipped."""
    def listed(dest):
        m = dest / "manifest.jsonl"
        if not m.exists():
            return set()
        out = set()
        for line in open(m, encoding="utf-8"):
            try:
                out.add(json.loads(line).get("file"))
            except ValueError:
                pass
        return out

    written = 0
    if DEST_A.exists():
        have = listed(DEST_A)
        entries, moved = [], []
        for tif in sorted(DEST_A.glob("*.tif")):
            if tif.name in have:
                continue
            src = attic_record(tif.name) or {}
            for f in [tif] + [tif.with_suffix(ext) for ext in (".tfw", ".htm") if tif.with_suffix(ext).exists()]:
                entries.append({
                    "file": f.name,
                    "original_path": src.get("source_path", "(unknown; see attic manifest)"),
                    "source_identifier": {"mft_record": src.get("mft_record"), "attic_out": src.get("out")},
                    "state_on_source": src.get("state_on_disk", "unknown"),
                    "bytes": f.stat().st_size,
                    "verification": ("tiff magic ok" if f.suffix == ".tif" and tiff_ok(f)
                                     else "sidecar" if f.suffix != ".tif" else "TIFF MAGIC FAILED"),
                    "promoted": "2026-08-29",
                    "reconciled": datetime.date.today().isoformat(),
                })
            if src:
                moved.append({"out": src.get("out"), "source_filename": tif.name,
                              "moved_to": str(tif.relative_to(RAW)),
                              "moved_on": "2026-08-29", "recorded": datetime.date.today().isoformat()})
        write_salvage_metadata(DEST_A, README_DEST_A, entries)
        attic_note_moved(moved)
        written += len(entries)
    if SPLITS.exists():
        by_file = {r["filename"]: r for r in rows}
        for cont in sorted(p for p in SPLITS.iterdir() if p.is_dir()):
            have = listed(cont)
            row = by_file.get(cont.name + ".zip")
            entries = []
            for f in sorted(cont.iterdir()):
                if f.name in ("README.txt", "manifest.jsonl") or f.name in have:
                    continue
                entries.append({
                    "file": f.name,
                    "original_path": "(parent container; see README)",
                    "parent_row": {"location": row["location"], "date": row["date"]} if row else None,
                    "bytes": f.stat().st_size,
                    "verification": ("tiff magic ok" if f.suffix == ".tif" and tiff_ok(f)
                                     else "sidecar" if f.suffix != ".tif" else "TIFF MAGIC FAILED"),
                    "reconciled": datetime.date.today().isoformat(),
                })
            parent = "(see catalog note)"
            if row and "from the " in row.get("note", ""):
                parent = row["note"].split("container ")[1].split(" ")[0] if "container " in row["note"] else parent
            parent_loc = ("Western Aleutian Islands" if "Aleutian" in cont.name else "Hawaiian Islands")
            write_salvage_metadata(cont, README_SPLIT.format(
                cont=cont.name, underline="=" * len(cont.name), when="2026-08-29",
                parent=parent, parent_loc=parent_loc,
                cycle=(row["date"] if row else "?")), entries)
            written += len(entries)
    print("reconcile: %d manifest line(s) written" % written)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--gap-json", type=Path, default=PLAN_DIR / "attic_truegap.json",
                    help="case-A plan: charts in the attic with no other copy")
    ap.add_argument("--ondisk-json", type=Path, default=PLAN_DIR / "inset_rows_ondisk.json",
                    help="case-B plan: inset files already in rawtiffs that lack rows")
    ap.add_argument("--reconcile", action="store_true",
                    help="write the README/manifest metadata a previous run left out and stop")
    a = ap.parse_args()

    rows = dole_v2.load_rows(CSV_PATH)
    if a.reconcile:
        return reconcile(rows)
    for pth in (a.gap_json, a.ondisk_json):
        if not pth.exists():
            raise SystemExit("plan input missing: %s (the 2026-08-29 inputs were session scratch and "
                             "are gone; regenerate them into %s or pass --gap-json/--ondisk-json)"
                             % (pth, PLAN_DIR))
    by_key = {(r["location"], r["date"]): r for r in rows}
    cut_for = {}
    for r in rows:
        if r["cutline"]:
            cut_for.setdefault(r["location"], r["cutline"])

    gapA = json.load(open(a.gap_json))
    ondisk = json.load(open(a.ondisk_json))
    new_rows, moves, problems = [], [], []

    # --- A: promote out of the attic -------------------------------------
    for r in sorted(gapA, key=lambda x: (x["date"], x["location"])):
        loc, d = r["location"], r["date"]
        if (loc, d) in by_key:
            problems.append("A %s %s already catalogued" % (loc, d))
            continue
        cut = cut_for.get(loc)
        if not cut:
            problems.append("A %s: no cutline known" % loc)
            continue
        src = ATTIC / r["dir"]
        moves.append((src, r["file"].rsplit(".", 1)[0]))
        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({"filename": r["file"], "download_link": "", "location": loc,
                    "date": d, "end_date": plus_one(r["end_date"]),
                    "edition": str(r["edition"]).strip() or "Unknown",
                    "cutline": cut, "note": NOTE_A})
        new_rows.append(row)

    # --- B: rows for files already in rawtiffs ----------------------------
    for r in sorted(ondisk, key=lambda x: (x["date"], x["location"])):
        loc, d = r["location"], r["date"]
        if (loc, d) in by_key:
            problems.append("B %s %s already catalogued" % (loc, d))
            continue
        row = {k: "" for k in dole_v2.V2_FIELDS}
        row.update({"filename": r["filename"], "download_link": "", "location": loc,
                    "date": d, "end_date": plus_one(r["end_date"]),
                    "edition": str(r["edition"]).strip(),
                    "cutline": INSET_CUT[loc], "note": NOTE_B})
        new_rows.append(row)

    # --- C: split the 2015-04-30 Wayback container ------------------------
    cyc = "2015-04-30"
    parent = by_key.get(("Hawaiian Islands", cyc))
    splits = []
    if parent is None:
        problems.append("C: no Hawaiian Islands row for %s" % cyc)
    else:
        stem = parent["filename"][:-4] if parent["filename"].endswith(".zip") else None
        src = None
        for root, _ds, _fs in os.walk(RAW):
            if os.path.basename(root) == stem:
                src = Path(root)
                break
        if src is None:
            problems.append("C: container %s not on disk" % stem)
        else:
            for loc in INSET_CUT:
                mem = loc + " SEC 92"
                if not (src / (mem + ".tif")).exists():
                    problems.append("C: %s missing in %s" % (mem, stem))
                    continue
                if (loc, cyc) in by_key:
                    problems.append("C %s %s already catalogued" % (loc, cyc))
                    continue
                cont = container(cyc, loc)
                splits.append((src, mem, cont))
                row = {k: "" for k in dole_v2.V2_FIELDS}
                row.update({"filename": cont + ".zip", "download_link": "",
                            "location": loc, "date": cyc,
                            "end_date": parent["end_date"], "edition": "92",
                            "cutline": INSET_CUT[loc],
                            "note": NOTE_C.format(parent=parent["filename"], cont=cont)})
                new_rows.append(row)

    print("A promote from attic: %d   B catalog-only: %d   C container split: %d"
          % (len(moves), len(ondisk), len(splits)))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1

    # --- end_date re-chaining ---------------------------------------------
    # ONLY where a new edition lands strictly inside an existing row's validity
    # interval: that predecessor was chained to the next edition known when it
    # was written, and the insert supersedes it. Rows whose interval ends before
    # the new date are left alone -- a long end_date across a genuine gap in the
    # archive is correct, not stale.
    rechained = []
    new_dates_src = new_rows
    all_dates = collections.defaultdict(set)
    for r in rows + new_rows:
        if r["date"]:
            all_dates[r["location"]].add(r["date"])

    new_dates = collections.defaultdict(set)
    for r in new_dates_src:
        new_dates[r["location"]].add(r["date"])

    def shorten(r, tag, dates):
        """An edition ends when the next one starts. Only ever SHORTENS: a long
        end_date across a genuine gap in the archive is correct, not stale.
        Existing rows are measured against the INSERTED dates only -- pre-existing
        overlaps between two existing editions are a separate problem, and fixing
        them here would change already-published era keys."""
        if not r["date"] or not r["end_date"]:
            return
        inside = [d for d in dates[r["location"]] if r["date"] < d < r["end_date"]]
        if inside:
            nxt = min(inside)
            rechained.append((r["location"], r["date"], r["end_date"], nxt, tag))
            r["end_date"] = nxt

    for nr in new_rows:                      # new editions superseding each other
        shorten(nr, "new", all_dates)
    for r in rows:                           # predecessors the inserts land inside
        shorten(r, "existing", new_dates)
    print("\nend_date shortened on %d rows:" % len(rechained))
    for loc, dd, old_e, new_e, tag in sorted(rechained):
        print("   %-26s %s  %s -> %s  [%s]" % (loc, dd, old_e, new_e, tag))
    blank = [(r["location"], r["date"]) for r in new_rows if not r["end_date"]]
    if blank:
        problems.append("new rows with no end_date: %s" % blank[:5])
    if problems:
        print("\nPROBLEMS:")
        for p_ in problems:
            print("  " + p_)
        return 1
    if not a.write:
        print("\n(pass --write to apply)")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = BACKUP_DIR / ("master_dole_v2.csv.pre_attic_promote_%s.bak" % stamp)
    shutil.copy2(CSV_PATH, backup)

    DEST_A.mkdir(parents=True, exist_ok=True)
    nmoved = 0
    today = datetime.date.today().isoformat()
    a_entries, a_moved = [], []
    for src, stem in moves:
        rec = attic_record(stem + ".tif") or {}
        for ext in (".tif", ".tfw", ".htm"):
            s = src / (stem + ext)
            if s.exists():
                dst = DEST_A / (stem + ext)
                shutil.move(str(s), str(dst))
                nmoved += 1
                a_entries.append({"file": dst.name, "original_path": rec.get("source_path", str(s)),
                                  "source_identifier": {"mft_record": rec.get("mft_record"), "attic_out": rec.get("out")},
                                  "state_on_source": rec.get("state_on_disk", "unknown"),
                                  "bytes": dst.stat().st_size,
                                  "verification": ("tiff magic ok" if ext == ".tif" and tiff_ok(dst)
                                                   else "sidecar" if ext != ".tif" else "TIFF MAGIC FAILED"),
                                  "promoted": today})
        if rec:
            a_moved.append({"out": rec.get("out"), "source_filename": stem + ".tif",
                            "moved_to": str((DEST_A / (stem + ".tif")).relative_to(RAW)), "moved_on": today})
    write_salvage_metadata(DEST_A, README_DEST_A, a_entries)
    attic_note_moved(a_moved)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for src, mem, cont in splits:
        d = SPLITS / cont
        d.mkdir(parents=True, exist_ok=True)
        entries = []
        for ext in (".tif", ".tfw", ".htm"):
            s = src / (mem + ext)
            if s.exists():
                dst = d / (mem + ext)
                shutil.move(str(s), str(dst))
                nmoved += 1
                entries.append({"file": dst.name, "original_path": str(s.relative_to(RAW)),
                                "parent_row": {"location": "Hawaiian Islands", "date": cyc},
                                "bytes": dst.stat().st_size,
                                "verification": ("tiff magic ok" if ext == ".tif" and tiff_ok(dst)
                                                 else "sidecar" if ext != ".tif" else "TIFF MAGIC FAILED"),
                                "moved": today})
        write_salvage_metadata(d, README_SPLIT.format(
            cont=cont, underline="=" * len(cont), when=today, parent=parent["filename"],
            parent_loc="Hawaiian Islands", cycle=cyc), entries)

    tmp = str(CSV_PATH) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dole_v2.V2_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in dole_v2.V2_FIELDS})
        w.writerows(new_rows)
    os.replace(tmp, CSV_PATH)
    print("\nmoved %d files; +%d rows; dole %d -> %d\nbackup: %s"
          % (nmoved, len(new_rows), len(rows), len(rows) + len(new_rows), backup))
    return 0


if __name__ == "__main__":
    sys.exit(main())
