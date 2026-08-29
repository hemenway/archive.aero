#!/usr/bin/env python3
"""Stage only-copy FAA sectionals off the ACASIS root per-cycle sectional sets.

The ACASIS enclosure's root holds five whole-cycle FAA sectional downloads kept
by the iFly build machine's operator -- `Sectional (4) nov 2025.zip` (also
extracted at `recovery/Sectional`), `Sectional (3) jan`, `Sectional (2) march`,
`Sectional (5) may` and `Sectional july 2`.  Each is the complete 57-chart FAA
`sectional-files` set for one 56-day cycle, tif + tfw + htm, FAA names verbatim.

Only charts that pass CLAUDE.md's admission test are staged here: the (location,
cycle) pair is absent from master_dole_v2.csv, the bytes are absent from rawtiffs
and rawtiffs_attic, and the chart is not retrievable from Wayback or the live FAA.
The may and july sets are wholly redundant -- the archive already downloaded those
cycles from aeronav -- and are not touched.

Layout: one directory per (cycle, chart), named `acasis_sec_<MM-DD-YYYY>_<Location>`
with the FAA filenames unchanged inside it.  The dole row's filename is that
directory name plus `.zip`, which is how slicer.py's resolve_filename and
audit_disk_vs_dole.py both address an extracted container.  A per-chart directory
is what lets the FAA names stay verbatim: `Honolulu Inset SEC.tif` is the same
basename in all five cycles, and slicer.py's source index is keyed on the bare
basename, last-writer-wins.

Run with ~/venv/bin/python from the repo root.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

STAGE = "/Volumes/projects/rawtiffs/acasis_sectional_cycles"
DATE_RE = re.compile(r"(Publication|Beginning|Ending)_Date:\s*(\d{8})")

# (source set dir, ACASIS container as found, FAA cycle date)
SETS = {
    "2025-11-27": ("/Volumes/ACASIS/recovery/Sectional",
                   "Sectional (4) nov 2025.zip (extracted at recovery/Sectional)"),
    "2026-01-22": ("/Volumes/ACASIS/Sectional (3) jan", "Sectional (3) jan"),
    "2026-03-19": ("/Volumes/ACASIS/Sectional (2) march", "Sectional (2) march"),
}

INSET_SET = ["Honolulu Inset SEC", "Mariana Islands Inset SEC",
             "Samoan Islands Inset SEC",
             "Western Aleutian Islands East SEC",
             "Western Aleutian Islands West SEC"]

# (cycle, FAA file stem) -> staged.  Insets and the Western Aleutian halves are
# distributed only inside Hawaiian_Islands.zip / Western_Aleutian_Islands.zip,
# which Wayback never captured for these three cycles; the three named sheets are
# cycle gaps whose only Wayback captures are 5 MiB truncations.
ITEMS = (
    [("2025-11-27", s) for s in INSET_SET] +
    [("2026-01-22", s) for s in ["Cincinnati SEC", "Los Angeles SEC"] + INSET_SET] +
    [("2026-03-19", s) for s in ["Las Vegas SEC"] + INSET_SET]
)

# FAA file stem -> (dole location, cutline slug)
CHART = {
    "Cincinnati SEC": ("Cincinnati", "cincinnati"),
    "Los Angeles SEC": ("Los Angeles", "los_angeles"),
    "Las Vegas SEC": ("Las Vegas", "las_vegas"),
    "Honolulu Inset SEC": ("Honolulu Inset", "honolulu_inset"),
    "Mariana Islands Inset SEC": ("Mariana Islands Inset", "mariana_islands_inset"),
    "Samoan Islands Inset SEC": ("Samoan Islands Inset", "samoan_islands_inset"),
    "Western Aleutian Islands East SEC": ("Western Aleutian Islands East",
                                          "western_aleutian_islands_east"),
    "Western Aleutian Islands West SEC": ("Western Aleutian Islands West",
                                          "western_aleutian_islands_west"),
}


def sha256(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def htm_dates(path):
    """FAA Beginning_Date / Ending_Date, ISO. Returns (begin, ending)."""
    txt = re.sub(r"<[^>]*>", " ", open(path, encoding="utf-8", errors="replace").read())
    d = {}
    for k, v in DATE_RE.findall(txt):
        d.setdefault(k, v)
    fmt = lambda s: "%s-%s-%s" % (s[:4], s[4:6], s[6:]) if s else None
    return fmt(d.get("Beginning") or d.get("Publication")), fmt(d.get("Ending"))


def container(cycle, location):
    mm, dd, yyyy = cycle[5:7], cycle[8:10], cycle[:4]
    return "acasis_sec_%s-%s-%s_%s" % (mm, dd, yyyy, location.replace(" ", "_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=STAGE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    mpath = os.path.join(a.dest, "manifest.jsonl")
    seen = set()
    if os.path.exists(mpath):
        for line in open(mpath):
            try:
                seen.add(json.loads(line)["container"])
            except (ValueError, KeyError):
                pass

    if not a.dry_run:
        os.makedirs(a.dest, exist_ok=True)
    mf = None if a.dry_run else open(mpath, "a")
    ok = skip = bad = 0
    for cycle, stem in ITEMS:
        src_dir, as_found = SETS[cycle]
        location, _cut = CHART[stem]
        cont = container(cycle, location)
        dest_dir = os.path.join(a.dest, cont)

        if cont in seen and os.path.isdir(dest_dir):
            skip += 1
            continue

        begin, ending = htm_dates(os.path.join(src_dir, stem + ".htm"))
        if begin != cycle:
            # Lake Huron 2026-01-22 carries a bad FAA Beginning_Date; none of the
            # staged charts should, so refuse rather than mis-date a row.
            print("DATE MISMATCH %s %s: htm says %s" % (cycle, stem, begin))
            bad += 1
            continue

        if a.dry_run:
            print("would stage %-46s <- %s/%s.tif" % (cont, as_found, stem))
            continue

        os.makedirs(dest_dir, exist_ok=True)
        files, failed = [], False
        for ext in (".tif", ".tfw", ".htm"):
            src = os.path.join(src_dir, stem + ext)
            dst = os.path.join(dest_dir, stem + ext)
            if not os.path.exists(src):
                print("MISSING SOURCE %s" % src)
                failed = True
                break
            digest = sha256(src)
            shutil.copy2(src, dst)
            if sha256(dst) != digest:
                os.remove(dst)
                print("HASH MISMATCH after copy: %s" % dst)
                failed = True
                break
            st = os.stat(src)
            files.append({"name": stem + ext, "source_path": src,
                          "source_container": as_found,
                          "source_mtime": __import__("datetime").datetime
                              .fromtimestamp(st.st_mtime, __import__("datetime").timezone.utc)
                              .isoformat(),
                          "state": "live", "bytes": st.st_size,
                          "sha256": digest, "verified": "copy sha256 match"})
        if failed:
            bad += 1
            continue

        mf.write(json.dumps({
            "container": cont,
            "chart": {"location": location, "type": "SEC", "edition": None,
                      "date": begin, "faa_ending_date": ending,
                      "end_date": _plus_one(ending)},
            "source_volume": "/Volumes/ACASIS (exFAT) root per-cycle FAA sectional sets",
            "files": files}) + "\n")
        mf.flush()
        ok += 1
        print("staged %-46s (%6.1f MB)" % (cont, files[0]["bytes"] / 2 ** 20))

    if mf:
        mf.close()
    print("\nstaged=%d skipped=%d failed=%d" % (ok, skip, bad))
    return 1 if bad else 0


def _plus_one(iso):
    import datetime
    d = datetime.date.fromisoformat(iso) + datetime.timedelta(days=1)
    return d.isoformat()


if __name__ == "__main__":
    sys.exit(main())
