#!/usr/bin/env python
"""Two-way rawtiffs <-> dole audit (re-runnable version of worklist 02's method).

Forward: files on disk claimed by NO dole row, minus the explained buckets
(versos, non-sectional NARA products, wholecycle bundles, short-name twins of
claimed long s3 names, gap-dir intermediates). Anything left is written to
worklists/data/uncataloged_on_disk_<date>.csv for manual disposition.

Reverse: dole rows whose filename resolves to nothing on disk (these warp
nothing and vanish silently from mosaics).

Claim rules mirror slicer.py (_build_source_index + resolve_filename):
  - a non-zip row filename claims that basename anywhere in the tree
  - a zip row claims the zip file AND every .tif under any dir named <zip stem>
  - a .pdf row also claims the same-stem .tif (materialized), and vice versa

Run with ~/venv/bin/python from the repo root.
"""
import csv
import datetime
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dole_v2

REPO = Path(__file__).resolve().parent.parent
RAW = Path('/Volumes/projects/rawtiffs')

VERSO = re.compile(r'^ca\d+[a-z]*v\d*\.tif$', re.I)
NONSEC = re.compile(
    r'_TAC|_FLY|_HEL|_Planning|_Graphic|Caribbean|Grand_Canyon|GrandCanyon|_WAC'
    r'|_Inset_|G4332\.G7\.P6',  # sectional-sheet insets + LOC Grand Canyon VFR charts (identified 2026-08-20)
    re.I)

# Per-file dispositions from the 2026-08-20 re-audit (see worklists/02).
KNOWN_DISPOSITIONS = {
    'wasp_02-2007-02-423_02.tif': 'blank verso (noted in the _01 row)',
    'wasp_02-2007-02-427_02.tif': 'blank verso (noted in the _01 row)',
    'ca000444.tif': 'Boston 1955-05-26 verso scanned without the v suffix (recto = ca000444r)',
    'alaska-overview.jpg': '2004-set overview graphic, not a chart',
    'archive.org_download_micro_ia41152645_0081_micro_ia41152645_0081_micro_jp2.tif':
        'stray GPO-fiche camera-tile conversion (HUNT26 stitch lane, not a standalone chart)',
    'web.archive.org_web_20140813232548_aeronav.faa.gov_content_aeronav_sectional_files_pdfs_western_aleutian_islands_47_p.zip':
        'print-PDF variant of W. Aleutian ed 47, held as GeoTIFF zip (other capture)',
    'western aleutian islands east sec 47.pdf': 'member of the redundant 47_P print zip',
    'western aleutian islands west sec 47.pdf': 'member of the redundant 47_P print zip',
}

# Sources a row used to point at before it was repointed at a better copy (the
# 2026-08-28 ACASIS FAA-original upgrade). They stay on disk as the as-found
# captures they are, so they are explained, not unclaimed.
SUPERSEDED_LEDGER = REPO / 'worklists' / 'superseded_sources.csv'


def superseded_paths():
    if not SUPERSEDED_LEDGER.exists():
        return set()
    with open(SUPERSEDED_LEDGER, newline='', encoding='utf-8') as fh:
        return {r['path_in_rawtiffs'].lower() for r in csv.DictReader(fh)}


def main():
    if not RAW.exists():
        sys.exit(f"{RAW} not mounted")
    rows = dole_v2.load_rows(REPO / 'master_dole_v2.csv')

    claimed_names = set()
    claimed_zipstems = set()
    for r in rows:
        fn = (r['filename'] or '').strip().lower()
        if not fn:
            continue
        claimed_names.add(fn)
        stem, _, ext = fn.rpartition('.')
        if ext == 'zip':
            claimed_zipstems.add(stem)
        elif ext == 'pdf':
            claimed_names.add(stem + '.tif')
        elif ext == 'tif':
            claimed_names.add(stem + '.pdf')

    disk_names = set()
    disk_dirnames = set()
    relevant = []
    for root, dirs, files in os.walk(RAW):
        rootp = Path(root)
        for d in dirs:
            disk_dirnames.add(d.lower())
        for f in files:
            disk_names.add(f.lower())
            if f.lower().endswith(('.tif', '.zip', '.pdf', '.jpg')):
                relevant.append((rootp / f, f))

    # ---- forward: unclaimed disk files
    def claimed(path, name):
        low = name.lower()
        if low in claimed_names:
            return True
        for anc in path.parents:
            if anc == RAW:
                break
            if anc.name.lower() in claimed_zipstems:
                return True
        return False

    superseded = superseded_paths()

    def explained(path, name):
        rel = str(path.relative_to(RAW))
        top = rel.split('/')[0]
        low = name.lower()
        if rel.lower() in superseded:
            return 'superseded_by_faa_original'
        if VERSO.match(name):
            return 'verso_policy'
        if NONSEC.search(name) or top == 'TAC':
            return 'non_sectional'
        if 'All_Files_Sectional' in name or 'wholecycle' in rel.lower():
            return 'wholecycle_bundle'
        if top == 'failed_extractions':
            return 'failed_extractions'
        if top == 'dole_gap_2026-07' and ('usahas' in rel or 'simviation' in rel.lower()):
            return 'gap_dir_intermediates'
        if any(c.endswith('_' + low) for c in claimed_names):
            return 'shortname_twin_of_claimed'
        if low in KNOWN_DISPOSITIONS:
            return 'known_disposition'
        return None

    unexplained = []
    bucket_counts = {}
    for p, n in relevant:
        if claimed(p, n):
            continue
        b = explained(p, n) or 'UNEXPLAINED'
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
        if b == 'UNEXPLAINED':
            unexplained.append((p, n))

    # ---- reverse: rows resolving to nothing
    ghost_rows = []
    for r in rows:
        fn = (r['filename'] or '').strip().lower()
        if not fn:
            ghost_rows.append((r, 'EMPTY-FILENAME'))
            continue
        if fn.endswith('.zip'):
            if fn in disk_names or fn[:-4] in disk_dirnames:
                continue
        else:
            if fn in disk_names:
                continue
            stem = fn.rsplit('.', 1)[0]
            if stem + '.pdf' in disk_names or stem + '.tif' in disk_names:
                continue
        ghost_rows.append((r, 'NOT-ON-DISK'))

    # ---- report
    print(f"disk relevant files: {len(relevant)}")
    for b in sorted(bucket_counts):
        print(f"  {b}: {bucket_counts[b]}")
    print(f"rows not resolvable on disk: {len(ghost_rows)}")
    for r, why in ghost_rows:
        print(f"  [{why}] {r['location']} {r['date']} fn={r['filename'][:70]}")

    if unexplained:
        out = REPO / 'worklists' / 'data' / (
            'uncataloged_on_disk_%s.csv' % datetime.date.today().isoformat())
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['path', 'basename'])
            for pth, n in sorted(unexplained, key=lambda t: str(t[0])):
                w.writerow([str(pth), n])
        print(f"unexplained list -> {out}")


if __name__ == '__main__':
    main()
