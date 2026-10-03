# Worklist 02 — Disk ↔ dole mismatches (rawtiffs audit)

## Current status — 2026-09-21 (donation update 2026-10-01): source reconciliation open

A read-only walk of the mounted `/Volumes/projects/rawtiffs` against **7,672**
rows loaded through `scripts/dole_v2.py`, using the claim/resolution rules in
`scripts/audit_disk_vs_dole.py`, found **12,061 relevant files**, **142 catalog
rows not resolvable**, and **48 unclaimed/unexplained TIFFs**. The walk reported
no filesystem errors. These are resolver findings, not proof that sources were
lost or that published tiles are missing; no downloads, extraction, moves,
content validation or live-site checks were performed.

| Current resolution misses | Rows |
|---|---:|
| archive.org 2004 JPGs | 78 |
| usahas ChartGeek mosaics | 23 |
| Juneau NARA Batch19 JPGs | 17 |
| AVSIM Pack 12 JPGs | 11 |
| simviation chart TIFFs | 3 |
| aviationtoolbox GeoTIFFs | 3 |
| NOAA Anchorage sources | 2 |
| San Francisco 1966/1978 chart TIFFs | 2 |
| FAA sample, ESRI sample, Seward 93 ZIP | 3 |

The `dole_gap_2026-07/archive2004/` directory is present but empty. All 11 Pack-12
rows have blank download links. The 48 unexplained files are the 24 TIFFs in each
of `Ross donation/Side a/` and `Side b/`; their uniqueness and catalog eligibility
have not been established by this filename audit.

- [ ] Reconcile the 142 misses with the current volume, prior inventories and
  retained sources before rebuilding. Restore/repoint only after verifying the
  actual file and provenance; do not infer a deletion from this audit.
- [ ] Identify and disposition the 48 Ross scans by edition/side against current
  holdings before adding rows.
- [ ] Refresh any cleanup plan from current files and full hashes. The current
  classifier finds **590 short-name candidates**, not the historical 623; suffix
  matching is not byte-identity proof. The `failed_extractions/` directory is now
  empty, so its old four-TAC-ZIP deletion task is retired.

Other unclaimed-but-explained buckets today: 2,406 non-sectional, 1,223 versos,
214 ledgered superseded sources, 4 known dispositions and 1 whole-cycle bundle.
The [September 19 duplicate-extraction](dup_extractions_cleanup_2026-09-19.md)
and [deep-archive cleanup](deep_archive_cleanup_2026-09-19.md) reports cover their
own fixed inventories; they do not establish today's overall rawtiffs parity.

## Historical audit — July/August 2026

*Generated 2026-07-16 by walking `/Volumes/projects/rawtiffs/` with slicer.py's exact
indexing rules (`_build_source_index`: recursive, basenames `.tif/.zip/.pdf/.jpg`; zip rows
resolve via an extracted dir named `<zip stem>` at any depth; `.tif` rows fall back to a
same-stem local `.pdf`). Totals: 12,989 relevant files / 1,520 dirs vs 7,457 dole rows.
Row-level detail: [`data/dole_rows_missing_on_disk.csv`](data/dole_rows_missing_on_disk.csv)
and [`data/disk_files_not_in_dole_classified.csv`](data/disk_files_not_in_dole_classified.csv).*

Note: rawtiffs is the slicer's **only** source tree — there are no sibling indexes.
This supersedes the older `unlogged_in_rawtiffs.csv` (2025-12) and complements the CLOSED
`search_archive/missing_from_dole.csv` local audit (2026-07-14).

> **2026-08-20 historical result — both directions were CLEAN at that audit.**
> Re-runnable tool: `scripts/audit_disk_vs_dole.py` (same claim rules as this doc).
> Reverse direction: **0 rows unresolvable** (the 11 Pack-12 JPGs were staged, §A done).
> Forward direction: 13 unclaimed-and-unexplained files, every one dispositioned —
> row-level record in `data/uncataloged_on_disk_2026-08-20.csv`. Highlights: the two
> G4332 mysteries are **Grand Canyon VFR Aeronautical Charts** (2nd ed. 1998-09-10,
> 3rd ed. 2001-04-19) → out of scope like the NARA GC air-tour specials; `ca000444.tif`
> is the Boston 1955-05-26 **verso scanned without the v suffix**; WASP 423_02/427_02
> are blank versos already noted in their _01 rows; the W. Aleutian 47_P print zip
> duplicates the held ed-47 GeoTIFF zip; 3 NARA rg-237 `*_Inset_SEC_95` files stay out
> under the no-insets policy (the Mariana/Samoan pair is the only Guam/Samoa content on
> disk if that scope ever opens). Worklist 07's "31 on disk but uncataloged" is thereby
> stale — everything real got rows during the July–August cataloging sessions.

### July missing-source baseline — 11 Pack-12 rows

All 11 are the AVSIM **us_sectionals Pack 12** (Matt Fox) JPGs cataloged 2026-07-14 with
`DATE-APPROX 2004-07-01` and **no download_link** — the slicer cannot download them and
skips them ("No source files found"):

`US_SEC_{CAPE LISBURNE, DAWSON, FAIRBANKS, NOME, POINT BARROW} {EAST,WEST}.jpg` +
`US_SEC_HAWAIIAN ISLANDS.jpg`

The July audit located the unstaged files in
`~/Downloads/us_sectionals_pack_12/` (with JGW/prj sidecars); that location was not
rechecked in this update.

- [x] Copy the 11 JPGs into `/Volumes/projects/rawtiffs/` (bring the sidecars too) —
  done (verified resolvable by the 2026-08-20 re-audit).
- The current rows carry `src_crs=EPSG:4269` for their JGW geotransforms;
  blank GCPs alone do not establish a georef backlog. Verify files/world files
  and a rendered warp after source availability is reconciled.

The July audit also identified a file that resolved by name but was corrupt:

- [ ] **Seward_93.zip** — claimed by a dole row (ed 93, 2013-11-14), the copy then on
  disk was a truncated 1.2 MB capture in `failed_extractions/`.
  All known captures failed in July; find an alternative source (01), rather
  than re-pulling the same capture. It is absent from the current source tree.

### July unclaimed-file baseline — 3,746 files

| Bucket | Files | Size | Verdict |
|---|---|---|---|
| LOC `ca######v.tif` **verso** scans (root) | 1,223 | 607 GB | Deliberate — every verso's recto/base IS in dole; versos are uncataloged by policy. Not a gap. |
| NARA non-sectional chart types: `*_TAC` (714), `*_FLY` (440), `*_HEL` (204), `*_Planning_Chart` (24), `*_Graphic` (22) | 1,404 | — | Out of scope (dole is sectionals-only). Not a gap. |
| NARA **short-name duplicates** of claimed long s3 names (e.g. `Albuquerque_SEC_100.tif` = byte-identical twin of the claimed `s3.amazonaws.com_NARAprodstorage_..._Albuquerque_SEC_100.tif`) | 623 | 38 GB | Pre-rename originals. Historical byte-duplicate estimate: 38 GB — filter `bucket=root_shortname_dupe_of_claimed_longname` in the classified CSV. |
| `TAC/` cycle dirs (wayback/FAA TAC PDFs+zips, 4 recent cycles) | 270 | — | Out of scope. |
| `dole_gap_2026-07/` residue: usahas KMZ tile intermediates (130), simviation Alaska staging zips whose member PDFs are separately cataloged (7), HowTo/overview images | 138 | — | Intermediates/duplicates of cataloged outputs. Not a gap. |
| Non-sectional NARA specials (Caribbean VFR, Grand Canyon air-tour, NY TAC planning, SLC HEL, Alaska wall planning) | 77 | — | Out of scope. |
| Truncated unclaimed TAC zips in `failed_extractions/` | 4 | ~0 | Historical failed captures; current directory is empty. |
| Already dispositioned by the closed 2026-07 audit (inset tifs, WASP versos, W. Aleutian halves, wholecycle bundle) | 10 | — | Done. |

### The two unexplained July files — identified in August

- [x] `G4332.G7.P6_1998_front.jpg` (52 MB) and `G4332.G7.P6_2001_front.jpg` (69 MB) —
  **IDENTIFIED 2026-08-20 by reading the title blocks: Grand Canyon VFR Aeronautical
  Chart (General Aviation)**, 2nd edition 1998-09-10 and 3rd edition 2001-04-19
  (1:250,000, SFRA corridor/sector chart). Non-sectional product → out of scope, same
  disposition as the NARA Grand Canyon air-tour specials. Kept on disk, not cataloged.

## Zip-handling notes (for future audits)

In the July snapshot, 1,215 `.zip` rows resolved through extracted dirs named after the zip stem — member
basenames never need their own rows. Wholecycle `*_All_Files_Sectional.zip` bundles are
intentionally rowless. The July audit found no rows depending on the PDF-rasterization fallback; that
is not a current content-validation result.

## Roslewski / O'Barr donation — 2026-10-01

All 23 sectional faces (12 sheets) were catalogued, hand georeferenced, sliced
and published in the [October 1 batch](georef_publish_2026-10-01.md). Chesapeake
Bay 1952 is a WAC and remains preserved outside the sectional catalog. This is
a separate donation from the 48 Ross scans in the September audit; those
disposition items remain open.
