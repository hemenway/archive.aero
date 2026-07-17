# Worklist 02 — Disk ↔ dole mismatches (rawtiffs audit)

*Generated 2026-07-16 by walking `/Volumes/projects/rawtiffs/` with slicer.py's exact
indexing rules (`_build_source_index`: recursive, basenames `.tif/.zip/.pdf/.jpg`; zip rows
resolve via an extracted dir named `<zip stem>` at any depth; `.tif` rows fall back to a
same-stem local `.pdf`). Totals: 12,989 relevant files / 1,520 dirs vs 7,457 dole rows.
Row-level detail: [`data/dole_rows_missing_on_disk.csv`](data/dole_rows_missing_on_disk.csv)
and [`data/disk_files_not_in_dole_classified.csv`](data/disk_files_not_in_dole_classified.csv).*

Note: rawtiffs is the slicer's **only** source tree — there are no sibling indexes.
This supersedes the older `unlogged_in_rawtiffs.csv` (2025-12) and complements the CLOSED
`search_archive/missing_from_dole.csv` local audit (2026-07-14).

## A. Dole rows NOT resolvable on disk — 11 rows (99.85% resolve) ⚠ FIX FIRST

All 11 are the AVSIM **us_sectionals Pack 12** (Matt Fox) JPGs cataloged 2026-07-14 with
`DATE-APPROX 2004-07-01` and **no download_link** — the slicer cannot download them and
skips them ("No source files found"):

`US_SEC_{CAPE LISBURNE, DAWSON, FAIRBANKS, NOME, POINT BARROW} {EAST,WEST}.jpg` +
`US_SEC_HAWAIIAN ISLANDS.jpg`

**The files exist, they were never staged**: they're in
`~/Downloads/us_sectionals_pack_12/` (with JGW/prj sidecars).

- [ ] Copy the 11 JPGs into `/Volumes/projects/rawtiffs/` (bring the sidecars too).
- [ ] Note they're also on Worklist 04 (no GCPs) — georeference after staging.

Plus one hidden hazard that "resolves" but will fail at runtime:

- [ ] **Seward_93.zip** — claimed by a dole row (ed 93, 2013-11-14), but the only copy on
  disk is the truncated 1.2 MB capture in `failed_extractions/`; extraction will fail.
  Re-pull the largest wayback CDX capture of the same URL (open ACTION_PLAN item).

## B. Disk files with no dole row — 3,746 files, almost all explainable

| Bucket | Files | Size | Verdict |
|---|---|---|---|
| LOC `ca######v.tif` **verso** scans (root) | 1,223 | 607 GB | Deliberate — every verso's recto/base IS in dole; versos are uncataloged by policy. Not a gap. |
| NARA non-sectional chart types: `*_TAC` (714), `*_FLY` (440), `*_HEL` (204), `*_Planning_Chart` (24), `*_Graphic` (22) | 1,404 | — | Out of scope (dole is sectionals-only). Not a gap. |
| NARA **short-name duplicates** of claimed long s3 names (e.g. `Albuquerque_SEC_100.tif` = byte-identical twin of the claimed `s3.amazonaws.com_NARAprodstorage_..._Albuquerque_SEC_100.tif`) | 623 | 38 GB | Pre-rename originals. **Deletable for 38 GB** — filter `bucket=root_shortname_dupe_of_claimed_longname` in the classified CSV. |
| `TAC/` cycle dirs (wayback/FAA TAC PDFs+zips, 4 recent cycles) | 270 | — | Out of scope. |
| `dole_gap_2026-07/` residue: usahas KMZ tile intermediates (130), simviation Alaska staging zips whose member PDFs are separately cataloged (7), HowTo/overview images | 138 | — | Intermediates/duplicates of cataloged outputs. Not a gap. |
| Non-sectional NARA specials (Caribbean VFR, Grand Canyon air-tour, NY TAC planning, SLC HEL, Alaska wall planning) | 77 | — | Out of scope. |
| Truncated unclaimed TAC zips in `failed_extractions/` | 4 | ~0 | Junk — delete. |
| Already dispositioned by the closed 2026-07 audit (inset tifs, WASP versos, W. Aleutian halves, wholecycle bundle) | 10 | — | Done. |

### The only genuinely unexplained files (2)

- [ ] `G4332.G7.P6_1998_front.jpg` (52 MB) and `G4332.G7.P6_2001_front.jpg` (69 MB) —
  LOC call-number-style scans at rawtiffs root. In neither the dole, the closed audit, nor
  the additions report. Identify (call number G4332.G7 ≈ Gulf-coast region) and either
  catalog or discard. *(G4332.G7.P6 is a Guam/Pacific-style call number pattern — open
  them and read the margin before deciding.)*

### Optional cleanups

- [ ] Delete the 623 short-name NARA duplicates → reclaim ~38 GB.
- [ ] Delete the 4 truncated TAC zips in `failed_extractions/`.
- [ ] If TACs/FLYs/HELs will never be in scope, consider moving the 1,674 NARA
  non-sectional files + `TAC/` dirs off rawtiffs so source indexing stays fast.

## Zip-handling notes (for future audits)

Dole's 1,215 `.zip` rows resolve through extracted dirs named after the zip stem — member
basenames never need their own rows. Wholecycle `*_All_Files_Sectional.zip` bundles are
intentionally rowless. 0 rows currently depend on the PDF-rasterization fallback.
