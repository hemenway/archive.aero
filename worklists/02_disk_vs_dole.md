# Worklist 02 — Disk ↔ dole mismatches (rawtiffs audit)

## Current status — 2026-10-04: both directions clean

`scripts/audit_disk_vs_dole.py` against the mounted `/Volumes/projects/rawtiffs`
and **7,744** rows: **14,339 relevant files, 0 catalog rows unresolvable,
0 unexplained files.** Filename resolution only; no content validation.

- [x] The 142 misses of the September 21 audit all resolve today with no file
  moved or repointed in between (`dole_gap_2026-07/archive2004/` holds 79 JPGs,
  the Pack-12, usahas, Juneau Batch19, simviation and aviationtoolbox sources
  are in place). That walk saw an incomplete tree; it was not a loss.
- [x] The 48 `Ross donation/Side a|b` TIFFs are the letter-size flatbed tiles
  (24 per side, 600 dpi) of the O'Barr Dallas-Ft Worth 1969-07-24 sheet, stitched
  under `/Volumes/projects/ross_stitch/`. The sheet is catalogued from the
  1200 dpi M40 scans `Dallas-Ft_Worth_SEC_2_{North,South}.tif` (published
  October 1), so the tiles get no rows. They stay as found.
- [x] The 179 files under `acasis_website_2026-09-19/` are the documented
  September 19 import (README + manifest in the directory): TAC/FLY/HEL,
  enroute and planning sheets, plus `Anchorage SEC 103`, `Fairbanks SEC 103`
  and `Houston SEC 103`, kept as pixel variants of editions already catalogued
  from NARA. The audit script now names both directories as explained buckets.
- [ ] Refresh any cleanup plan from current files and full hashes. The
  classifier finds **590 short-name candidates**; suffix matching is not
  byte-identity proof.
- [ ] Seward 93 still resolves only by name; see the July note below and 01.

Explained buckets today: 2,414 non-sectional, 1,223 versos, 590 short-name
twins, 222 ledgered superseded sources, 179 ACASIS website import, 137 gap-dir
intermediates, 48 Ross flatbed tiles, 8 known dispositions, 1 whole-cycle bundle.

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

## Pericles donation + eBay purchases — 2026-10-04

49 M40 scans from October 2–3 were filed in `rawtiffs/Pericles, Matthew/` (6 files)
and `rawtiffs/eBay/` (43) and catalogued (7,695 → 7,744 rows; backup
`pre_pericles_ebay_scans_2026-10-04.csv`). Georef tool items A9–A11.

- 12 sectional sheets, 24 faces, 1980–92: Albuquerque 36, Atlanta 25, Dallas-Ft Worth 35,
  Denver 34, Jacksonville 27, Las Vegas 35, Los Angeles 38, New Orleans 43/44/45/50,
  Phoenix 34. None was held. Cutline and LCC prefilled from siblings, GCPs blank.
- First non-sectional rows: 6 TACs (Denver 25, Las Vegas 24, New Orleans 28 and 30,
  San Diego 11, Puerto Rico-VI 10) and 9 WAC sheets, 18 faces (CF-16 21, CF-17 21,
  CF-19 18, CG-18 16, CG-19 16 and 22, CG-20 20, CG-21 17, CH-25 22). TAC cutline
  blank; WAC rows carry `wac/<code>` (below).
- Flight Case Planning Chart 1985-08-01 is a CONUS planning chart, not a WAC. It has a
  row but should stay without GCPs unless it is wanted under the mosaics.

Chart type is the location suffix (` TAC`, ` WAC`, ` Planning Chart`;
`dole_v2.location_chart_type`). The slicer stacks one era's sources WAC, then
sectional, then TAC (`dole_v2.CHART_TYPE_LAYER`). Chart URIs come out as
`sectionals/chart/new_orleans_tac/<date>` and `sectionals/chart/cf_16_wac/<date>-north`.

`shapefiles/wac/` (new, 8 outlines) holds the nominal sheet limits read off the printed
corner labels: CF row 40–48N (CF-16 125–109W, CF-17 109–93W, CF-19 77–61W), CG row
32–40N (CG-18 125–111W, CG-19 114–100W, CG-20 100–86W, CG-21 86–72W), CH-25 24–32N
85–73W. The sheets bleed past these limits (CF-16 to about 49N, the others about 0.3°),
and the rectangle trims that bleed. The georef tool's cutline picker now lists every
shapefile folder (`terminal/`, `wac/`, …), and Check Overlay draws the selected
cutline's real outline in red over the cyan graticule (15′ steps on TAC-sized sheets).

Printed dates that differ from the scan names: Albuquerque ed 36 is 1985-11-21,
Denver TAC ed 25 is 1986-02-13. The Puerto Rico–Virgin Islands sheet is a TAC.

Open:
- The stacking only orders charts inside one era file. A WAC whose date range differs
  from the sectionals around it is its own era, and the viewer draws eras in effect by
  start date, later on top (`src/viewer.js`, `rangesInEffect.sort`). A WAC that starts
  after a sectional will cover it until the viewer learns chart types.
- slicerd on the NUC runs its own copy of the slicer; it does not have the stacking yet.
- Chesapeake Bay WAC (357) 1952 in the Roslewski folder is still uncatalogued.
- Not scanned from this pile: the enroute low charts, Flight Test Guide, South Carolina,
  Dominican Republic and Florida charts on the donation list.
