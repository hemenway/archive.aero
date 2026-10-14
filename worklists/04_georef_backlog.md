# Worklist 04 — Georeferencing and candidate-selection backlog

## Current status — 2026-10-13

**All six rebuilt eras in `2026-10-13_georef` and the five changed full-sheet artifacts are published and byte-verified.**
The five changed full-sheet artifacts are Boston 1957, SF 1971 north/south,
Dallas 1981 north and Dutch Harbor 2004 south. Mosaic-grid alignment errors are
≤0.01 px against a 0.3 px tolerance; all six final mosaics have been rendered and inspected.
Release evidence is at `~/archive-next-build/2026-10-13-georef`. Live manifest `3399e551dd77` and metadata
bundle `61e8c828` include 3,762 eras; all six rebuilt metadata records match their
archives byte for byte. The isolated shell build, budget check and deployment dry run pass. All nine
browser views passed (610 candidate responses, all 200/204, zero browser errors);
held historical airspace draws and a missing-cycle gap matches disabled-layer pixels.
Evidence is in the candidate's `qa/summary.json` and `qa/browser-results.json`.
After explicit publication approval, all 297 planned objects passed full-byte storage
verification. Public served-byte checks passed for 30 objects: three sampled ranges
on each of 23 archives, plus four cards, the bundle, manifest and timeline.
The tiles Worker and apex shell are deployed; **all ten live browser views and
all 50 airport page/JSON-LD checks passed**. The live run observed 2,579 successful
data responses, with no browser errors or failed requests. Final evidence is in
`qa-live/summary.json` and `qa-live/browser-results.json`.

The new source-aware B3 audit classifies **7,742 rows** as **4,237 GCP-ready,
3,398 source-georeferenced and 107 incomplete**. Those 107 are a triage queue,
not 107 confirmed manual-georef jobs: source replacements, cutline decisions and
existing inference can change the remedy. Metadata checks do not establish full
raster readability or warp accuracy, and incorrect complete GCPs can still need repair.
For comparison, the dated October 2 GCP-field-only review counted 7,695 rows,
3,482 without complete GCPs and 115 of those dated before 2011; do not use that
older predicate as today's work count. The two GlidePlan rows still need a
cutline decision; the four Key West strips explicitly use `cutline=none` and `proj=merc`.

Source availability: the October 4 rerun of [02](02_disk_vs_dole.md) resolves
every catalog row on disk; the 142 misses of September 21 were an incomplete walk. The October 1 batch verified all of its source files, rendered its warps, and
read back all published bytes.
See [October 1 publication](georef_publish_2026-10-01.md) for the completed batch.

| Remaining item | Evidence and next action |
|---|---|
| **Corner-GCP fit check — first catalog run (new 2026-10-07)** | `python3 scripts/dole_v2.py check-gcps master_dole_v2.csv` fits every GCP row's four corners in its chart projection and flags a pixel square that does not land as a ground square (aspect off by >3% warn, >10% error; the Boston 1953–70 rows read 0.76), mirror-order corners, and corner residual (>500 m warn, the wrong-SP band; >5 km error). The slicer now **refuses error rows at load** (an alternate row can still win the group; `--allow-gcp-misfit` keeps them). **First real run 2026-10-12:** 4,237 GCP rows → 6 error, 47 warn (Boston 1957 was a seventh, fixed the same day). Errors, all aspect mismatches, i.e. corner lat/lons from another sheet format: Aroostook 1969-02-01 `ca000156r` (0.83), El Paso 1967-12-07 `ca001408r` (0.87), Nashville 1965-07-01 `ca002582r` (0.82), New York 2006-05-11 `New_York_SEC_73_North.tif` (0.89, a donated scan), and **Hawaiian Islands 1968-12-12 `ca001655r` / 1969-05-29 `ca001656r` (0.63, 18.7 km corner residual — published eras, almost certainly wrong corners)**. The slicer now refuses these six at load, so their eras lose the chart on the next reslice unless fixed or `--allow-gcp-misfit`. Warnings: 500–620 m residuals across Wichita 1933–59 (a systematic wrong-SP band), Cincinnati 1968–69 / San Antonio 1934–35 / New Orleans 1956 / Twin Cities 1967 aspect 0.92–0.96, Mobile 1935 3.6 km, Aberdeen 1951 2.1 km. Review the errors first (read the printed corner labels), then decide whether the aspect thresholds fit the 1930s formats. |
| ~~**Boston 1953-12 → 1970 sheet extent (27 rows)**~~ | **Fixed and republished 2026-09-25** — see the 2026-09-25 entry below. The separate `ca000440r` (1957-06-01) redo was published October 13, next row. |
| **Boston 1957 `ca000440r`** | **Published and byte-verified 2026-10-13; postdeploy browser check passed.** The October 12 correction located the 72/69W × 44/40N graticule intersections at full resolution in the rotation-90 display frame (rotation 90 is right; the September "270" diagnosis was wrong): 74 m RMS, cutline `extents/boston_ma_1953`, backup `pre_boston_1957_gcps_2026-10-12.csv`. Era `1957-06-01_to_1957-11-21` and `chart/boston_ma/1957-06-01` are live at permanent and hashed addresses. The corresponding publication hold has been retired after byte and live-viewer checks. |
| **SF 1971 WASP 415 pair** | **Published and byte-verified 2026-10-13; postdeploy browser check passed.** Four graticule corners per face were re-read/refined, and the legend confirms SP 38°40′ / 33°20′. RMS fell from 1,574 to 221 m north and 623 to 52 m south. Traced fold edges intersected with `sectional/san_francisco` replace the rectangular GCP-box clips that left a 0.5° seam gap. The October 12 `half` values remain 415_01 south / 415_02 north. Evidence and backup are recorded in the dated section below. Era `1971-04-29_to_1971-11-11` and both half artifacts are live; the permanent bare-date key is retained. |
| **Juneau 2013 Batch19 hold conflict** | The 2013 scan (`…_023_001.jpg`) has GCPs although its note says withheld; 2014 (`…_012_001.jpg`) remains GCP-less. The live 2013 era and `chart/juneau/2013-04-04-north`/`-south` still come from the Juneau 53 wayback ZIP, so nothing published is wrong, but the next slice of 2013-04-04 would rank the scan first. Either blank the 2013 GCPs (backup + dated note) or drop the hold. The 2011/12/15/16 scans won on purpose (published from scan GCPs on August 19). Sidebar B4 now lists both rows, so the conflict is visible there. |
| **Hawaiian 1947–66** | 15 ca-series rows still lack GCPs (includes `ca001666X`); two-panel scans need per-panel treatment. 1968/69 rows already have GCPs. |
| **simviation 2010–11** | 55 rows still lack GCPs. Since September 12 they point to chart TIFFs extracted losslessly from embedded JPEGs; the old manual page-assembly instruction is superseded. |
| **Dutch Harbor 2004 south** | **Three rebuilt eras and south artifact published and byte-verified 2026-10-13; postdeploy browser check passed.** August 20 refinement fit was applied October 12 (`worklists/data/georef_batch2/georef_transfer_refine.json`: scale 1.0164, 10/18 windows, max residual 2.47 px; scan 5386×2025; backup `pre_dutch_harbor_2004s_gcps_2026-10-12.csv`). The three `2004-03-18_to_…` eras and `chart/dutch_harbor/2004-03-18-south` are live with the release metadata; two of those era rebuilds were byte-identical to the prior artifacts. |
| **Hawaiian 2004 archive.org JPG** | Optional upgrade, not a gap. The AVSIM JPG's world-file path works: it was published August 19 as `chart/hawaiian_islands/2004-07-01` (warp `georef`), and its live header bounds (161.18–154.06°W, 18.21–23.62°N, z8–10) were checked on October 2. The archive.org scan (`dole_gap_2026-07/archive2004/`) has the same ~6.8k px, is uncropped and has 3.4× the bytes; it needs hand GCPs only if it should replace the AVSIM source. |
| **ESRI SF 2008** | Still lacks GCPs. SF 1966 was verified and published October 1; SF 1978 north was published September 15. |
| **GlidePlan Mt. Shasta + Reno Whites** | Still lack GCPs and a cutline decision; consider explicit whole-sheet `none` for these collar-free mosaics, then verify the warp. |
| ~~**Wichita 1972 WASP 418 pair**~~ | **Published October 1**: both hand GCPs ready, sides explicit, modern SP verified by residuals, fold cutlines traced; era plus north/south chart artifacts read back. |
| **Dallas 1981 north (419_01)** | **Published and byte-verified 2026-10-13; postdeploy browser check passed.** The October 4 correction changed `gcp2_lon`/`gcp3_lon` from −94 to the printed −95; the traced fold cutline matches the 1982 north-face envelope within 0.02°. The check warp placed Oklahoma City, Tulsa and Amarillo correctly. Backups `pre_dallas_1981_north_lon_2026-10-04.csv`, `pre_fold_cutline_2026-10-04.csv`. Era `1981-01-22_to_1981-07-09` and `chart/dallas_ft_worth/1981-01-22-north` are live. The final live-viewer check passed; sidebar A8 is complete. |
| **Denver 1975 seam** | Both halves published September 15; a ≤4 km pale band was recorded. Reconcile their GCPs at the fold before republishing improved same-key artifacts. |
| ~~**Tool status/sidebar (B3)**~~ | **Fixed 2026-10-13.** `row_status()` now checks the source's pixel-to-ground metadata and a usable CRS (embedded or `src_crs`), including every face of extracted/unextracted ZIPs. A cutline, world file without CRS, or `src_crs` alone no longer hides an unreferenced scan. Read-only checks confirmed Hawaiian ca-scans, archive.org Hawaiian JPG, simviation Seward and Seward 93 are incomplete, while AVSIM + world file/CRS, ChartGeek and modern FAA sources remain georeferenced. Metadata probes are cached with source/sidecar signatures, persisted locally, and warmed in the background. A full 7,742-row check found 4,237 GCP-ready, 3,398 source-georeferenced and 107 incomplete rows; a fresh index using the saved metadata cache reproduced those counts in 5.1 seconds (the initial cold check took 211 seconds on SMB). Regression coverage: `tests/test_georef_status.py`. The tool remains gitignored local tooling; the 5001 instance was restarted and its live `/timeline_data` verified HTTP 200 with the same counts in 1.03 seconds. This status is a metadata check, not a verified warp or a full-raster readability test. |

**Completed since the original list:** most 2004 JPGs, Juneau 1994–2010, Cheyenne
2009, Dallas 1981 south/1982 both halves, Key West 1928–35, Chicago 1939,
Portland 1954, Reno 1960, Denver 1972/1975 both halves, SF 1978 north and
Cincinnati 2011 north. The dated notes below preserve methods and limitations.
The deferred HUNT26 import belongs to [07](07_download_queue.md), not the active
catalog's GCP count. Seward 93 is a replacement-source problem in 01.

## Dated progress and decisions

> **2026-10-13 (evening) — twelve 1985–1992 eras sliced and published from the eBay
> batch's georeferenced faces.** The timeline had two eras between 1982 and 1991
> (Dallas 1982, Cincinnati 1991); the catalog held 23 georef-ready faces there that
> had never been sliced. Run `/Volumes/projects/slicer-runs/2026-10-13_eighties` on
> the NUC: Albuquerque 36 north (`1985-11-21`), Dallas 35 south (`1986-01-16`),
> Denver TAC 25 (`1986-02-13`), and nine WAC pairs (CG-18/19/21 1986, CF-19 1988,
> CG-20 and CF-17 1991, CH-25, CF-16 and CG-19 1992), 21 full-sheet artifacts, every
> era re-converted with `-min-zoom 0` (the MCP pipeline does not add slicerctl's
> z0 floor; the next renderer draws nothing below an archive's min zoom), all
> alignment probes 0.00 px. **Fold margins:** the M40 scans end in white scanner
> background, not the dark background `wasp_fold_cutline.py` expected, and the
> georef tool's Crop button had boxed only the top of most faces. Nine WAC seams
> were already clean (their crops stop at the print edge); Albuquerque north
> carried 3,064 px of white below the print, CF-17 north 596 px, and the
> uncropped CG-18 pair 650/724 px (pale band across the mosaic at 36°N). The tracer
> now reads the fold-edge window once (43 s per 3.6 GB face instead of hours),
> takes `--max-walk`, and `--set-crop` moves the crop's fold side to the traced
> print edge + 40 px so the full-sheet artifact drops the background too; applied
> to those four faces (backups `pre_eighties_fold_cutlines_2026-10-13.csv`,
> `pre_fold_cutline_2026-10-13*.csv`), synced as release `20261013-151222`, the
> three eras re-sliced. Publication via the new `scripts/next_release_run.py`
> (plan → upload → permanent → stage) plus the worklist-18 overlay/manifest steps;
> evidence under `~/archive-next-build/2026-10-13-eighties/`. Still dark in that
> span: the 23 un-georeferenced faces (New Orleans 43/44/45/50, Las Vegas 35,
> Los Angeles 38, Phoenix 34, Denver 34, Jacksonville 27, Atlanta 25 N,
> Albuquerque 36 S, Dallas 35 N, five TACs, the 1985 planning chart).

> **2026-10-13 — release published after approval; browser follow-up passed.**
> All 297 planned objects were read back and verified byte for byte:
> 12 immutable archives, 11 permanent archive copies, four cards, 267 index files,
> the manifest, metadata bundle and timeline. The six era rebuilds add Boston 1957
> and replace SF 1971, Dallas 1981 and Dutch Harbor 2004–2005; two other 2004
> eras rebuilt byte-identically. All six permanent era addresses were published.
> Public checks passed for 23 archives (three sampled ranges each) plus four cards,
> the bundle, manifest and timeline. Evidence under
> `~/archive-next-build/2026-10-13-georef/publication/`:
> `verified-objects.jsonl`, `served-byte-checks.json` and deployment logs.
> Live manifest `next/manifest.3399e551dd77.json` and
> `sectionals/metadata-61e8c828.bundle` cover 3,762 eras; the shell has 7,121
> edition URLs and 50 airport records. The same release publishes recovered NASR
> history in `airspace/class-20261013.cce35551b42d.pmtiles`.
> Tiles Worker `16d60293-72c5-4b7d-b755-cb982f5e77e2` uses cache revision
> `2026-10-13-georef` for permanent chart, era and timeline keys; site Worker
> `4ef92237-747b-41e6-94e0-bb1afd3d94d8` serves the new manifest.
> All ten live browser views and all 50 airport page/JSON-LD checks passed;
> evidence is in the release folder’s `qa-live/`. Current production pointers and
> rebuild/rollback instructions are in [18](18_apex_cutover.md).

> **2026-10-13 — preparation before publication.**
> `/Volumes/projects/slicer-runs/2026-10-13_georef` completed slice, conversion and
> archive for Boston `1957-06-01_to_1957-11-21`, SF
> `1971-04-29_to_1971-11-11`, Dallas `1981-01-22_to_1981-07-09`, and Dutch Harbor
> `2004-03-18_to_2004-09-02`, `…_to_2004-09-30`, `…_to_2005-03-17`.
> All six mosaic-grid alignment errors are ≤0.01 px (tolerance 0.3 px).
> Five changed full-sheet keys are built: `chart/boston_ma/1957-06-01`,
> `chart/san_francisco/1971-04-29-north`, `…-south`,
> `chart/dallas_ft_worth/1981-01-22-north`, and
> `chart/dutch_harbor/2004-03-18-south`. Boston/Dallas rendered checks are complete.
> SF's four corners per face, printed SP 38°40′ / 33°20′, and fold cutlines were
> saved after backup `pre_sf1971_grid_and_fold_2026-10-13.csv`; north/south RMS
> are 221/52 m. Exact changes and source evidence:
> `data/georef_2026-10-13/catalog_change.json`, `sf_refinement.json`,
> `sf_proposed.json` and `sf-geographic-check.png`.
> Candidate `~/archive-next-build/2026-10-13-georef` is being assembled with
> the new airspace build (30 verified older NASR cycles; 153 held regional cycles,
> including 116 US). No live publication has occurred. Finish candidate review,
> publish both permanent and hashed artifacts with their metadata, then verify
> served data and viewer behavior. B3's source-aware status implementation is
> complete; the server restart is being handled separately. The wider backlog
> remains open. The [Jev acceptance review](16_jev_validation_2026-10-13.md) also
> leaves source-review corrections open; this release preserves the existing
> published airfields reference, whose hash matches the local artifact.

> **2026-10-12 — four rows made slice-ready, no slicing yet.** Boston 1957 corners
> redone, SF 1971 halves set, Dutch Harbor 2004 south fit applied, Dallas 1981
> north already fixed (10-04). All four wait on one NUC run:
> `slicerctl sync` → `slicerctl pipeline 2026-10-12_georef 1957-06-01_to_1957-11-21
> 1971-04-29_to_1971-11-11 1981-01-22_to_1981-07-09 2004-03-18_to_2004-09-02
> 2004-03-18_to_2004-09-30 2004-03-18_to_2005-03-17 --charts` → eyeball each
> mosaic → `slicerctl publish` per era (`--allow-new` for 1957-06-01) →
> `publish_chart_pmtiles.py` for the five chart artifacts → rebuild
> `timeline_data.json`. Still hand work with no donor in reach: the 21 remaining
> M40 faces of A9 (nearest georeferenced siblings are 2004 JPGs, 15–25 years
> apart — beyond what the NCC transfer survived on the WASP scans), the 24
> TAC/WAC faces of A10, A11, the 55 simviation 2010–11 rows (no sibling of any
> kind except Juneau), the Hawaiian 1947–66 two-panel scans. The reslice error
> log (01, 2026-10-12 section) lists what else is dark.

*Generated 2026-07-16 from `master_dole_v2.csv` (7,457 rows). Row-level detail:
[`data/georef_backlog.csv`](data/georef_backlog.csv) (226 rows) and
[`data/no_cutline.csv`](data/no_cutline.csv) (8 rows).*

> **2026-07-25 — HUNT26 staged, then reverted to the 6 ready zips.** The 264
> "new" download-queue candidates were fully identified, dated, and staged for
> import (reviewed plan: [`data/hunt26_plan.csv`](data/hunt26_plan.csv);
> importer: `scripts/import_hunt26.py`), but only the **6 tfw-georeferenced
> wayback FAA zips** were kept in the dole
> (`/Volumes/projects/rawtiffs/hunt_2026-07/wayback_old_layout/`). The other
> 258 files went back to `/Volumes/projects/rawtiffcandidates/` and their 229
> would-be rows were backed out — revalidate the plan and current holdings before
> importing any further hand-GCP work (sidebar item text preserved in
> `data/hunt26_worklist_items.py`).

> **2026-08-19 — Juneau NARA lane closed.** All 12 rg-370 Batch19 Juneau scans
> solved by the new affine variant `scripts/georef_infer_affine_jpg.py` (the
> similarity fit failed on them because the scans are skewed 0.05–0.2°). 10
> applied after visual approval → eras 1994–2010 published (dates.csv 3,709,
> bundle `metadata-025bed1d`); 2013/2014 GCPs deliberately withheld — their
> wayback-zip editions must keep winning the candidate group (GCPs force
> rank 1). The 09-03-2026 FAA cycle was cataloged + published the same night.

> **2026-08-20 — batch-2 rejection post-mortem: the overlays lied, not (only)
> the fits.** Ryan rejected four loose-threshold fits as "misaligned due
> south" — root cause found and measured: the *approval-preview* warp fed the
> 4 corner GCPs to gdal.Warp in lat/lon, and GDAL's affine-in-degrees fit is
> exact at the corners but bows mid-chart south by the sagitta of the curved
> LCC parallels (**17.9 km measured** on the 9.6°-wide Cheyenne sheet). The
> slicer publishes via GCPs in **LCC meters** (truly affine) and is unaffected
> — live data verified correct. Preview fixed to the LCC path (18 m residual).
> A full-res refinement pass (768 px templates re-matched around a coarse
> seed, strict 3 px band, target-space spread gate) then produced tight fits
> for Dutch Harbor 2004 S / Cheyenne 2009 / Cincinnati 2011 — each verified
> against its own printed graticule (85 m–1 km, at scan resolution). WASP
> scans are beyond automated reach (terrain shading correlates at 1/8 scale
> but zero full-res windows survive 40 years of symbology change): hand GCPs.
> Also confirmed: Hawaiian 1947–66 ca-scans are TWO-PANEL (recto+verso on one
> image — needs per-panel handling); `ca000815` (Chicago 1939) had a mid-file
> TIFF read error — **fixed 2026-09-15**: the LOC *master* tif is truncated
> server-side at row 4,548 of 7,069 (same byte count on tile.loc.gov, so a
> re-fetch cannot help); rebuilt from the complete *service* jp2, link
> re-pointed, truncated master quarantined — the Fargo ca001479r recipe.
> `dole_v2.open_raw` now decodes eagerly and falls back to a GDAL partial
> read (missing rows white) so a truncated file shows its surviving rows in
> the georef tool instead of a blank 500; Boston 1957 / Portland 1954 / Reno
> 1960 have no modern donor (discontinued sectionals) — hand GCPs only.

> **2026-09-15 — georef-tool batch sliced + published** (13 hand-GCP'd rows
> from the tool since the 09-01 publish → 11 eras: Key West 1928/1929/1932/1935
> Navy strips, Chicago 1939 (same-key rebuild), Portland 1954, Reno 1960,
> Denver WASP 1972 + 1975 (both faces), SF 1978 north face, Cincinnati 2011
> north face; +13 chart artifacts). Pre-slice affine-residual test on every
> row (4 corners, RMS): 31–270 m except where the projection was wrong —
> **(1)** the Key West strips fit **Mercator** to 30–80 m and LCC 45/33 to
> 0.8–1.3 km (polyconic/tmerc no better): new optional catalog column
> `proj` (`merc`; blank = LCC), honoured by `dole_v2.row_lcc_crs`; the tool's
> LCC-only fit check will warn on those rows — Save anyway.
> **(2)** Denver 1972/75 WASP: 45/33 → 33°20′/38°40′ (300 m → 80–240 m), the
> SP warning above was right. **(3)** Cutline refs copied from older sheets:
> SF 1978 had `extents/san_francisco_ca` (the 1966 sheet, 34–38N — warp
> refused, no intersection) and Cincinnati 2011 had `extents/cincinnati_oh`
> (−90..−84 — clipped the modern north face to a 1° sliver, silently);
> both now `sectional/<modern>`. Check the cutline extent against the GCP
> box before slicing any hand-GCP'd row. **(4)** Denver WASP `half` was
> blank → the faces were warp *alternatives* (one face published as the
> era, bare `chart/denver/<date>` URIs minted — deleted before upload, not
> live); `half` set from the GCP latitudes, fold cutlines traced by the new
> `scripts/wasp_fold_cutline.py` (paper edge → blank margin → first inky
> block, per-column clamped to the face's p20–p50 margin band, ∩ outline).
> 1972 seam is clean; **1975 keeps a ≤4 km pale band** (its faces' georefs
> disagree ~1 km at the seam — any deeper cut opens a gap; `--clamp-hi 20`)
> — re-GCP one 1975 face against the other to close it.
> **Not published: Boston 1957 ca000440r** — the sheet is 40–44N × 69–72W
> (corner labels read off the scan), the pre-filled lat/lon and
> `extents/boston_ma` say 41–44, the scan needs `rotation` 270 (text reads
> upright only after 90° CCW), and the stored pixel GCPs sit at ~½ scale
> inside the sheet — redo it in the tool from scratch; nothing of it was
> uploaded. **Not sliced: Juneau 2013-04-04 Batch19** — it now carries GCPs
> but B4 says hold; the row's note still says withheld. Decide, then blank
> the GCPs or drop B4.

> **2026-09-25 (afternoon) — Boston 1953-12 → 1970 bottom latitude FIXED,
> 26 eras + 26 chart artifacts republished.** Verified first: the printed
> corner labels on `ca000447r` (ed 32) and `ca000420r` (ed 65) read 40° at
> the stored bottom-GCP rows and 72°/69° along that edge, while `ca000449r`
> (ed 31) reads 41° there; the GCP pixel-span ratio 1.73 is exactly a
> 40–44N × 3° sheet in LCC 45/33 (444 km tall over the 256 km 40N edge) and
> 1.32 exactly the 41–44N one. Catalog: `gcp3_lat`/`gcp4_lat` 41 → 40.0 on
> **27 rows** (26 non-blank + `ca000433` BLANK_MAP, which is sliced and has
> a published artifact like any other), dated note on each, backups
> `pre_boston_lat_fix_2026-09-25.csv` / `_2.csv` / `pre_boston_extent_1953_…`.
> New extent shapefile `extents/boston_ma_1953` (40–44N × 72–69W, NAD83,
> same schema as `boston_ma`, precedent `washington_dc_1932/1954`) and the 27
> rows point at it; `extents/boston_ma` (41–44) stays for the 35 older rows
> and still wins the timeline's per-location majority vote, so
> timeline_data.json/coverage.json are unchanged. Run dir
> `/Volumes/projects/2026-09-25 boston fix slicer run/` (drive mirror still
> unmounted): one slicer invocation per start date (`--start-date S
> --end-date S`; the filter is by key *start* date, so sibling keys came
> along — 11 were rebuilt before 1-byte placeholder mosaics made the slicer
> skip the other 19; none uploaded), fresh temps, g2p built from upstream
> main b300c9e (run-length fix merged 09-19; the Xcode-licence block on cgo
> is bypassed with `DEVELOPER_DIR=/Library/Developer/CommandLineTools`).
> Rendered checks with reference crosshairs (Logan 42.36, Provincetown 42.05,
> Chatham 41.68, Nantucket 41.28, all on their symbols) on 1953-12-02,
> 1965-03-04 (multi-chart) and 1970-03-05 mosaics plus a z11 tile of the
> 1953-12-02 artifact. Uploads: **26 era keys** (same-key republish, 0.99 →
> 1.19 GB; the 4° sheet is 1.33× the area) and **26 `chart/boston_ma/<date>`
> artifacts** (364 MB), each `rclone copyto` + size-verified; bundle
> `metadata-83ed9b48.bundle`. Same-key republish caveat
> ([06](06_publish_sync.md)): the tiles Worker's per-block edge cache is
> version-less, so a post-upload pass re-read every block of all 52 objects
> with `Cache-Control: no-cache`, which the Worker treats as bypass **and
> refreshes the cached block** — clean at the colo serving this machine,
> ≤24 h elsewhere. `build_metadata_bundle.py`'s RemoteSource now sends
> no-cache too, so a bundle built right after a republish can no longer
> embed a stale prefix. Open: `ca000440r` (below).

> **2026-09-25 — Boston 1953-12 → 1970: 27 rows carry the wrong bottom
> latitude (found, not fixed).** Animating Cape Cod at z9 with
> `scripts/era_gif.py` put every frame from 1954-04 to 1970-04 with
> Provincetown at ~42.54N (true 42.05) and Nantucket at ~41.96 (true 41.28),
> while the 1933–53 frames sit right. Cause: from edition 32 (`ca000447r`,
> 1953-12-02) through 65 (`ca000420r`/`ca000422r`, 1970-03-05) the Boston
> sheet is 44–40N × 72–69W — on `ca000420r` the printed "40°" sits at the
> stored bottom GCP rows (~11200) and "44°" / "72°" / "69°" at the top ones —
> but the rows keep the older sheet's 44/41 corners. The GCP pixel spans show
> it without images: y/x ≈ 1.72–1.74 on those rows vs 1.32 on every 1933–53
> row (`dole_v2.row_gcp_pixels`). The warp pins 44N and squeezes the 4° sheet
> into 3°, so everything south of 44N drifts north (Boston → 42.77). Affects
> every published era 1953-12-02 → 1970-09-17 (the 1965–70 ones are
> multi-chart eras) and the `chart/boston_ma/<date>` artifacts. The 09-15
> `ca000440r` diagnosis (40–44 sheet) was this same format change seen on one
> row; `ca000440r` (yspan 3500, edition "84") stays a hand redo and
> `ca000433` (BLANK_MAP) is excluded, leaving 26 rows for the bulk fix. Fix:
> back up the CSV, set `gcp3_lat`/`gcp4_lat` = 40.0 on those 26 with a dated
> note, check `extents/boston_ma` (41–44) against the new box — a 41–44 clip
> only drops ocean and W-506, but the extents should say 40–44 — verify one
> warp visually (Provincetown at 42.05N), re-slice the affected date range,
> then bundle + chart artifacts + timeline per [06](06_publish_sync.md). The
> same per-location aspect-ratio check would flag any other sheet whose
> format changed mid-series under copied corners.

> **2026-08-26 — Dallas–Ft Worth WASP 1981/1982 sliced + published** (hand
> GCPs by Ryan, three of four sides). Three infrastructure fixes rode along:
> **(1)** New optional catalog column `half` (+ `dole_v2.row_half()`): the
> WASP `_01/_02` suffixes are recto/verso scan order, not sides — 419's `_01`
> is north, 420's `_01` is south — so the slicer's cardinal-filename half
> detection saw the 1982 pair as warp *alternatives*, published one half as
> the whole era (live since 07-28) and minted the bare
> `chart/dallas_ft_worth/1982-01-21` URI (now permanently stale; halves live
> at `-north`/`-south`). `half` overrides both the candidate-group split and
> the chart-URI suffix when the stem has no cardinal token.
> **(2)** SP correction: all four rows carried LCC 45/33 or a 38.666/33.222
> variant; affine-residual test against the corner GCPs says the modern
> 33°20′/38°40′ parallels fit 2–10× better (25 m vs 237 m RMS on 419_02) —
> these are modern-projection sheets. **The 1971–75 WASP rows (SF 415 pair,
> Denver 416/417, Wichita 418) still say 45/33 — rerun the residual test
> before slicing any of them.** The live `1971-04-29` era (SF pair) also has
> the half-collapse bug: needs `half` values + reslice + era republish.
> **(3)** Fold-margin cutlines: a folded sheet's scan carries an unprinted
> margin + dark edge at the fold that the shared chart-ring cutline can't
> remove (8 km dark band across the era mosaic), and the fold line bows
> ~0.08° in latitude (LCC curvature) so no straight cut works. Fix = per-row
> `cutline_wkt` tracing the printed-face boundary (edge-attached dark-run +
> luminance walk, p10/p90 bias 30 px into ink, clamped to keep both-halves
> coverage); the 1982 faces overlap ~0.05° at the fold so the seam is
> feature-continuous. 419_01 (1981 north) is still unreadied: its GCP
> prefill (-94 east edge, 45/33 SP) predates all of this — re-derive when
> hand-GCP'd, and give it its own traced fold cutline.

## October 1 completed batch

[17 eras + 30 chart artifacts](georef_publish_2026-10-01.md) were verified and
published, including all donated sectional faces, SF 1966, Denver 1972 and
Wichita 1972. These are closed; the remaining table above stays actionable.

## Interpreting the historical counts

`slicer.py` warps each row one of two ways: **(a)** the row's 4 GCPs (`gcp1_*`…`gcp4_*`), or
**(b)** fallback to the row's cutline shapefile **relying on georeferencing embedded in the
source file itself**. A row with no GCPs *and* no embedded georef cannot be warped — it is
absent from affected mosaics; the slicer now logs exhausted candidates rather
than silently treating the location as complete.

**In the July 16 baseline**, 3,389 of 7,457 rows have empty GCP fields, but 3,163 of those are 2011+ FAA digital
products with native SRS — fallback (b) handles them. The real backlog is the
**226 pre-2011 candidates** below (including 23 already-georeferenced mosaics).
Do not use that historical total as today's backlog.

### July 16 breakdown of the 226 candidates

| Group | Rows | Embedded georef? | Action needed |
|---|---|---|---|
| archive.org 2004 half-sheet JPGs + other 2003-2010 web finds | 142 | ❌ none (plain JPG/TIF) | GCPs via georeftool (`1georef_toolv10.py`) |
| LOC `ca*` scans (scattered 1928–1972, incl. the 6 rows added 2026-07-14: Portland ×4, Milwaukee, Hawaiian Is. run) | 43 | ❌ none | GCPs via georeftool |
| usahas ChartGeek 2009 mosaics | 23 | ✅ EPSG:4326 from KML LatLonBoxes | **None** — fallback (b) works; verify they appear in the 2009 mosaics |
| Misc zips/tifs (wayback FAA 2010-11 zips, NOAA Anchorage 1970, GlidePlan, etc.) | 15 | mixed — FAA zips contain GeoTIFFs (OK); scans do not | Triage per file: open with `gdalinfo`, GCP only the ones without SRS |
| PDFs (ESRI SF 2008, SF 1966/1978 archive.org) | 3 | ❌ (pip GDAL wheel reads no FAA GeoPDF georef; these aren't GeoPDFs anyway) | Convert at 300 dpi (slicer convention) then GCP |

The bulk of this list is the **189 rows cataloged 2026-07-14** from the Tier-1 downloads
(`search_archive/catalog_additions_report.csv`) — ACTION_PLAN's open item
"*Push the new rows through the slicer pipeline (GCPs/georef via georeftool first)*".

## Working and rechecking the remainder

1. Load the current catalog via `scripts/dole_v2.py` and resolve the source (02).
2. Check embedded SRS/world files plus `src_crs`; missing GCP fields alone are not
   a failure. Use `row_cutline()` so inline WKT and explicit `none` count correctly.
3. Review any existing inference/hand-GCP result, projection, rotation, half-sheet
   grouping and cutline footprint. Warp and visually inspect one representative
   case before batching. Use projection-space GCPs for preview as well as slicing.
4. Back up the catalog before writes and record method/source/fit/date in notes.
   Slice affected ranges only; publication and metadata follow [06](06_publish_sync.md).

The linked `data/georef_backlog.csv` and `data/no_cutline.csv` remain July 16
snapshots. Regenerate a new dated audit from the loader when starting work; the
old date-before-2011/blank-first-GCP predicate is a triage filter, not an embedded
SRS check or a publication test.
