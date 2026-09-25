# Worklist 04 — Georeferencing and candidate-selection backlog

## Current status — 2026-09-21

Read-only catalog review through `scripts/dole_v2.py`: **7,672 rows**, **3,484
without complete GCPs**, including **117 dated before 2011**. These are metadata
counts, not the number needing hand work: native GeoTIFFs and world-file JPGs can
warp without row GCPs, while incorrect saved GCPs can still need repair. Only the
**two GlidePlan rows** have neither a cutline reference nor inline WKT; the four
Key West strips now explicitly use `cutline=none` and `proj=merc`.

Source availability is a separate prerequisite: [02](02_disk_vs_dole.md) found
142 catalog resolution misses in the current mounted tree, including several
lanes below. Locate/verify each source before editing GCPs. This review did not
render charts or run the slicer; publication claims below are from dated commits
and `data/chart_pmtiles/uploads.jsonl`, not a fresh live check.

| Remaining item | Evidence and next action |
|---|---|
| ~~**Boston 1953-12 → 1970 sheet extent (27 rows)**~~ | **Fixed and republished 2026-09-25** — see the 2026-09-25 entry below. `ca000440r` (1957-06-01) is the only Boston row of the period still open (hand redo, next row). |
| **Boston 1957 `ca000440r`** | Stored GCPs remain suspect; rotation is still 90, with `extents/boston_ma`. Redo against the actual 40–44N × 69–72W sheet, rotation 270, as diagnosed on September 15. Not published in that batch. |
| **SF 1971 WASP 415 pair** | Both have GCPs and inline cutlines, but `half` remains blank and SP remains 45/33. Set sides from scan evidence, compare projection residuals, verify fold coverage and republish; upload ledger has only the bare-date artifact. |
| **Juneau 2013 Batch19 hold conflict** | The row has GCPs although its note/sidebar still say hold; 2014 remains GCP-less. Resolve the August 19 preference for native ZIP editions before another slice; do not blindly apply/reapply inference. |
| **Hawaiian 1947–66** | 15 ca-series rows still lack GCPs (includes `ca001666X`); two-panel scans need per-panel treatment. 1968/69 rows already have GCPs. |
| **simviation 2010–11** | 55 rows still lack GCPs. Since September 12 they point to chart TIFFs extracted losslessly from embedded JPEGs; the old manual page-assembly instruction is superseded. |
| **Dutch Harbor 2004 south** | Row still lacks GCPs; review/apply the August 20 full-resolution fit after locating the source. |
| **Hawaiian 2004 archive.org JPG** | Still lacks GCPs. The separate AVSIM JPG has `src_crs=EPSG:4269` and provenance for its JGW; first verify that world-file path before assuming hand GCPs are needed. |
| **SF 1966 + ESRI SF 2008** | Still lack GCPs. SF 1978 north was published September 15 and is no longer in this queue. |
| **GlidePlan Mt. Shasta + Reno Whites** | Still lack GCPs and a cutline decision; consider explicit whole-sheet `none` for these collar-free mosaics, then verify the warp. |
| **Wichita 1972 WASP 418 pair** | `_01` lacks GCPs; `_02` has them, but both `half` fields are blank and SP is 45/33. Review the pair together to avoid the known half-collapse bug. |
| **Dallas 1981 north (419_01)** | Now has GCPs, `half=north`, corrected SP; no traced `cutline_wkt` and no north artifact in the upload ledger. Verify edges/fold and publication status before treating it as complete. |
| **Denver 1975 seam** | Both halves published September 15; a ≤4 km pale band was recorded. Reconcile their GCPs at the fold before republishing improved same-key artifacts. |
| **Tool status/sidebar** | `row_status()` still equates a GCP-less `sectional/` cutline with embedded georef. Correct that heuristic and refresh the stale A2/A4/B1 and B4 descriptions (01). |

**Completed since the original list:** most 2004 JPGs, Juneau 1994–2010, Cheyenne
2009, Dallas 1981 south/1982 both halves, Key West 1928–35, Chicago 1939,
Portland 1954, Reno 1960, Denver 1972/1975 both halves, SF 1978 north and
Cincinnati 2011 north. The dated notes below preserve methods and limitations.
The deferred HUNT26 import belongs to [07](07_download_queue.md), not the active
catalog's GCP count. Seward 93 is a replacement-source problem in 01.

## Dated progress and decisions

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
