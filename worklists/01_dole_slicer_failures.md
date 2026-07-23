# Worklist 01 — Dole rows that failed to process in the slicer

*Generated 2026-07-16 from `/Volumes/projects/2026-07-14 slicer run/slicer_run.log`
(161,719 lines; `L…` = log line numbers) + `processing_review_20260714_232303.csv`.*

> **Tracked live:** sections A–E are mirrored as the checkbox to-do list in the
> georef tool's Timeline sidebar (`scripts/1georef_toolv10.py`, items A1–E2, with
> live dole-row matching). Done-state persists in `worklists/data/georef_todo.json`.

## Run summary

| Metric | Value |
|---|---|
| Dole rows accepted at load | 7,457 / 7,457 (0 undated, 0 missing-filename) |
| Date ranges loaded → mosaics completed | 3,756 → 3,666 |
| Date ranges **skipped entirely** ("no data available") | **90** |
| Distinct source files that failed to warp and never succeeded | **222** |
| "No source files found" dole rows | 51 distinct |
| Warp-retry events → recovered on a duplicate option | 46 → only 3 recovered |

Work through the categories in order — A/B are highest-impact and partly mechanical;
C is the long georeferencing tail (overlaps [04_georef_backlog.md](04_georef_backlog.md)).

---

## A. GeoPDF SRS failures — the two crippled newest mosaics ⚠ TOP PRIORITY

Error: `Cannot compute bounding box of cutline. Cannot find source SRS`.
**Root cause corrected 2026-07-22:** the FAA `/PDFs/` sectionals (and the `_P` print
bundles, and the archive.org mirror) carry **no georeferencing at all** — they are Adobe
Photoshop image-conversion PDFs. No GDAL build reads georef from them; "use homebrew/conda
GDAL" was a dead end.

**Fix applied 2026-07-22:** corner GCPs + LCC params inferred from georeferenced sibling
editions (05-14/07-09-2026 FAA `sectional-files` GeoTIFFs, same 300-dpi/1:500k grid) via
NCC image correlation + RANSAC similarity fit (`scripts/georef_infer_from_sibling.py`),
written into the dole rows so a from-scratch rebuild needs nothing but the rows.

| Mosaic | Damage | Status |
|---|---|---|
| **2026-01-22_to_2026-03-19** | Only Wichita survived | ✔ rows carry inferred GCPs |
| **2026-03-19_to_2026-05-14** | 21 of ~52 locations | ✔ inferred GCPs; 20 more locations covered by existing wayback `03-19-2026 sectional-files` zip rows (native georef, captured 2026-03-29) |
| 2025-11-27_to_2026-01-22 | missing Hawaiian Islands + Denver | ✔ Denver row re-pointed at the real 20251225041526 capture (was a saved `Service/Unavailable` error page), PDF downloaded + converted; both rows carry inferred GCPs |

(2026-05-14 and 2026-07-09 ranges completed cleanly with 57 sources each.)

## B. Pipeline bugs (code/data mechanics, not georeferencing)

- [x] **`resolve_filename` misses PDFs inside zip dirs** — FIXED 2026-07-22: the resolver
  now indexes/extracts local zips it finds, rasterizes sheet-size PDFs found in extracted
  zip dirs at 300 dpi, and skips print-size pages. ⚠ BUT the `realcharts_sec*.zip`
  payloads turned out to be **half-size multi-page A4 print PDFs** (simviation realcharts,
  no georef, 5 pages each) — not machine-warpable; those ~21 ranges moved to section C
  (manual assembly + georeferencing).
- [x] **13 `realvfrcharts_sec*.pdf` files are actually ZIP archives** — FIXED 2026-07-22:
  extracted in place (real PDFs inside) and slicer now auto-detects zip payloads saved
  under a row's .pdf/.tif name and swaps in the member matching the filename
  (`_extract_row_file_from_zip_payload`, also hooked into download_file). ⚠ Same caveat:
  the PDFs are half-size multi-page print sets → georef work moved to section C.
- [x] **Wayback `_P.zip` print-PDF bundles contain no .tif** — resolver now materializes
  sheet-size PDFs from zip dirs. Hawaiian Islands 93: the native wayback
  `Hawaiian_Islands_93.zip` row (added by the coverage audit) carries fully georeferenced
  tifs for that range; the `_P` row is a documented last-resort fallback (its print PDFs
  have no georef).
- [x] **Seward_93.zip is a truncated 1.2 MB capture** — CONFIRMED UNRECOVERABLE 2026-07-22:
  it is the *only* CDX capture of that URL, truncated in the archive itself; no NARA copy
  (`Seward_SEC_93.tif` 404s). Row note documents it; range needs an alternative scan.
- [x] Silent-exhaustion gap — FIXED 2026-07-22: slicer now logs
  `✗ <location>: all N option(s) exhausted - location will be missing from <date>`.
- [x] *(new)* **0-byte mosaics logged as COMPLETE** — FIXED 2026-07-22: `create_mosaic_geotiff`
  fails loudly and deletes the output when the mosaic writes no pixels (<256 KB), and the
  row-GCP warp path pre-checks that the cutline bbox intersects the GCP footprint
  (`_gcp_cutline_mismatch`) so wrong-cutline/wrong-latlon rows fail with a pointed message
  instead of shipping empty rasters.
- [x] *(new)* **16-bit LOC scans** (Trinidad CO ca003733r–52, Portland OR ca003005r–21,
  23 files) — converted to 8-bit in place 2026-07-22, and slicer now rescales UInt16 tifs
  on download (`_ensure_8bit_tif`), so rebuilds stay Byte end-to-end.

## C. Sources needing georeferencing (GCPs) — 129 files, ~68 skipped ranges

All fail with the no-source-SRS error; they're plain scans/JPGs with cutlines but no GCPs.
This is the actionable core of [04_georef_backlog.md](04_georef_backlog.md).
*(2026-07-22: also inherits the simviation realcharts/realvfrcharts print-PDF sets from
§B — half-size multi-page A4 pages that need manual assembly before georeferencing.
Where a georeferenced sibling edition of the same layout exists, try
`scripts/georef_infer_from_sibling.py` before hand-placing GCPs.)*

| Group | Files | Ranges lost | Notes |
|---|---|---|---|
| 2003–05 archive.org **North/South half-sheet JPGs** | 78 (38 pairs + Cold Bay + Hawaiian Islands singles) | ~21 | Each pair was its range's only content (L141341–L141792) |
| **Juneau** NARA catalog Batch19 JPGs, eds 34–51 | 18 | 18 annual ranges 1994–2012 | The whole Juneau 1994-2012 timeline is dark until these get GCPs (L141186–L142937) |
| **Hawaiian Islands ca-series** 1947–1969 | 15 | 15 | ca001657r–ca001670 (each sole chart of its range) |
| **WASP scans** (Denver eds 6/13, Dallas ed 25) | 6 | 3 | 1972/1975/1981 — rare desert-era content! (L141066–L141144) |
| Individual ca-files: ca000440r (Boston 1957), ca000719r (Cheyenne 2009), ca000827r (Cincinnati 2011), ca003008r (Portland 1954), ca003126r (Reno 1960), ca000815 (Chicago 1939, non-fatal IFD noise too) | 6 | 5 (+1 silent gap) | |
| Ungeoreferenced **PDFs**: SF 1978, ESRI SF 2008, realcharts Whitehorse + Bethel | 4 | 4 | ⚠ then ✗ (L141127–L142797) |
| **No GCPs AND no cutline** (can't even attempt): Key West eds 1/3/7 (ca000001r/2r/3r/4r, 1928–1935!) + GlidePlan Reno_Whites + Mt_Shasta JPGs | 6 rows | 6 | Assign cutline shapefiles first (L281–L4502, L141851–56) |

## D. Corrupt / unusable sources — need replacement scans

- [x] **ca001479r.tif** (Fargo ND ed 27, 1952-11-19) — FIXED 2026-07-22. The LOC *master*
  tif is truncated **server-side** (every pull dies at the same byte); rebuilt the local
  file from the complete LOC *service* JP2 derivative (same 12444×6981 grid, row GCPs
  stay valid) and re-pointed the row's download_link at the jp2.
- [x] **Denver 2025-11-27 row** — FIXED (see §A).
- [ ] **Seward_93.zip** — unrecoverable online (see §B); needs an alternative scan.

## E. Silent single-chart holes in otherwise-completed mosaics

Mosaics built fine but are missing one chart — invisible unless you look:

| Missing chart | Mosaic range | Status 2026-07-22 |
|---|---|---|
| Chicago 1939 (ca000815) | 1939-02-01→1940-02-01 | still needs GCPs (§C) |
| Los Angeles SEC 101 | 2017-06-22→2017-12-07 | ✔ inferred GCPs override the garbage NARA georef (fit was pixel-exact — the NARA file is a digital raster, not a scan) |
| Washington SEC 101 | 2017-02-02→2017-07-20 | ✔ same |
| Hawaiian Islands 93 (main) | 2015-10-15→2016-04-28 | ✔ native wayback `Hawaiian_Islands_93.zip` row (georeferenced tifs) |
| Seward 93 | 2013-11-14→2014-05-29 | ✗ unrecoverable online (§B) |
| Hawaiian Islands + Denver | 2025-11-27→2026-01-22 | ✔ §A (Hawaiian needed a rotation-aware fit — the print strip is rotated 34° vs. geography) |

No wayback `Los_Angeles_101.zip` / `Washington_101.zip` captures exist (checked CDX
2026-07-22: LA stops at 99, Washington skips from 100 to 104), so the Denver-94-style
duplicate-row rescue was impossible — hence the GCP override. The NARA "Too large output
raster" cause is now understood: the embedded georef maps the whole raster to a
sub-arcsecond extent; row GCPs replace it because gdal.Translate with GCPs drops the
source geotransform.

## F. Benign noise (do NOT chase)

- 931 × `Several coordinate operations…` GDAL warnings — cosmetic.
- **859 of 860 `Skipping stale/unreadable mosaic source: *.vrt`** — leftover VRT
  intermediates from a prior run whose sources moved; each chart was re-warped fresh and
  the dead VRT deleted. Only the Fargo ca001479r VRT was real loss (§D).
- Winston-Salem ca004037–ca004049 IFD errors (`offset 490` / sanity check) — libtiff noise
  from a bogus second IFD; all 12 warps completed ✓. Same class: `Bogus StripByteCounts`
  on ca000815, antimeridian split notices (handled).
- `(N files)` cells in the review CSV — normal multi-file matches (insets, N/S splits).

## The 90 skipped ranges at a glance

4 Key West 1928–35 (no cutline) · 15 Hawaiian 1947–69 (§C) · 5 single-ca-file ranges ·
1 NOAA Anchorage 1970-11 · 3 WASP 1972/75/81 · 2 SF PDFs 1978/2008 · 2 GlidePlan 2006-07 ·
18 Juneau annuals 1994–2012 · ~21 N/S-jpg ranges 2003–05 · ~19 realcharts/realvfrcharts
ranges 2010–11 · 1 Cincinnati 2011 (ca000827r). Full cause-tagged list is in section 7 of
the agent log analysis; each range's failure lines immediately precede its skip line in
`slicer_run.log`.
