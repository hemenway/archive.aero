# Worklist 01 — Dole rows that failed to process in the slicer

*Generated 2026-07-16 from `/Volumes/projects/2026-07-14 slicer run/slicer_run.log`
(161,719 lines; `L…` = log line numbers) + `processing_review_20260714_232303.csv`.*

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

Error: `Cannot compute bounding box of cutline. Cannot find source SRS` — the pip-wheel
GDAL reads no georeferencing from FAA GeoPDF-converted TIFs (known issue).
Fix: re-warp with an SRS-capable GDAL build (e.g. conda/homebrew GDAL with PDF driver) or
add GCPs to the rows.

| Mosaic | Damage | Failed files |
|---|---|---|
| **2026-01-22_to_2026-03-19** | **Only 1 of 43 locations survived (Wichita)** | 42 wayback `…01-22-2026_PDFs_<name>.tif` files (L160865–L160946) |
| **2026-03-19_to_2026-05-14** | 21 of ~52 locations (20 restored intermediates + fresh Chicago) | 27 `archive.org_…Sectional_Charts_<name>.tif` + 5 wayback files (L161031–L161115) |
| 2025-11-27_to_2026-01-22 | 51 of 53 — missing Hawaiian Islands (L160723) + Denver | Denver's row is **broken data, not SRS**: its download_link is literally `https://web.archive.org/web/Service/Unavailable</h1>` and the saved file is a wayback error page. **Fix the dole row** (L160720). |

(2026-05-14 and 2026-07-09 ranges completed cleanly with 57 sources each.)

## B. Pipeline bugs (code/data mechanics, not georeferencing)

- [ ] **`resolve_filename` misses PDFs inside zip dirs** ([slicer.py:726](../scripts/slicer.py#L726)
  globs only `**/*.tif`) → **38 `realcharts_sec*.zip` rows dropped** ("No source files
  found") even though the extracted dirs exist (double-nested, e.g.
  `realcharts_secJacksonville/realcharts_secJacksonville/realcharts_secJacksonville.pdf`).
  Every 2010–2011 range those rows anchor was skipped (~21 ranges). Fix the resolver to
  also glob `*.pdf` and rasterize (the 300-dpi conversion path already exists).
- [ ] **13 `realvfrcharts_sec*.pdf` files are actually ZIP archives** — the prep downloader
  saved `simviation.com/1/download?file=realvfrcharts_alaska_pt*.zip` raw bytes under the
  dole `.pdf` filenames (`file` says `Zip archive data`). GDAL can't open them; all their
  Alaska 2009–2011 ranges skipped. **Unzip each in place** (real PDFs are inside) or
  re-point the rows. Files: Anchorage, CapeLisburne, Dawson, DutchHarbor, Fairbanks,
  Juneau, Ketchikan, Kodiak, McGrath, Nome, PointBarrow, Seward, WesternAleutianIslands
  (L142046–L142794).
- [ ] **Wayback `_P.zip` print-PDF bundles contain no .tif** → same resolver gap:
  `Hawaiian_Islands_90_P.zip` recovered via a duplicate row, but
  `Hawaiian_Islands_93_P.zip` (2015-10-15→2016-04-28) had no fallback — that mosaic shipped
  with the HI insets but **without the main Hawaiian Islands chart** (L147367).
- [ ] **Seward_93.zip is a truncated 1.2 MB capture** — extraction fails, re-download dies
  at the same byte (`IncompleteRead`, L70–76, L145520–29). Re-pull the largest CDX capture
  (also on Worklists 02/03).
- [ ] Silent-exhaustion gap: when a row runs out of duplicate options the log says nothing —
  consider adding an explicit "all options exhausted" line to slicer.py.

## C. Sources needing georeferencing (GCPs) — 129 files, ~68 skipped ranges

All fail with the no-source-SRS error; they're plain scans/JPGs with cutlines but no GCPs.
This is the actionable core of [04_georef_backlog.md](04_georef_backlog.md):

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

- [ ] **ca001479r.tif** (Fargo ND ed 27, 1952-11-19) — truncated TIFF, `TIFFReadEncodedStrip`
  read error mid-file; dole note already says "Tiff file corrupt". Range skipped; this is
  the run's one real data-loss VRT eviction (L80093–98). Re-download from LOC.
- [ ] **Denver 2025-11-27 row** — garbage `Service_Unavailable` download_link (see §A).
- [ ] **Seward_93.zip** (see §B).

## E. Silent single-chart holes in otherwise-completed mosaics

Mosaics built fine but are missing one chart — invisible unless you look:

| Missing chart | Mosaic range | Cause |
|---|---|---|
| Chicago 1939 (ca000815) | 1939-02-01→1940-02-01 | no SRS |
| Los Angeles SEC 101 | 2017-06-22→2017-12-07 | NARA tif "Too large output raster size" (bad NARA georef), no duplicate row (L149657) |
| Washington SEC 101 | 2017-02-02→2017-07-20 | same (L149177) |
| Hawaiian Islands 93 (main) | 2015-10-15→2016-04-28 | `_P.zip` resolver gap (§B) |
| Seward 93 | 2013-11-14→2014-05-29 | truncated zip (§B) |
| Hawaiian Islands + Denver | 2025-11-27→2026-01-22 | §A |

Denver SEC 94 and San Francisco SEC 97 hit the same NARA too-large error but recovered via
wayback duplicate zips — the pattern to replicate for LA/Washington 101: add wayback
`<City>_101.zip` duplicate rows.

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
