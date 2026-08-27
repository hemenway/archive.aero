# Worklist 04 — Georeferencing backlog (rows in the dole with no GCPs)

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
> would-be rows were backed out — re-run the importer when ready to start the
> hand-GCP effort (sidebar item text preserved in
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
> image — needs per-panel handling); `ca000815` (Chicago 1939) has a mid-file
> TIFF read error (re-fetch from LOC); Boston 1957 / Portland 1954 / Reno
> 1960 have no modern donor (discontinued sectionals) — hand GCPs only.

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

## Why this matters

`slicer.py` warps each row one of two ways: **(a)** the row's 4 GCPs (`gcp1_*`…`gcp4_*`), or
**(b)** fallback to the row's cutline shapefile **relying on georeferencing embedded in the
source file itself**. A row with no GCPs *and* no embedded georef cannot be warped — it is
silently absent from every mosaic it should appear in.

3,389 of 7,457 rows have empty GCP fields, but 3,163 of those are 2011+ FAA digital
products with native SRS — fallback (b) handles them. The real backlog is the
**226 pre-2011 rows** below.

## Breakdown of the 226

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

## Rows with no cutline at all (8)

These can't be warped even with GCPs — assign an `extents/` shapefile (or inline
`cutline_wkt`) first. See [`data/no_cutline.csv`](data/no_cutline.csv).

## Suggested workflow

1. Skip the 23 usahas mosaics (georef OK) — spot-check one in the output mosaics instead.
2. `gdalinfo` the 15 misc files; drop any with valid SRS from the list.
3. Batch the 142 half-sheets by chart name in georeftool — same chart, different editions
   share corner coordinates, so GCPs copy across editions with only pixel tweaks.
4. Do the 43 LOC scans the same way (most have `extents/` shapefiles already).
5. Re-run slicer for the affected date ranges only.

## How to regenerate

```bash
~/venv/bin/python - <<'EOF'
# rows with no gcp1_px and date < 2011 → data/georef_backlog.csv
# rows with empty cutline → data/no_cutline.csv
# (see worklists/README.md for the full snippet)
EOF
```
