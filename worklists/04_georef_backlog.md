# Worklist 04 — Georeferencing backlog (rows in the dole with no GCPs)

*Generated 2026-07-16 from `master_dole_v2.csv` (7,457 rows). Row-level detail:
[`data/georef_backlog.csv`](data/georef_backlog.csv) (226 rows) and
[`data/no_cutline.csv`](data/no_cutline.csv) (8 rows).*

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
