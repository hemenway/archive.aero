# 15 — Airspace overlay (FAA NASR)

Started 2026-09-14. The viewer gains a time-aware airspace layer built from the
NASR 28-day subscription series held in `/Volumes/projects/aisdata/us_faa/`
(see `scripts/aisdata_pull.py`). Class airspace first; the rest of the FAA's
airspace products follow the same pipeline once their history source is parsed.

## Design (settled 2026-09-14)

- **One vector PMTiles archive per build**, all 86 held cycles merged: a polygon
  *version* (geometry + drawing attributes) that is unchanged across consecutive
  cycles is stored once with a `from`/`to` validity interval. Churn is 1–2 % per
  cycle, so 86 cycles cost ~1.5× one cycle: 8,139 versions over 7,654 shapes.
- **Rendered by a second protomaps-leaflet layer** (`AirspaceLayer` in
  index.html) with sectional-legend paint rules; the timeline date is applied
  in the rule filters and `rerenderTiles()` repaints cached tiles — scrubbing
  never refetches, and repaints only when the cycle in effect changes. Outside
  the held window (before 2020-03-26, or 28 days past the newest cycle) nothing
  is drawn and the status line says why — only dates with data get airspace
  (Ryan's call, 2026-09-15; the first cut clamped to the earliest cycle).
- **Pin card "Airspace here"**: the class stack under the pin for the selected
  date, from the layer's decoded tiles (no second click handler, no second
  fetch). Sorted by floor; exclusion polygons drawn but not listed.
- **Dated immutable key** (`airspace/nasr-<stamp>.pmtiles`), same reasoning as
  the basemap; `--update-html` rewrites `CONFIG.airspaceUrl`.
- Off by default (opt-in, remembered): over a 1950s chart today's airspace is a
  comparison, not context.

## Facts measured on the way

| | |
|---|---|
| Class_Airspace.shp per cycle | ~5,600 polygons, 12.3 M vertices (arcs densified), NAD83 |
| Stable id | none in the shapefile (GLOBAL_ID is ADDS-only) → content hash |
| Ident churn | 1,329 idents KSTL→STL between 2020 and 2021, no geometry change → name/ident excluded from the hash |
| Antimeridian | FAA pre-splits CONTROL 1234L at 180°; Guam/Saipan at 145°E; no wrapdateline needed |
| Exclusions | 4 polygons (San Diego Class B notches), `ex=1` |
| /vsizip/ trap | ~2 GB leaked per cycle, 62 GB peak, two runs killed; unpack to a temp dir instead |
| 1,200 ft blankets | share every inner edge with the 700 ft areas → the blue vignette must be edge-aware |
| Vignette side (checked on the 2026 Elko sectional) | fade on the controlled side for both colours |

## State

- [x] A1. `scripts/airspace_build.py` — parse (resumable sqlite cache) → merge
      → floor edges → tippecanoe → metadata stamp → verify → manifest → upload.
- [x] A2. `scripts/dev_server.py` + launch config `viewer` — Range-capable
      static server so an unpublished archive can be viewed from the mirror.
- [x] A3. Viewer: Layers-panel row + chips (B/C/D, Class E) + cycle status
      line; `RibbonSymbolizer` vignettes; pin-card stack; attribution.
- [ ] A4. Publish: `--upload --update-html index.html`, then commit + push
      (the archive is unreferenced until index.html deploys).
- [ ] A5. Add the FAA NASR entry to sources.html (done in the same change set).

## Next

- [ ] B1. Special-use airspace (MOA/R/P/W/A) + Mode C veils, current-only from
      the ADDS GeoJSON (`us_faa/adds/<date>/Special_Use_Airspace.geojson`,
      `Class_Airspace.geojson` MODE C rows) as a `sua` layer with a
      "current only" caveat in the panel.
- [ ] B2. SUA history from the per-cycle AIXM `SaaSubscriberFile.zip`
      (1,236 XML, GDAL's GML driver opens them; ~300 use ArcByCenterPoint —
      verify arc densification before trusting).
- [ ] B3. Labels: an anchor-point layer (polylabel) with floor/ceiling text,
      `CenteredTextSymbolizer`.
- [ ] B4. Airspace_Boundary set (TRSA, ADIZ, SFRA, ARTCC) — current-only.
- [ ] B5. Share-link state for the layer toggle (URI-POLICY rule 8: a new
      query parameter is permanent API — decide the name once).
