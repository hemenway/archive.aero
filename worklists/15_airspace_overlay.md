# 15 — Airspace overlay (class airspace, three regions)

Started 2026-09-14. The viewer gains a time-aware airspace layer built from the
AIS series held in `/Volumes/projects/aisdata/` (see `scripts/aisdata_pull.py`):
FAA NASR first (US, 2026-09-15), then the other regions on disk (2026-09-15/16):
France (SIA per-AIRAC exports) and Brazil (DECEA GeoAISWEB snapshots).
Switzerland's BAZL geodata has no airspace vectors, so nothing to add there.
Class airspace first; special-use airspace follows the same pipeline once its
history sources are parsed (all three regions have it).

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

## Regions (added 2026-09-16)

- **One archive, three regions**: every feature carries `rg` (us/fr/br) beside
  `from`/`to`; the metadata's `archive_aero.regions.<rg>` holds that region's
  cycle list, extent `boxes` and counts. The viewer resolves the cycle in
  effect **per region** — the newest held cycle on or before the date, valid
  28 days — so a hole in a series (France 2020-09..2021-01 and 2022-04..
  2023-09) or a date past a region's newest cycle draws nothing for that region
  while the others still draw. The status line lists the regions whose boxes
  meet the view, one line each; the pin card's title names the source and
  cycle of the region under the pin.
- **Symbology keyed on the ICAO class**, not the FAA local type: A/B solid
  blue, C solid magenta, D dashed blue, E dashed magenta (FAA E5–E7 stay
  vignette-only). Chip 1 is "Class A – D". Brazil's WFS publishes no class, so
  its TMA/CTR/CTA/ATZ draw as plain solid blue and the pin card badge shows
  the type instead (`.pin-as-cls-type`), with the metadata `note` under the
  rows.
- **France (SIA XML, not AIXM)**: `Partie/Geometrie` is the polygon already
  densified by the SIA (arcs, circles, border-following segments resolved),
  `Volume` rows are the stacked shelves with `Classe`, floor/ceiling
  (`SFC`, `ft ASFC`, `ft AMSL`, `FL`, `UNL`) and `HorCode`. Kept: CTR, TMA,
  CTA volumes of class A–E, plus LTA parts that have a Class E volume (the
  Alps/Pyrenees parts drawn on the OACI chart; the national FL115 Class D
  blanket is skipped). Skipped: FIR/UIR/UTA/OCA/FRA, CTL/SIV sectors,
  RMZ/TMZ, "other" (cross-border delegations), every SUA type. Two exports
  exist for 2020-03-26; the newer `SiaExport Date` wins. 34 cycles → 810
  versions over 655 shapes; churn is at the AIRAC redesigns (2019-05,
  2020-03, 2021-03, 2023-10).
- **Brazil (GeoAISWEB)**: the TMA layer already contains the shelf parts
  (SBXP, SBXP_01, SBXP_02…); `setores_tma` is ATC sectorisation (SECT 07/09
  with the TMA floor), not shelves — not drawn. The cycle is the AIP
  amendment (`emenda`, 2026-09-03) the snapshot was taken under; each row's
  own `effectived` is kept as `eff` but never extends validity backwards.
  158 versions from 162 rows (4 outlines duplicate their "_01" shelf exactly).
- **Caches**: one sqlite per region (`class_versions.sqlite` is the untouched
  US cache, `class_versions_fr.sqlite`, `class_versions_br.sqlite`), keyed on
  the pull manifest's sha256 per file (France 34 cycles parse in 11 s).
- **Key**: `airspace/class-<stamp>.pmtiles`, stamp = build date; the US-only
  `nasr-20261001` stays in R2 untouched (never delete a published key).

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
- [x] A4. Publish. index.html shipped in 5d6379a (2026-09-15) **before the
      archive was uploaded**, so the live toggle failed with "Airspace data
      unavailable" until `rclone copyto` put `airspace/nasr-20261001.pmtiles`
      in R2 later that day (`worklists/data/airspace/uploads.jsonl`). Order
      for next time: upload first, then push — the key is referenced the
      moment index.html deploys.
- [x] A5. FAA NASR entry on sources.html.
- [x] A6. France + Brazil (2026-09-16): `airspace_build.py` region registry
      (`--regions us,fr,br`), per-region caches, class-keyed paint rules,
      per-region cycle/status/pin card, sources.html entries for SIA (Licence
      Ouverte 2.0) and DECEA (AISWEB terms). Archive `class-20260916` (UTC build date)
      (figures in `worklists/data/airspace/builds.jsonl`).

## Next

- [ ] B1. Special-use airspace (MOA/R/P/W/A) + Mode C veils, current-only from
      the ADDS GeoJSON (`us_faa/adds/<date>/Special_Use_Airspace.geojson`,
      `Class_Airspace.geojson` MODE C rows) as a `sua` layer with a
      "current only" caveat in the panel. France (SIA `R`/`D`/`P`/`TRA`/`CBA`
      espaces, per cycle) and Brazil (`eac_r`/`eac_p`/`eac_d`) slot into the
      same layer.
- [~] B0. France after 2023-10: Ryan downloads each cycle from the SIA shop
      ("AIM Data" → "Données aéronautiques XML AIRAC mm/yy", 5.6 MB — not the
      eAIP); the pull files it (zip or the folder Safari expands) under
      `fr_sia/cycles/<date>/`. 09/26 + 10/26 filed and published 2026-09-16
      (`class-20260916b`: FR 36 cycles, 1,024 versions; 2023-11-02..2026-09-02
      stays a hole). Repeat every 28 days — the shop drops old cycles. Brazil
      accumulates a cycle per snapshot the pull script saves.
- [ ] B0b. Brazil's ICAO classes: not in the WFS; AIP Brasil ENR 2.1 lists
      them per TMA/CTR (a documents source). Until then Brazil is drawn as
      controlled airspace of unstated class.
- [ ] B2. SUA history from the per-cycle AIXM `SaaSubscriberFile.zip`
      (1,236 XML, GDAL's GML driver opens them; ~300 use ArcByCenterPoint —
      verify arc densification before trusting).
- [ ] B3. Labels: an anchor-point layer (polylabel) with floor/ceiling text,
      `CenteredTextSymbolizer`.
- [ ] B4. Airspace_Boundary set (TRSA, ADIZ, SFRA, ARTCC) — current-only.
- [ ] B5. Share-link state for the layer toggle (URI-POLICY rule 8: a new
      query parameter is permanent API — decide the name once).
