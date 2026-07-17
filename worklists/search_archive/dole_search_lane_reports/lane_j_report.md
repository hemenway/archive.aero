# LANE J REPORT — round 7: GIS geoportals / GeoBlacklight / OpenGeoMetadata / OldMapsOnline

Date: 2026-07-08. Read handoff + log Parts 1–7 first; nothing below re-searches closed ground.
Findings CSV: lane_j_findings.csv (7 rows, all LOW-RES thumbnails of dole-absent charts, all curl-verified 200).

## HEADLINE

1. **OpenGeoMetadata org: DEFINITIVELY CLOSED, zero US sectionals.** Shallow-cloned ALL 36
   institution repos (5.0 GB metadata: edu.stanford.purl, edu.harvard, edu.princeton.arks,
   edu.nyu, edu.utexas, edu.umn, edu.wisc, edu.virginia, edu.tufts, edu.columbia, edu.mit,
   edu.berkeley, edu.cornell, edu.upenn, edu.uchicago, edu.indiana, edu.illinois, edu.umich,
   edu.psu, edu.msu, edu.uiowa, edu.umd, edu.gmu, edu.purdue, edu.unl, edu.osu, edu.rutgers,
   edu.uwm, edu.northwestern, edu.unr, edu.uarizona, edu.lclark, geobtaa, gov.data,
   ca.frdr.geodisy, org.humdata) and grepped every file. "sectional aeronautical" appears in
   exactly ONE file org-wide: edu.cornell/00/16/01/fgdc.xml — a 1979 NY hydrography DLG that
   cites sectionals as National-Atlas lineage. "sectional chart" appears in exactly ONE file:
   gov.data CKAN record for FAA's CURRENT "Sectional VFR Charts" product. Broad "aeronautical"
   grep = 2,619 files, all triaged by title: NYU = foreign TPC/ONC/JOG/GNC sets (~1,780),
   Harvard = DCW/AMS-foreign/Global-GIS + "sectional center facility" ZIP-code layers (551),
   UT Austin = AMS Mexico 1:500k + GSGS Asia 1:1M (95), Wisconsin = the known WisDOT state
   chart run 1967–2025 (42, excluded class), gov.data = current FAA/NOAA products (110),
   the rest crumbs (Princeton air-route maps, Indiana Rand-McNally 1936 state air-trails map,
   Stanford upper-air pilot charts). GitHub code-search API materially under-reports
   (386 "sectional" hits vs full-clone truth) — clone-and-grep is the only reliable method.
2. **Harvard Geospatial Library (HGL): CLOSED without touching the bot wall.** hgl.harvard.edu
   is behind an AWS WAF challenge (HTTP 202, x-amzn-waf-action: challenge — curl and WebFetch
   both blocked). Bypass: the OpenGeoMetadata edu.harvard repo IS the full HGL4 export
   (22,074 records, committed 2024-01). Zero US sectionals (see above). HGL ≠ LibraryCloud
   (Part 6) — both now independently closed.
3. **THE LANE FIND (record/contact class + 7 LOW-RES rows): ASU Library Map & Geospatial Hub.**
   Discovered via ArcGIS Online sharing API (never swept in any round). AGO user `mapgeohub`
   (org 0OPQIK59PJJqLK0A = asu.maps.arcgis.com; hub geodata-asu.hub.arcgis.com; lib.asu.edu/geo)
   publishes its whole paper-map inventory as feature service
   `services3.arcgis.com/0OPQIK59PJJqLK0A/arcgis/rest/services/maps_master/FeatureServer/0`
   (53 fields incl. SCANNED, FILE_NAME, SCAN_DATE, SCANNER, THUMB_URL, FILE_URL, barcode,
   SuDoc). Queried all `%ERONAUTICAL%` rows: **22 US sectional scans** on a WideTEK48 at
   600 dpi (2022-10 .. 2023-10), of which SIX are editions the dole lacks entirely:
   - **Juneau 1971** (dole+NARA Juneau coverage starts at ed 34/1994 — located NOWHERE else)
   - **Cape Lisburne 1971** (dole AK starts 2011)
   - **Anchorage 1971** (NOAA has 1970 only)
   - **Albuquerque 1971**
   - **Hawaiian Is./Marianas/Samoan Is. combined 1971**
   - **Prescott 1964 front+back** (dole Prescott jumps 1961→1965)
   Rest are distinct scans of dole-held editions: Douglas 1935, 1964 f/b; Grand Canyon 1935,
   1938 (1930s sectional series, NOT the excluded modern GC VFR — their 2001 GC VFR was
   excluded); Phoenix 1941, 1955 f/b, 1957 f/b; Prescott 1932, 1948, 1950 f/b. Plus one
   UNSCANNED paper: Phoenix 1968-01-04 (dole holds 1968 via LOC; scan-request candidate).
   **BUT: only 800px JPEG thumbnails are public** (asu.maps.arcgis.com item /data, 220–360KB;
   FILE_URL blank on all chart rows; their other-map FILE_URLs are also web-res only).
   Masters are unpublished ⇒ CONTACT CLASS: ASU Library Map & Geospatial Hub (lib.asu.edu/geo)
   — precise ask = the 600dpi masters for FILE_NAMEs
   1971_500k_aeronautical_chart_{albuquerque,anchorage,juneau,cape_lisbume(sic),
   hawaiian_islands_marianas_islands_samoan_islands} and
   1964_sectional_aeronautical_chart_prescott_{front,back} (+ the 15 AZ-chart re-scans and a
   scan of the 1968-01-04 Phoenix). The 7 thumbnails of dole-absent editions are recorded in
   lane_j_findings.csv explicitly flagged LOW-RES 800px THUMBNAIL.
   ASU KEEP repository (keep.lib.asu.edu) 403-bot-walled; nothing indexed for these charts.

## Zero-yields (terse)

- maps.princeton.edu live API: `catalog.json?q=aeronautical` = 22 hits, all foreign
  Fliegerkarte/air-route/AMS; phrase "sectional aeronautical" = 0. Matches clone. CLOSED.
- geo.nyu.edu: curl/WebFetch 403 (bot-wall) — moot, edu.nyu clone fresh (2026-05), TPC/GNC
  only. CLOSED.
- geodata.lib.utexas.edu: connection-reset/307-loop to curl AND WebFetch (hard wall) — moot,
  edu.utexas clone fresh (2026-01), AMS foreign topo only. CLOSED.
- geodata.mit.edu: 410 Gone (service retired); edu.mit clone = DCW/foreign. CLOSED.
- geodata.wustl.edu: DNS dead (geoportal retired). CLOSED.
- CUGIR (cugir.library.cornell.edu/catalog.json): 4 "aeronautical" hits = trench map +
  cropland lineage. NY-state data only. CLOSED.
- MN Geospatial Commons (gisdata.mn.gov): ShieldSquare/perfdrive bot-wall on every path;
  state-clearinghouse class already ruled out in Part 6. CLOSED (bot-wall list).
- Johns Hopkins: no public GeoBlacklight exists; archive.data.jhu.edu (Dataverse) Cloudflare
  403. Implausible holder. CLOSED.
- Scholars GeoPortal (Ontario): 403. Implausible holder. CLOSED.
- ArcGIS Online full sweep beyond ASU: `"sectional aeronautical"` = 29 items (16 = ASU thumbs,
  rest FAA current-edition services/web maps, Merced County 2017 service, IN DNR 2023 web
  maps, 1 Norfolk/Minot classroom maps); `"sectional chart" type:Service` = 11 (CalOES
  archive = Dec-2021/Jul-2022 editions the dole holds; NC_Sectional_Charts 2019 tile cache =
  current-edition class; JSchneider/merybar91 = current). No historical rasters on AGO other
  than ASU's. CLOSED for historical content.
- OldMapsOnline: STILL WALLED — Cloudflare 403 on www./search./api. hosts for curl AND
  WebFetch (same as Part 4); wayback CDX of search.oldmapsonline.org/api/* and
  api.oldmapsonline.org/* = ZERO captures ever. Only route left is a real browser.
  (Expected value low: member catalogs with US sectionals — Rumsey, LOC, Harvard, NYPL —
  are all independently closed.)
- HathiTrust/Google-Books class: not re-touched (closed Part 6).

## Technique notes (exact endpoints that worked)

- OGM: `git clone --depth 1 https://github.com/OpenGeoMetadata/<repo>.git` × 36 ≈ 5 GB, then
  `grep -ril`. Org repo list: `gh api orgs/OpenGeoMetadata/repos?per_page=100`. Code-search
  API (`gh api "search/code?q=org:OpenGeoMetadata+<term>"`) works unauthenticated-scale but
  under-indexes big repos — use only for scouting.
- Harvard HGL4 record format: JSON that is DOUBLE-ENCODED (json.loads twice) GeoBlacklight 1.0.
- GeoBlacklight JSON API pattern confirmed: `<host>/catalog.json?q=...&per_page=N` (works on
  maps.princeton.edu, cugir); JSON:API variant nests values under data[].attributes with
  document_value sub-objects.
- ArcGIS Online search: `www.arcgis.com/sharing/rest/search?q=...&num=100&f=json` — supports
  `owner:`, `type:`, quoted phrases; org portals substitute host (asu.maps.arcgis.com). Item
  file: `<portal>/sharing/rest/content/items/<id>/data` (follow 302; HEAD lies about final
  size only rarely — GET with -L). Feature-layer query:
  `.../FeatureServer/0/query?where=UPPER(TITLE) LIKE '%25...%25'&outFields=*&f=json`.
  University "map hub" inventories on AGO are a REPEATABLE LEAD CLASS — ASU proves libraries
  publish full drawer inventories (incl. SCANNED flags + file names) as public feature
  services even when the scans themselves are unpublished. Other big map libraries with Esri
  hubs could be probed the same way in a future round via
  `search?q=owner:<user>` after finding candidate owners with map-collection dashboards.
- Bot-wall additions for the master list: hgl.harvard.edu (AWS WAF challenge/202),
  gisdata.mn.gov (ShieldSquare), keep.lib.asu.edu (403), archive.data.jhu.edu (CF),
  geodata.lib.utexas.edu (TCP reset / UA-sensitive), geo.nyu.edu (403).

## Record-only summary (for log Part 8 + contact list)

- ASU Library Map & Geospatial Hub: 600dpi unpublished masters of 22 sectionals (6 dole-absent
  editions incl. Juneau 1971 + Cape Lisburne 1971 + Anchorage 1971 + Albuquerque 1971 +
  Hawaiian/Marianas/Samoan 1971 + Prescott 1964 f/b) + unscanned Phoenix 1968-01-04 paper.
  Contact: lib.asu.edu/geo (Map & Geospatial Hub). This slots into the round-7 contact-class
  queue alongside GlidePlan/Welch/Fox.
