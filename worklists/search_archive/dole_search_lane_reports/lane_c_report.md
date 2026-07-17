# LANE C report — Google Earth Community / KMZ ground-overlay archives (round 5)

## POSITIVE: usahas.com live ChartGeek sectional superoverlays (24 findings rows)
The one live survivor of the whole 2006-2010 GE-overlay chart ecosystem.
- usahas.com (US Avian Hazard Advisory System, USAF bird-strike site run by DeTect Inc.,
  IIS 10 server, still up) licensed ChartGeek's Google Earth sectional set and still serves
  it: index `http://www.usahas.com/downloads/Sectionals.kml` -> 23 per-chart KMLs under
  `/GE/ChartGeek/Sectionals/<City>.kml`, tiles under `/GE/chartgeek/gifs/<Chart>_kmz_L{l}_{r}_{c}.gif`.
- Each chart is an inline (no NetworkLink) 5-level GIF superoverlay (Hawaii 4-level),
  1024x1024 tiles, assembling the FULL collar-clipped chart (both halves merged, legend
  stripped) at ~12-13k x ~8k px, i.e. ~60-70% linear of the 300dpi source. Deep-tile spot
  checks legible (contours, obstacle elevations readable). All 24 URLs verified `curl -sIL`
  200; sample deep tile per chart HEAD 200; tile Last-Modified 2009-07-17..2009-08-25 →
  NACO editions current mid-2009 = squarely in the dole's pre-2011 desert (log Part 4 had
  declared this era "effectively unrecoverable" beyond 3 exceptions).
- 23 charts: Albuquerque, Brownsville, Charlotte, Cheyenne, Chicago, Denver, Detroit,
  El Paso, Green Bay, Hawaiian Islands, Kansas City, Klamath Falls, Lake Huron, Memphis,
  New Orleans, New York, Omaha, Phoenix, Salt Lake City, San Antonio, Seattle, Washington,
  Wichita. NOT present (probed 404): Atlanta, Billings, Cincinnati, Dallas-Ft Worth,
  Great Falls, Houston, Jacksonville, Las Vegas, Los Angeles, Miami, San Francisco,
  St Louis, Twin Cities, all Alaska, TACs.
- Caveats for coordinator: technically a tile pyramid (lane rule 4 accepts tiles that
  assemble a full chart; the general exclusion list bans pyramids — flagged in every note);
  collar clipped so per-chart edition numbers are not visible on the image; grab soon,
  it is a 16-year-old leftover directory on a .mil-adjacent contractor site.
  Download ≈ 100 tiles/chart, ~2.3k tiles total, few hundred MB.

## Provenance chain mapped (Forkboy2 / Google Earth Library / ChartGeek)
- GEC thread 64566 (2005-08-07, Forkboy2 = Matt Fox = "topomatt"): "Aeronautical Sectional
  Charts of the United States" — per-chart SuperOverlays of ALL US sectionals+TACs incl.
  Alaska/Hawaii, ~40,000 tiles / 1.5 GB, from NACO digital rasters.
- 2007 host gelib.fox-fam.com / fox-fam.com: page `aeronautical-charts-us.htm` (captured
  2007-05-04) lists per-chart OFFLINE ZIPS `fox-fam.com/maps/Sectionals/_Offline/<City>.zip`
  for the ENTIRE set incl. every Alaska chart (Anchorage, Bethel, Cape Lisburne, Cold Bay,
  Dawson, Dutch Harbor, Fairbanks, Juneau, Ketchikan, Kodiak, McGrath, Nome, Point Barrow,
  Seward, Whitehorse) + Halifax/Montreal + 30 TAC zips. **None of the zips or tiles were
  ever crawled** (CDX `fox-fam.com/maps/Sectionals/*` = empty; only an unrelated
  Census.kmz proves the crawler could fetch fox-fam binaries). fox-fam.com now domain-parked;
  its 2021/2022 "kml" captures are squatter junk despite kml mimetype in CDX.
- 2007-11 → 2012 host www.gelib.com (+ tile host www.chartgeek.com): gelib pages captured;
  root KMLs captured (`/maps/Sectionals/Aero_Charts_nl.kml`, `Sectionals.kml` → 50
  chartgeek.com/GE/US/Sectionals_[A-F]_[01-13].kmz grid-cell roots after the 8/2008
  mosaic merge; `TACs.kml`, `Flyways.kml`). chartgeek.com/GE/* has exactly 2 captures:
  AK.kmz (2009-02-13, 210KB = KML tree referencing 10,107 AK GIF tiles — tiles 404/never
  archived) and a banner gif. gelib.com dead (parked), chartgeek.com is an unrelated
  infographics blog since ~2011 (log Part 3 verdict was right for the wrong era — the
  2005-2010 chart era left almost nothing in wayback anyway).
- chartgeek.com was a 99c/chart + ~$90 DVD storefront; free 64MB San Francisco demo at
  `chartgeek.com/052ck2wcoid82/ChartGEEK_San_Francisco.exe|.zip` (password x45d38vi790) —
  CDX empty, never archived. ChartGeek DVDs are a physical-media lead (eBay/abandonware).

## Record-only (no recoverable download)
- gelib/chartgeek US sectional mosaic superoverlay 2007-2012 (above) — roots archived,
  tiles gone.
- chartgeek AK.kmz wayback capture `web.archive.org/web/20090213212611id_/http://www.chartgeek.com/GE/AK/AK.kmz`
  — KML only, no imagery.
- nlneilson.com (Neil Neilson) WorldWind FAA sectional set, 2008: cache packs
  L012/L01234/L5(.dds).zip (up to 1.25GB, level-5 ≈ 68m/px near-chart-res mosaic),
  Alaska/Hawaii/TAC zips, tile service `nlneilson.com/serv/FAA/...` — site 410 Gone;
  wayback holds only sectionals.html + 2 layer .java files; zips/tiles never crawled.
- gelib.com/maps/phoenix-1948/ — "Phoenix Arizona Aeronautical Chart 1948 (USAF)" 85-tile
  quadtree + KML; only the 2019 dir LISTING archived, zero tiles; source stated as NOAA
  Historical Map & Chart Project (already fully audited in Part 3), so no new source lost.
- x-plane.org file 38646 "1940s WAC and Sectional Charts - West" (Bob Denny, updated
  Sep 2025): PDF scans + ForeFlight mbtiles of 1940s western sectionals used by his
  "LF Range Experience" add-on. Site is Cloudflare-walled to curl/WebFetch and downloads
  need a free login; zero wayback captures of the file page. MANUAL-BROWSER LEAD for next
  round — could contain 1940s editions/scans not in LOC/NARA sets.

## Zero-yield (searched, nothing to recover)
- bbs.keyhole.com attachments: `ubb/download.php` full CDX (5,000+ urlkeys) = empty-body
  302s/404s; only 8 real captures, all 400-950-byte placemark KMZs. Attachment store
  `ubb/placemarks/*` (4,819 captured files, 4,008 kmz) grepped for
  sectional/aero/chart/faa/vfr/tac/airspace/aviat → only 64566-Aero_Charts_nl.kml (449B
  network link). download.php 302 Location mapping (`?Number=` → `/ubb/placemarks/<file>`)
  documented for future use. No /ubb/attachments dir, no files.keyhole.com.
- gearthhacks.com: full dlfile CDX inventory (27 pages, 27,967 urlkeys) grepped —
  only 3 chart items: dlfile28261 (Forkboy2 → gelib link, no file), dlfile9843 dead by
  2007, dlfile8876 404-only. Domain-wide sectional/aeronautical/naco filter re-checked;
  modern WordPress relaunch hosts old KMZs at wp-content/uploads/downloads/ but the
  aeronautical entry's live page is 404 and the 2020 capture has no local file. GEH forums
  thread 13497 capture = discussion only.
- productforums.google.com domain CDX filter original:.*sectional.* → 0 captures.
- Flight-sim/GIS forums: simviation search endpoints 404/no chart-scan files;
  fsdeveloper = tooling threads only; flightsim.com + forums.x-plane.org 403 to
  curl/WebFetch (x-plane downloads login-walled); airliners.net "Old Sectional Charts"
  thread (2017 capture) links only airfields-freeman + myairplane.com (both long closed
  in log); tollbit paywall on live airliners.net.
- Misc overlay libraries from gelib blogroll: earthview.nl, sgrillo.net, xbbster
  superoverlay (tool only), usahas covered above; aero.sors.fr = navaid/airport KMZ only.
- archive.org: forkboy2/gelib/"google earth library"/chartgeek → 0 relevant items.
- Web searches for "index of" sectional kmz, vintage-year sectional KMZ phrasings,
  Aero_Charts/chartgeek/gelib mirrors → only usahas (captured above) and known sites.
- backcountrypilot.org "Old sectional charts" thread → NOAA historicalcharts only (known).

## Leads for future rounds
1. x-plane.org file 38646 (Bob Denny 1940s sectional PDFs/mbtiles) — needs manual browser
   + free forum login.
2. ChartGeek DVD (~2009, ~$90, all sectionals+TACs as GE overlays) and MFox/gelib offline
   zip set — physical/abandonware hunt (also fits round-5 lead #1 CD-ROM track).
3. usahas.com download of all 23 pyramids before the site is modernized (trivial scripted
   fetch, ~2.3k tiles).
4. Matt Fox (Forkboy2/topomatt, gelib.com owner) and Neil Neilson (nlneilson.com,
   neil@nlneilson.com) — both plausibly still have the 2007-2009 source rasters/tile sets;
   direct-contact candidates.
