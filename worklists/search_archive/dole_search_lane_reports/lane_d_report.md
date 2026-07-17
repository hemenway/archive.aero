# LANE D REPORT — round 5 (personal pilot sites/blogs, GeoCities graveyards, pilot forums, open dirs)
Date: 2026-07-08

## POSITIVE: simviation.com "realcharts/realvfrcharts" series by Sean Morrissey (May 2011 uploads, ca. 2010 EDITIONS)
THE LANE FIND. Simviation.com (open download, no login) hosts a complete US sectional set in
PDF form: 38 single-chart zips (realcharts_sec<City>.zip, CONUS + Hawaiian Islands + the
US-series border charts Halifax/Montreal/Lake Huron) + 4 Alaska multi-chart packs
(realvfrcharts_alaska_pt1-4). Per the in-zip readme: "half-size copies of raster scans of
real FAA aeronautical charts, most dating from the first half of 2010", sourced from
aeronav.faa.gov chartlist_sect as it stood in 2010 — i.e. PRE-2011 editions, squarely in the
dole's desert (dole FAA coverage starts 2011; e.g. dole's earliest SF of that era is
2011-08-25). SF sample downloaded + opened: 21MB single-PDF full chart, page images 5112px
long side (multi-page, front+back panels), plus HowTo/readme. CONUS zips ~24-31MB each;
Alaska packs pt1=86.3MB, pt2=86.7MB, pt3=102MB, pt4=24.9MB (multi-chart bundles covering
the 16-chart AK series across the 4 parts; pt1 downloaded in full = Anchorage, Cape
Lisburne, Dawson, McGrath as separate 19-23MB full-chart PDFs dated Aug 2010 + HowTo;
pt2-4 contents not enumerated — server too slow — but same series per uploader). All 42
zips verified HTTP 200 with full content-length via ranged GET (see lane_d_findings.csv).
DOWNLOAD MECHANICS (2-step, note for download phase): GET
https://simviation.com/1/download-file?file=<zip>&fileId=<id> to obtain a `simviation`
session cookie, then GET https://simviation.com/1/download?file=<zip>&fileId=<id> with that
cookie -> 302 to dl.simviation.com/download/?file=...&sessionId=... -> 200. Direct hit
without cookie returns 302 + "x-strngr: not downloading". Server is slow (~1-2 MB/10s);
use long timeouts. Zips contain a full-chart PDF + readme/HowTo folder.
CAVEAT: "half-size" lossy JPEG-in-PDF derivative of the FAA 2010 rasters, not archival
scans — but the only recovered form of the 2010 editions found anywhere so far.
Per-chart edition stamps vary ("most dating from the first half of 2010") — download phase
should read each chart's edition/date panel. Dole-overlap caveat: dole holds Dawson ed 44
(2010-10-21, wayback FAA zip) — the pack Dawson may be an earlier 2010 edition or a
distinct derivative of the same; everything else 2010 is absent from the dole (dole's only
other 2009-10 row is LOC Cheyenne ca000719r dated 2009-07-30).
Dedupe: zero 'simviation|realcharts|realvfrcharts' matches in master_dole_v2.csv and
missing_from_dole_online.csv.

## RECORD-ONLY / LEADS (no verifiable public URL)
- AVSIM library (login-walled, per log): Matt Fox series —
  (a) Jan 2004 "High_Resolution_<City>_Sectional_Chart.zip" one-per-city ~8-10MB series
  (likely the same content as the archive.org "sectional-charts" 2004 item already recorded);
  (b) Feb-Mar 2005 "Sectional_Charts_of_the_United_States_Pack_1..12.zip" (20-42MB each),
  reprojected for FSM Moving Map — Pack 11 = 11 Alaska sectionals (Anchorage, Bethel, Cold
  Bay, Dutch Harbor, Juneau, Ketchikan, Kodiak, McGrath, Seward, W Aleutian, Whitehorse),
  Pack 12 = Cape Lisburne/Dawson/Fairbanks/Nome/Point Barrow/Hawaiian Is. (Pack 12 is already
  on local disk at ~/Downloads/us_sectionals_pack_12; jpgs are 5020x6745px, ca. 2004-05
  editions — the local audit already lists them). Packs 1-11 not found on any open mirror
  (simviation/flyaway/surclaro/web searches negative; author's readme says AVSIM-only).
  download IDs: /download/62084 (P8), 62131 (P9), 62132 (P10), 62313 (P11), 62554 (P12).
  NOTE: the recorded archive.org "sectional-charts" 2004 item already covers most of the
  same AK names (Anchorage..Nome halves + Hawaiian Is.) but NOT Point Barrow, Seward,
  Western Aleutian or Whitehorse — Matt Fox packs 11/12 (2005) and the simviation 2010
  alaska packs are the only located sources for those four in the 1970-2010 window.
- AVSIM: David Myers Mar 2010 "Hawaii Sectional, Inset & Extras" (27.71MB, /download/145983,
  incl. Guam/Saipan/American Samoa sectionals from Aug 2009 editions) + "250k Inserts" pack
  (/download/145373; insets are half-sheet versos - counts). His per-city "Chart Chunks ...
  Moving Map" series (5,000+ files) is chunked = excluded class.
- pilotsofamerica.com thread 141250 "80 Years-old Cleveland Sectional chart": attachment
  "1943-cleveland-sectional-map-pdf" (attachment id 113812) = full scanned 1943 Cleveland
  sectional PDF; attachment download is login-walled (curl 403), zero wayback captures.
  Would be a distinct scan of a 1943 Cleveland edition (dole has WASP Cleveland 1943-11-11).
- flyawaysimulation.com file 18855 "Yet Another Moving Map V2.0" (10.78MB, open site but
  download POST is JS/session-gated): contains Seattle_Sectional_front_l.jpg (3.48MB) +
  Los_Angeles.jpg (2.52MB) textures, edition unknown ~2008-10, marginal derived textures.

## ZERO-YIELD (this lane, round 5)
- Search-engine passes (~18 query variants): "sectional chart 1978 scan(ned) jpg/tif",
  "scanned my old sectional", "expired/outdated sectional scanned", city+year phrasings
  (LA/Chicago/Denver/Anchorage/Fairbanks 1975-1998), "1978_sectional"-style filename dorks,
  blogspot/wordpress pilot-blog phrasings, smugmug/flickr/photobucket, filetype:tif dorks,
  "index of"/"parent directory" sectional dorks — all converge on the known set
  (LOC gct00089, NOAA, airfields-freeman excerpts, vfrmap/skyvector, storefronts). Nothing new.
- GeoCities graveyards: "sectional/aeronautical chart" on oocities.org,
  geocities.restorativland.org, geocities.ws, angelfire.com, tripod.com — only pilot-training
  text pages and airfields-freeman mirrors (excerpts, excluded). Zero chart scans.
- Forums: backcountrypilot.org 14537 ("Old sectional charts. Do you have them?") — scanning
  offers, Dropbox talk, NO hosted collection ever posted; backcountrypilot 7117 (high-res
  images = FAA aeronav links); POA 80299/153413/38700/37352 (all point to LOC/NOAA/FAA);
  cessna170.org 15565 (LOC only); supercub.org t-53434 (apps talk, no scans);
  rec.aviation.piloting "outdated sectional charts" Google Groups thread (physical reuse
  only); PPRuNe 48553 (history talk, no US sectional scans). X-Plane.org: XPCharts etc are
  nav-DB apps, no scanned-chart packs found.
- Bot-walls hit this round: airliners.net thread 355863 (tollbit/402 + curl 202 challenge;
  no wayback capture), shortwingpipers.org 13955 "Old sectional charts now online" (403 all
  UAs, zero wayback captures — title suggests it's about the 2020 LOC release),
  homebuiltairplanes.com 52931 (WebFetch 403; curl OK — links are LOC gct00498 only),
  avsim.com/forums (403), flightsim.com (Cloudflare 403 + login-wall library).
- Wayback CDX new hosts this round: ranainside.com (FSM Moving Map home; 200 urlkeys, only
  site graphics, no map packs); pilotsofamerica.com/community/attachments/* (no captures of
  the Cleveland PDF); airliners.net thread (no captures).
- simviation.com other keywords (aeronautical, high resolution chart, hawaii sectional,
  myers, chart chunks) + full FS Navigation category browse (70 files): only the realcharts
  series + HSC9MS1.zip (59K PDF + 325K jpg = excerpt, excluded) + project_hawaii.zip
  (scenery, not charts); everything else is airfield-locator gauges.
- surclaro.com: dead (Plesk default page); CDX has zero sectional URLs.
- calclassic.com propliner charts page: worldwide AIP/approach-chart links only;
  dc3airways.net vfrcharts tutorial: Nantucket excerpt PDF only.
- vansairforce.net / eaaforums.org / cessna-pilots searches: chart-format chatter only,
  no posted full scans. StuckMic: nothing indexed.
- SF PDF structure check: 5 pages, main chart images 5112px long side (passes >=4000px bar,
  but confirms "half-size" derivative status).

## LEADS FOR FUTURE ROUNDS
- AVSIM login wall is the only thing between the dole and: Matt Fox 2005 Packs 1-11
  (ca. 2004-05 editions incl. full Alaska) and David Myers Aug-2009-edition Hawaii/Guam/
  Samoa packs. A free AVSIM account would unlock all of these (manual task).
- POA account (free) unlocks the 1943 Cleveland sectional PDF attachment (thread 141250).
- flightsim.com library (login) may mirror the same/more packs — unverifiable from curl.
- Sean Morrissey (rottydaddy@msn.com) stated the sectional PDFs are simviation-exclusive;
  his tutorial packs (REALVFRCHARTS_TUTORIAL_PT01-05) are teaching PDFs, not charts.
