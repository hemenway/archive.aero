# LANE E REPORT — Usenet archives as a URL mine (round 5)

Verified findings: **2** (lane_e_findings.csv). Both are wayback-recovered GlidePlan
"Experimental" custom charts: full-resolution seamless JPG mosaics built from ~2006
NACO-era sectional rasters (the desert era), surfaced via a rec.aviation.soaring post
chain -> glideplan.com CDX. Same artifact class as the 23 aviationtoolbox
"georeferenced mosaic chunks (2005)" recorded as findings in round 2; note says so, so
the coordinator can drop them if mosaics are now out-of-policy.

## Mboxes downloaded & grepped (archive.org)
Item `usenet-rec.aviation` (giganews; coverage starts ~2003-07, essentially nothing
pre-2003 — date histogram of piloting: 50 msgs total before 2001):
  rec.aviation (1,530 msgs), .misc x2 (10,480 ea), .products (10,780), .ifr (43,567),
  .owning (67,331), .homebuilt (72,665), .student (80,814), .marketplace (5,200),
  .simulators (6,631), .soaring (134,923), .aerobatics, .ultralight, .rotorcraft,
  .restoration, .hang-gliding, .balloon, .air-traffic, .stories, .powerchutes.
Item `FULL-USENET-BACKUP-2020-Oct-rec.aviation.piloting...7z` (265,749 msgs, 2003-2020)
and `...rec.aviation.1596` (1,626 msgs). comp.infosystems.gis mboxes = 18 msgs of 2014
spam, zero. utzoo skipped per instructions (pre-1991).
Scan method: per-message filter (sectional | chart+scan/geotiff/tif/raster), extract all
URLs + context; second pass for "aeronautical chart"-only messages. ~8,600 URL hits,
~700 unique non-noise hosts triaged. NOTE: no rec.aviation binaries archives exist on
archive.org (alt.binaries.pictures.aviation never archived).

## Hosts CDX'd this round (none previously in log) — verdicts
- **glideplan.com** (449 urlkeys): POSITIVE — 2 findings above. Also: page0/files/
   37 per-city zips ~13-16MB (Nov 2016 captures) + downloads-2/-3 "1304/1307" contest
  zips = 2013/2016 FAA editions already in dole (derived) — skipped; GlidePlan
  "sectional_download.html" geolocated files never captured.
- **welch.econ.brown.edu** (Ivo Welch, "FAA Sectional and TAC Maps on my Website",
  rec.aviation.piloting 2006-01-05): hosted the complete Jan-2006 NACO set — 38
  half-sheet JPEGs 1.9-4.9MB (/aviation/sectionals/, eastern US: Atlanta 75,
  Brownsville 76, Charlotte 78, Chicago 71, Cincinnati 75, Detroit 71, Green Bay 70,
  Halifax 73, Houston 76, Jacksonville 76, Kansas City 75, Lake Huron 70, Memphis 75,
  Miami 77, Montreal 73, New Orleans 77, New York 72, St Louis 73, Washington 78)
  + uncompressed GeoTIFFs. Wayback crawled only dir listings + .htm/.tfw metadata
  (221 captures) — ZERO chart jpg/tif captured. Dir gone from his live site
  (ivo-welch.org/aviation = 404). RECORD-ONLY; top lead: email Ivo Welch (site invites
  contact) — he may still have the 2006 set + the NACO DVDs it came from.
- **alexisparkinn.com** (Jay Honeck "Old Iowa City Sectional Charts" gallery, found via
  rec.aviation.piloting 2003-12): 7 wayback-captured "Des Moines Sectional Aeronautical
  Chart" jpgs (Apr 1939, Mar 1945, Dec 1948, Dec 1953, Dec 1960, Nov 1964, Nov 1968;
  1.0-3.2MB, 20051028 captures) — downloaded all 7 and inspected: 1753px-wide scans of
  the Iowa City fold-panel only, NOT full charts. IOW_19xx files = smaller crops;
  "hiQuality" variants listed in dir but never captured. EXCLUDED (excerpts),
  record-only.
- **xcski.com/~ptomblin/mojave1945/** (Paul Tomblin hosting Larry Dighera scans; live
  site): "Mojave Desert" May 1945 chart in 8 gifs — index page states 1:1,000,000 AAF
  chart (files named wac1-8, 1276x1754 ea). EXCLUDED (WAC scale, not sectional).
  gallery.xcski.com "1955 CAR" thread = scanned Civil Air Regulations books, not charts.
- **cdw.homelinux.com:8087** (9MB stitched "Dallas-FtWorth72.jpg", 2004 post): zero
  wayback captures of the host. Dead end.
- **airchart.com** (Air Chart Systems / Howie Keefe — handoff lead #2; 904 urlkeys):
  storefront only; Hi-Res_Samples = 1 IFR-hi PDF + tiny update PDF; VFR atlas sample
  gifs 100-150KB page excerpts. No atlas scans. Zero-yield.
- **chartgeek.com/GE/** (refines round-3 verdict "infographics blog, wrong name":
  2007-2012 the domain was "ChartGEEK Charts for Google Earth", a DVD product + host of
  the gelib sectional overlay grid Sectionals_A_01..F_10 KMZs): only /GE/AK/AK.kmz
  captured (210KB = 12MB KML of L1-L9 gif TILE PYRAMID, tiles uncaptured) + banner.
  Payloads never archived; tiles excluded anyway. Zero-yield, closed.
- **gelib.com + gelib.fox-fam.com** (Google Earth Library "Aeronautical Charts for the
  US", 2007-2010): /maps/Sectionals/ = KML network-link stubs pointing at
  chartgeek.com/GE/US/*.kmz (above); fox-fam mirror = screenshots only. Zero-yield.
- **soaringdata.info** (Lynn Alley, rec.aviation.soaring 2010-2014): Feb/Apr-2010 CMR
  sectional sets (pre-FAA-free-era!) announced on usenet were NEVER captured; wayback
  holds only 2024-07 CMRs (~1.03MB each — the 1MB-truncation signature) + 2025 stubs;
  live site today serves only the current FAA edition (260514 tifs = in dole).
  Zero-yield; the 2010 CMR sets are lost unless Alley is contacted (secondary lead).
- **aviationtoolbox.org exact-URL probes** (new usenet evidence: raw_data listed
  San Antonio 73 N.tif, Hawaiian Islands 72.tif, munge/masked full-chart jpgs,
  munge/chunked.zip 1.9GB, frozen 2005-06 "current" dir w/ ~40 cities incl.
  Albuquerque 75, Atlanta 74, Charlotte 77): zero captures beyond the 3 Chicago tifs
  already in the online CSV + 2 TACs. Domain DNS is dead (SERVFAIL). Round-3 verdict
  stands; closed.
- **Zero-yield quick CDX**: thericcs.net/aviation (0 captures), bayareapilot.com (701,
  no chart files), memory-map.com (2,099, none), tknowlogy.com (718, none),
  glide2.glideplan.com (0), lairds.org/.us (Kyler Laird personal domains — 10,764
  urlkeys, chart-ish hits are camera photos), gpsinformation.net (Jack Yeazel — no
  sectional scans), www2.bitstream.net/~storius (gridded paper charts storefront, not
  probed further).

## usenetarchives.com — pre-2003 gap partially unlocked (technique notes)
Giganews/FULL-backup mboxes start 2003; Google Groups is bot-walled. usenetarchives.com
(Jozef Jarosciak) holds 1990s rec.aviation. Access: POST index.php with
`age_verify=confirmed&dob_*` once -> cookie `ua_age_verified=1`; then POST
`api/search.php` `search_type=find_posts&search_term=...&per_page=100` works fine from
curl (headers/subjects/mids/dates incl. 1993-2002). **`search_type=get_posts` (bodies)
always returns `[]` from curl** — tried decoded/encoded mids, groups, referer, XRW
headers; likely needs real browser/JS. threads.php/view.php are client-rendered.
narkive group search endpoints 404 from curl; only Google-indexed pages reachable.
Era threads surfaced (subjects only, bodies unread — future-round targets):
  1993-02/03 "gif's or jpg's of sectional or TCA charts" (rec.aviation.misc/piloting)
  1998-03-31 "Sectionals online?" (rec.aviation.misc)
  1998-08-21 "Scanned sectionals?" (rec.aviation.misc+sci.geo.satellite-nav; reply by
    Jack Yeazel — his gpsinformation.net has no chart scans, checked)
  2000-08-04 "All the sectionals online for Simmers" (rec.aviation.simulators, deja
    mid <8mhkvf$rtn$1@nnrp1.deja.com>) — a full-set host in 2000, unknown which
  2000-08 "Free Online Sectional Charts and TripTicks!" = AeroPlanner launch (host
    already closed in log Part 4)
  2000-12 "online sectional charts?"; 2002-09 "Digitized sectionals"

## Zero-yield searches (this lane)
WebSearch: site:groups.google.com rec.aviation sectional scan (soaringdata/skyvector
only); rec.aviation "sectional" "scanned" ftp 1998-2000 (FAA/LOC noise);
geocities/mindspring/earthlink phrasings (ISP-history noise); "All the sectionals
online" simmers (sofa spam); site:narkive.com variants (nothing new).
archive.org: identifier usenet-rec.aviation* (1 item), collection:usenethistorical
rec.aviation (0), collection:usenet aviation (1 irrelevant), alt.binaries.pictures*
(0), A2K/"archive 2000" (0).
Mbox mining dead ends: memory-map "47MB tiffs" = NACO DVD talk; avsim SFO sectional
(login-walled, per log); toutle.com volcano jpgs / thericcs DKK.jpg / rstengineering
Oshkosh wallpaper / tknowlogy AdizFlightPath.png / aopa notam jpgs = excerpts;
62.141.38.86 screenshots; naviter.si + cumulus-soaring + flywithce/wingsandwheels CMR
talk = current-era derived; vfrchart.ahost4free.com = UK charts; 66.226.83.248 airport
db; hang-gliding "matches" = day-trading spam.

## Leads for future rounds (ranked)
1. **Email Ivo Welch** (ivo.welch@anderson.ucla.edu / site contact) for his Jan-2006
   NACO JPEG+GeoTIFF set (38+ half-sheets, eastern US) — files confirmed to have
   existed, wayback missed the binaries.
2. **usenetarchives.com bodies via real browser** (get_posts works in-browser only) —
   read the 1998-2002 threads above for host URLs; the Aug-2000 simulators full-set
   announcement is the most promising unread post.
3. **Contact Lynn Alley (soaringdata.info)** for the Feb/Apr-2010 CMR sectional sets
   (pre-free-era editions absent from dole).
4. Paul Tomblin (xcski.com, still live/responsive) and Jay Honeck hold original scans
   (Dighera's + Iowa City gallery "hiQuality" versions) — personal-contact class.
