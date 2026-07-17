# LANE G report — round 6: resurrect known-lost binaries via non-Wayback archives

Mission: query Memento TimeTravel, arquivo.pt, Common Crawl (CC), and archive.today for the
exact URLs of high-value sectional-chart binaries that the Wayback Machine never captured
(targets a-f from the round-6 brief).

## Bottom line

**Zero verified full-payload chart binaries recovered → lane_g_findings.csv is header-only.**
Every alternative archive either (1) never crawled the target binaries, (2) is dead, or
(3) is bot-walled. HOWEVER the lane produced substantial NEW record-only intel — most
importantly a previously-unknown GlidePlan free-download tree (39 win_maps exes + 39
Mac_maps zips per snapshot, three edition generations 2008/2010/2012, eds 75-89) whose
Feb-2012 Mac zips ARE in Common Crawl but truncated at 512 KB, plus exact filename lists
for every target set (fox-fam 53 zips, welch 38 jpgs, chartgeek 50 free city exes,
alexisparkinn 7 hiQuality jpgs, nlneilson 18 zips).

## Archive-infrastructure status (important for future rounds)

- **Memento TimeTravel aggregator is DEAD.** `timetravel.mementoweb.org` no longer resolves
  in DNS (checked 2026-07-08). Do not plan future lanes around it.
- **Bibliotheca Alexandrina mirror** (`web.archive.bibalex.org`): DNS resolves
  (196.204.180.101) but TCP connections hang/fail — server down. (Moot anyway: BA is a
  mirror of IA's 1996-2007 collection, not an independent crawler, so IA-negative ⇒ BA-negative.)
- **LOC web archive** (`webarchive.loc.gov`): Cloudflare bot-wall for curl AND WebFetch
  (403 "Just a moment..."). Manual-browser candidate only; prior is very low (curated
  collections, none of our hobbyist hosts).
- **archive.today** (`archive.ph`): persistent 429 rate-wall from this IP even after
  cooldown; and it archives *pages*, not binaries — low prior. Manual-browser candidate.
- **Stanford SWAP** (`swap.stanford.edu`): JS-shell search page; CDX endpoints 403.
  Curated Stanford/CA collections — prior ~0 for our hosts.
- **UKWA**: wayback syntax rejects non-UK URL queries (400 redirect); UK-domain scoped. Skipped.
- **arquivo.pt**: fully working CDX API (`arquivo.pt/wayback/cdx?url=<host>&matchType=domain
  &output=json`). Swept all target hosts — results below.
- **Common Crawl**: fully working. 125 indexes; earliest CC-MAIN-2008-2009. Pre-2013 crawls
  are **ARC format** (single-line header `url ip ts mime len\n` + HTTP headers), not WARC —
  parse accordingly. **Payload truncation is at 512 KB (524,288 B), not ~1 MB**, with NO
  truncation flag in ARC records — detect by captured-bytes < HTTP Content-Length.
  Fetch recipe: `https://data.commoncrawl.org/<filename>` with
  `Range: bytes=<offset>-<offset+length-1>`, gunzip the member, split headers.

## Per-target results

### a. chartgeek.com Google Earth set 2008-2012 — ZERO in all alt archives
- CC domain queries (2008-2009, 2009-2010, 2012, 2013-20/48, 5 mid-2014-16 indexes):
  HTML/store pages only. **No /GE/ URL was ever crawled by CC** (KMZ links live inside KML
  network links, which crawlers don't parse). GE/US/Sectionals_*.kmz, AK tiles, US.kml: absent.
- arquivo.pt: chartgeek.com captures are 2017+ only (domain became an infographics blog
  ~Aug 2012→). Zero GE-era content.
- Wayback AK.kmz (20090213212611) re-parsed: 10,107 tile hrefs are RELATIVE —
  `AK__kml_L<1-9+>_<x>_<y>.gif` under `http://www.chartgeek.com/GE/AK/`. None of these
  tile URLs exist in CC or arquivo.
- **NEW (record-only): free Windows per-city chart installers.** CC-MAIN-2012 captured
  `http://www.chartgeek.com/downloads/WindowsCharts.htm` (Feb-2012, page recovered from
  WARC, 11.7 KB): "FREE WINDOWS Versions" — 87 exes = 50 sectionals (incl. all Alaska:
  Anchorage, Bethel, Cape_Lisburne, Cold_Bay, Dawson, Dutch_Harbor, Fairbanks, Juneau,
  Ketchikan, Kodiak, McGrath, Point_Barrow, Seward, Western_Aleutian_Islands, Whitehorse)
  + 37 TAC exes, at `http://www.chartgeek.com/downloads/ChartGEEK_<City>.exe`.
  ~2011-12 editions. NONE of the exes captured by CC/IA/arquivo. (IA CDX re-checked for
  /downloads/: page known, binaries never fetched.)
- Demo `052ck2wcoid82/ChartGEEK_San_Francisco.exe|.zip`: nowhere.

### b. fox-fam.com Offline 2007 set — ZERO; exact filename list recovered
- Exact URL list recovered from wayback capture `20070504024207id_` of
  `www.gelib.fox-fam.com/aeronautical-charts-us.htm` (NOT fox-fam.com/maps/... as the brief
  said): **53 sectional zips** at `http://www.fox-fam.com/maps/Sectionals/_Offline/<City>.zip`
  — 38 CONUS/HI/Canada-border + all 15 Alaska (Anchorage, Bethel, Cape%20Lisburne, Cold
  [Bay], Dawson, Dutch%20Harbor, Fairbanks, Juneau, Ketchikan, Kodiak, McGrath, Nome,
  Point%20Barrow, Seward, Whitehorse) + SectionalCharts.kml. (Saved: scratchpad/laneg/gelib_aero.htm)
- CC: fox-fam.com has ZERO captures in 2008-2009/2009-2010; 2012+ = gelib.fox-fam.com
  404-era stubs. The Offline zips predate CC's first crawl.
- arquivo.pt: 2 captures, both 2022 domain-squatter junk.
- Note: 2021-2025 IA "200" captures under fox-fam.com/gefiles/ are squatter JS-challenge
  pages (incl. the 2022 `SectionalCharts.kml` capture — fake).

### c. welch.econ.brown.edu/aviation/sectionals — ZERO; exact filenames recovered
- Exact list from IA dir-listing capture 20060901075736: **38 half-sheet JPEGs**
  `<City>-<ed>-North|South.jpg` — Atlanta-75, Brownsville-76, Charlotte-78, Chicago-71,
  Cincinnati-75, Detroit-71, Green-Bay-70, Halifax-73, Houston-76, Jacksonville-76,
  Kansas-City-75, Lake-Huron-70, Memphis-75, Miami-77, Montreal-73, New-Orleans-77,
  New-York-72, St-Louis-73, Washington-78 (Jan-2006 NACO). GeoTIFF twins under
  `/aviation/uncompressed/Sectional-Area-Charts/<City> <ed> North|South.tif`.
- CC 2008-2009 (161 urls) + 2009-2010 (178) + 2012 (2): **no /aviation/ path at all** —
  the dir was deleted before CC's first crawl (2008). arquivo: 3 x 2011 redirects. Dead end;
  only route remains contacting Ivo Welch (round-6 lead #2).

### d. nlneilson.com WorldWind packs — ZERO binaries; full pack manifest recovered from CC
- CC-2009-2010 captured and I recovered complete: `sectionals.html` (Sep-2008 version),
  `download/FAA Sectionals/FAA.xml` (41,483 B complete — WWC LayerSet; tile ServerUrl =
  `http://nlneilson.com/serv/FAA`), `Sec_48.java`, `T_AK.java`.
- Full zip manifest with byte sizes (from page): L012.zip 17,140,585; L01234.zip 264,122,292;
  L5.zip 827,813,859; Alaska.zip 202,983,975; Hawaii.zip 4,252,742; 48St_WWC.zip; AK_WWC.zip;
  dds twins (L5dds.zip 1,254,502,550; Alaska_dds 304,452,758; ...); FAAserv set
  (48States_serv_4/5, Alaska_serv, Hawaii_serv). All at
  `www.nlneilson.com/download/FAA%20Sectionals/` and `/download/FAAserv/`.
- No zip and no /serv/FAA tile was ever crawled by CC (17+15 records = html/xml/java/pdf
  only). arquivo: zero captures of the domain. Contact class (Neil Neilson, round-5 lead).

### e. soaringdata.info Feb/Apr-2010 CMR sets — ZERO
- CC: 1 record total (2012 homepage). arquivo: zero. Live site checked: current-edition
  `.cmr` files named `<City> 260514.cmr` under `/aviation/sectionals/CMR Format/`
  (so 2010 files would be `<City> 100xxx.cmr/zip`); parent dirs 403, no old files exposed.

### f. Misc targets
- **ftp.maptech.com**: zero in CC (all 10 indexes) and arquivo. Closed for archives.
- **glideplan.com — THE LANE FIND (record-only).** CC-MAIN-2008-2009/2009-2010/2012
  captured `download/sectional_download.html` in Nov-2008, Sep-2010, Feb-2012 versions
  (all three recovered; saved under scratchpad/laneg/warc/glideplan/). Previously unknown
  free full-chart downloads, NOT visible in IA (IA has only the page-less 404 of 2013 and
  the two known Experimental JPGs):
  - `download/win_maps/<City>_<ed>.exe` — 39 cities, self-extracting geolocated seamless
    sectional mosaics; edition sets: 2008 page eds 75-83 (e.g. Seattle_75, Chicago_76,
    New York_77, Denver_78, SF_80, Charlotte_83), 2010 page eds 77-86, 2012 page eds 80-89.
    **117 distinct city+edition exes spanning 2008-2011 NACO editions.** None captured by
    any archive (CC crawled pages only; IA CDX: single 404).
  - `download/Mac_maps/<City>.zip` — same charts for Mac, unversioned names. **CC-MAIN-2012
    fetched ~39 of them (Feb-2012) but ALL >512 KB are truncated at 524,282-524,298 B.**
    Record headers prove content: e.g. Chicago.zip Content-Length 15,363,543,
    Last-Modified 2011-04-14, contains `Chicago/Chicago 81 North.jpg` (+South) — i.e.
    early-2011 editions: Billings 81, Cheyenne 83, Chicago 81, Cincinnati 85, DFW 86,
    Great Falls 80, Houston 87, Jacksonville 87, Kansas City 85, Klamath Falls 84,
    Lake Huron 81, Las Vegas 85, Montreal 84, New York 82, Omaha 83, SLC 85,
    San Antonio 86, ... (full inventory: scratchpad/laneg/glideplan_macmaps_inventory.txt).
  - `download/Mac_custom_maps/` (May-2010, "1005"): Albuquerque N-S (CL 10,247,631),
    Phoenix N-S (12,840,508), San Antonio N-S (11,507,424) full-chart May-2010 JPGs —
    truncated too. `Oahu_HI 1005.zip` IS complete in CC (115,316 B) but is an Oahu-area
    crop → excluded class.
  - glideplan.com is still LIVE (RapidWeaver rebuild; old /download/ tree 404s; company
    reachable) → strong CONTACT-CLASS lead: Lynn Alley/GlidePlan produced seamless
    full-chart mosaics for eds ~75-89 (~2005-2011) of 39 charts, plus the 2010 CMR sets.
- **alexisparkinn.com "hiQuality" Des Moines jpgs — exact URLs recovered** from CC-captured
  `iowa_city_sectional_charts.htm` (2008): 7 full-chart scans at
  `photogallery/Old%20Iowa%20City%20Sectional%20Charts/Des Moines Sectional Aeronautical
  Chart <date>.jpg` for April 1939, March 22 1945, December 8 1948, December 11 1953,
  December 1 1960, November 12 1964, November 14 1968. Binaries in NO archive (CC crawled
  page+PDFs only — 1 jpg record in 794; arquivo 2009 crawl fetched only videos).
  Pre-1970 era, but exact URLs enable a targeted future probe/contact.
- **gearthhacks dlfile9843/dlfile8876**: CC prefix queries zero across all early indexes;
  arquivo gearthhacks = modern 2019+ site (dlfile tree gone). Closed.
- **pilotsofamerica attachments/113812**: CC prefix zero; arquivo domain captures contain
  no /attachments/. Closed for archives (login task remains).
- **gelib.com (bonus domain-wide sweep)**: CC holds the sectional-mosaic root chain —
  `maps/Sectionals/Aero_Charts.kml` (complete, recovered; links to
  gelib.com/maps/Sectionals/Sectionals.kml + TACs.kml + Flyways.kml + chartgeek.com/GE/AK/AK.kmz)
  and `Aero-Charts-GEPlugin.html`. The child `Sectionals.kml` and ALL tiles: never crawled
  by CC or arquivo. Confirms the only surviving tile hosting is usahas.com (round-5 find).

## Zero-yield summary per archive
- arquivo.pt CDX (domain match): chartgeek(2017+ blog only), fox-fam(2 squatter),
  welch(3 redirects), nlneilson(0), soaringdata(0), ftp.maptech(0), glideplan(0),
  alexisparkinn(36: videos/photos, no charts), aviationtoolbox(robots.txt),
  gearthhacks(2019+ site), usahas(no /GE/), gelib(12: unrelated KMLs), pilotsofamerica(no attachments).
- arquivo.pt full-text: "sectional aeronautical chart" → known sources only
  (airfields-freeman, davidrumsey, navfltsm).
- Common Crawl (10-15 indexes/host incl. all pre-2013): binaries absent everywhere except
  glideplan Mac_maps/Mac_custom_maps (truncated) as above; aviationtoolbox = scripts only
  (no tifs, matches round-5 IA verdict); usahas = BAM webapp only, no /GE/ChartGeek tiles.
- Memento TimeTravel: service dead (DNS). Bibliotheca Alexandrina: down.
  LOC: Cloudflare-walled (curl + WebFetch). archive.today: 429-walled. Stanford SWAP: 403 CDX.

## Recovered artifacts kept in scratchpad/laneg/
- gelib_aero.htm (fox-fam 53-zip page), welch_sectionals_dir.html (38 jpg names),
  AK.kmz + ak_tile_urls.txt (10,107 tile URLs), warc/glideplan/* (3 page generations +
  editions.html), glideplan_macmaps_inventory.txt (per-zip CL/LM/inner-names),
  warc/chartgeek/WindowsCharts.htm.payload (87-exe page), warc/nlneilson/* (FAA.xml etc.),
  warc/gelib/Aero_Charts.kml.payload, warc/alexis/iowa_city_sectional_charts.htm.payload,
  cc/*.json (all CDX result sets), arquivo/*.json.

## Suggested follow-ups (for coordinator; NOT recorded as findings)
1. CONTACT GlidePlan (live company site glideplan.com): ask for the retired win_maps/
   Mac_maps masters (eds 75-89, 2008-2011) and 2010 CMR sets — now a precisely
   documented ask (filenames, sizes, dates from CC record headers).
2. CONTACT DeTect/usahas re: the 13 CONUS + AK ChartGeek city KMLs that 404 (they clearly
   licensed the full set in 2009), and Matt Fox re: fox-fam 53-zip Offline set (exact list now known).
3. Manual-browser probes (low prior): webarchive.loc.gov and archive.ph for
   glideplan win_maps exe URLs and chartgeek downloads exes.
4. The 7 alexisparkinn Des Moines hiQuality URLs → contact Alexis Park Inn / Jay Honeck
   (already a round-5 contact-class name) with exact filenames.
