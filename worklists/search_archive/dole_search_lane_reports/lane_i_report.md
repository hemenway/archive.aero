# LANE I REPORT — Dole coverage hunt, round 6

Focus: full-res sectional scans ~1970–2010 missing from the dole.
Four assigned angles: (1) FTP/open-dir search engines, (2) NACO-internal filename dorks,
(3) magazine cover discs + sim-compilation CDs, (4) Bob Denny "LF Range" mirrors.
Plus cheap follow-ups: 4shared/mediafire, rutracker record-only intel.

RESULT: **0 verified findings.** lane_i_findings.csv = header only.
This lane is a clean zero-yield that closes four ecosystems and adds Common Crawl as a
newly-exercised (but negative) archival source. Details below in terse log style.

================================================================
ANGLE 1 — FTP / OPEN-DIRECTORY SEARCH ENGINES — ECOSYSTEM DEAD/NEGATIVE
================================================================
The classic FTP-search services are all defunct in 2026:
  filesearch.ru / mmnt.net / mmnt.ru   DNS resolves (46.138.252.159 / 89.108.111.48); TCP
                                       port 80 CONNECTS but server sends no HTTP response
                                       (curl exit 56, recv failure) on / and on int/get?st=
                                       and cgi-bin/s?q= endpoints, browser UA or not. Dead.
  filemare.com                         HTTP 410 Gone (retired). /en path times out.
  searchftps.net (Napalm FTP Indexer)  Domain-registration-EXPIRED parking page
                                       ("Domain registration has expired"); no search UI.
  searchftps.org / globalfilesearch.com  302 redirect to parking; dead.
  filepursuit.com / www.filepursuit.com  DNS dead (no A record).
  filesearching.com, ftplike.com       connection fails.
Modern open-directory search engines (live) — queried, negative for sectionals:
  ODCrawler (odcrawler.xyz)            Elasticsearch backend reachable unauthenticated at
                                       search.odcrawler.xyz/elastic/links/_search (POST JSON;
                                       apiKey in app JS is not needed for _search). Queried
                                       sectional / aeronautical / "sectional chart" / vfr+chart
                                       / faa+chart / kap+chart / geotiff / seclosa / AeroPack /
                                       chartgeek. ZERO aeronautical charts. All "sectional"
                                       hits = AMS math "sectional meeting" docs (ftp.rush.edu),
                                       cross-sectional med studies, furniture. "aeronautical"
                                       hits = radio-comms textbook PDFs, an Army FM on
                                       "locate coordinates on aeronautical charts" (training
                                       manual, not a chart) at 72.21.17.51:15588. Only chart-ish
                                       file in whole index: bpp.umd.edu FAA Northeast Chart
                                       Supplement PDF (not a sectional).
  bpp.umd.edu open dir (h5ai, live)    /archives/TO_REVIEW/Library/Aviation and Airspace/ =
                                       AIM, Part 101, airspace KMZs, one Chart Supplement PDF.
                                       No sectionals.
  eyedex.org 301, opendirsearch.abifog.com / filesearch.link 200 but no usable chart index.
Fresh "index of" / "parent directory" dorks (WebSearch, multiple phrasings incl.
  seclosa/secsea/secden variants, north.tif/south.tif): converge only on FAA/LOC official
  pages. No open-dir chart mirror surfaced.

================================================================
ANGLE 2 — NACO-INTERNAL FILENAME DORKS (seclosa / secsea98 / secphx / secden / secnyc / secchi)
================================================================
  WebSearch                  only FAA product pages + the govdocs1/Tika copy of the NACO
                             "Sample Sectional Raster" catalog page. No hosting dir.
  archive.org full-text      seclosa / secsea / "sectional raster naco" = 0 hits.
  Common Crawl URL index     seclosa etc. appear ONLY as the naco.faa.gov / avn.faa.gov
                             catalog HTML page URL (index.asp?xml=...Sectional_Raster /
                             ...Raster_Sectional_Sample/Raster), NOT as .tif files.
  discmaster (1.7B files)    seclosa / secsea98 / secphx / secnyc = 0 results; secden = 2
                             false hits (German CHIP-magazine SECDEN.CD hypertext stub);
                             secchi = oceanography (Secchi disk). fileType=image "sectional"
                             = 100 hits, ALL furniture .dxf / HL2 model .vvd / html5-schema
                             .rnc / a .wav — zero aeronautical. Confirms round-5 corpus-wide
                             negative for NACO chart discs.
  govdocs1 corpus (NEW)      Downloaded the 299MB dump.sql (digitalcorpora S3) mapping all
                             ~1M docids -> original .gov source URLs. docid 262626
                             ("...Raster_Sectional_Sample/Raster") is a 34KB HTML catalog
                             page, NOT the tif; 262620 ("...Sectional_Raster") = 17KB HTML.
                             ZERO .tif rows referencing sectional/raster/naco/aeronav/faa in
                             the entire corpus. 241 naco/avn/aeronav URLs = all d-tpp approach
                             plates, IFR acifp, ProductDetails/Catalog HTML, sc_/se_/nc_ etc.
                             PDF *documents* (chart-users specs, not chart images). Dead end.
                             (The famous "Sample Sectional Raster Aeronautical Chart.tif" that
                             appears in Tika govdocs1 metadata is a filename mentioned inside
                             that HTML catalog page's text — no such binary is in the corpus.)

================================================================
ANGLE 3 — MAGAZINE COVER DISCS + SIM-COMPILATION CDs (archive.org)
================================================================
  computer-pilots-edition-volume-1  ISO (201MB) inspected via IA on-the-fly view: FS5.1
    (software item)                 Junker-52 panels/aircraft, ComPilot MTB40 runtime, .pcx/
                                    .ico bitmaps — NO chart tif/sid/kap. maps/*.tif present
                                    are FS airport diagram bitmaps (lowi/lowk/lowl = ICAO
                                     region prefixes), not sectionals.
  Computer Pilot Magazine items     October 2005 + Vol7/9/10 issues = PDF magazine SCANS only
                                    (60-124MB jp2/pdf). No cover-CD ISO items exist on IA.
  micro-wings-scenery-dallas-ft.-worth  3.4MB FS scenery zip. MicroWINGS Magazine v4/v5 =
                                    PDF scans. No charts.
  fs-tool-kit (135MB ISO),          FS5.1/6.0 shovelware compilations — sceneries/adventures/
  Flight_Sim_199x...compilation     aircraft. grep of ISO listings for sect/chart/.tif/.sid/
    (339MB ISO)                     .kap = nothing sectional.
  IA software mediatype sweeps      title:sectional -> only "Visible Human Sectional Anatomy";
                                    title:aeronautical -> NOAA 1988 Data Sampler (already
                                    known text-only) + demoscene noise; title:vfr -> UK VFR
                                    photo-scenery DVDs (England/Berlin) + FRFPVRFO trainer.
                                    "pc aviator" / "cd pilot" / megascenery = AU sceneries.
  Guessed FAA AOD_VFRCharting_<city>.zip (SEA/DEN/NY/CHI/DFW/ATL/MIA/PHX/SFO) on live
    aeronav.faa.gov/content/aeronav/chart_sample_files/ = all 404. Only LA exists (already
    in the dole per round 5). Dir re-listed: no new sectional zips beyond the 10 sampleVFR_*
    separation tifs already recorded.
  Conclusion: beyond the round-5 PC Pilot Issue 82 find, no magazine cover disc or FS
  compilation on IA carries full-chart rasters. Angle closed.

================================================================
ANGLE 4 — BOB DENNY "LF RANGE EXPERIENCE" 1940s WEST SECTIONALS
================================================================
  x-plane.org file 38646 "1940s WAC and Sectional Charts - West" (updated Sep 2025) +
    file 35825 "LF Range Experience" — both Cloudflare + free-login walled; Wayback CDX for
    both file pages = EMPTY (0 captures); Common Crawl = none. Charts offered as individual
    PDFs + georeferenced mbtiles for ForeFlight, but only from behind the x-plane.org wall.
  Bob Denny's personal hosting HUNTED and does NOT hold the charts: he is the DC-3 Dreams /
    ACP observatory-automation software author. dc3.com / acp.dc3.com / acpx.dc3.com /
    bobdenny.com / forums.dc3.com are 100% astronomy/telescope. Wayback CDX of dc3.com /
    acp.dc3.com / bobdenny.com filtered for chart|sectional|mbtiles|1940|foreflight|kmz|
    aviation|xplane = ZERO. No personal mirror of the source scans exists.
  Aggregator rehosts (x-plained / threshold / simtakeoff): WebSearch found none linking the
    file direct — all point back to the x-plane.org file pages.
  Status: RECORD-ONLY (unchanged from round 5). Also note these are 1940s West sectionals,
  outside the round-6 core 1970-2010 window.

================================================================
CHEAP FOLLOW-UPS
================================================================
  4shared public search (search.4shared.com / www.4shared.com/web/q?query=sectional chart):
    endpoint works (200), but "sectional" matches only Sectional-sofa .dwg/.doc, garage .jpg,
    "sectional views exercise.docx" — no chart zips. Download links login-gated regardless.
  mediafire: WebSearch site: dork returns only 4shared/mediafire category shells, no files.
  rutracker.org: WebSearch surfaced no NACO/Jeppesen chart-CD torrent threads (only FAA
    official pages). No verifiable download; torrents excluded by rule anyway. Record-only.

================================================================
NEW SOURCE EXERCISED THIS ROUND (technique note for round 7)
================================================================
  Common Crawl old crawls (CC-MAIN-2008-2009, 2009-2010, 2012, 2013-20/48) via
    index.commoncrawl.org/<coll>-index?url=HOST/*&output=json — a DIFFERENT capture set
    from Wayback. Checked the round-5 "contact-class" hosts: fox-fam.com (CC-2012 captured
    only root + /Other/ — the /maps/Sectionals/ zips NEVER crawled, matching Wayback empty);
    welch.econ.brown.edu (no /aviation/ dir captured in any CC crawl); aviationtoolbox.org,
    naco/avn.faa.gov, chartgeek, gelib, nlneilson, alexisparkinn — CC binary inventory =
    only d-tpp approach PDFs, law-case PDFs, Iowa-airport-history PDFs, .kmz airspace, FS
    sceneries. No sectional raster in any CC old crawl. CC does NOT beat Wayback for these
    hosts, but it is a valid independent second-look and was previously untried.
  usahas.com ChartGeek pyramids (round-5 positive): still live; Wayback also holds the
    usahas /ge/kml/fullareas/ airspace KMZs (GE_AIR/COUNTY/STATES etc.) — those are airspace
    overlays, NOT charts, so correctly excluded.

================================================================
LEADS FOR ROUND 7 (ranked)
================================================================
  1. The x-plane.org login wall remains the ONLY gate on real content this lane could not
     pass: file 38646 (Bob Denny 1940s West sectionals + mbtiles) — needs a manual free-login
     browser session, same class as the round-5 AVSIM/POA tasks. Not on any mirror or in CC.
  2. FTP-search angle is permanently exhausted — every indexer (filesearch.ru, filemare,
     searchftps, filepursuit) is dead in 2026. Do NOT re-run angle 1. ODCrawler is the only
     live open-dir engine and it is chart-negative; re-query only if its index grows.
  3. NACO internal filenames (seclosa etc.) are indexed NOWHERE public (web, IA, CC,
     discmaster all 0). The raster-CD-era filenames never leaked online. Close this angle.
  4. Magazine/CD angle exhausted on archive.org; any further yield needs PHYSICAL disc rips
     (MapTech AeroPack, ChartGeek DVD, NACO subscription CDs) per round-5 lead #5.
