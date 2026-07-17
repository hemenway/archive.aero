# HANDOFF — Dole coverage hunt, round 8+ (written 2026-07-08, end of round 7)

Supersedes `dole_search_handoff_1970-2010.md` (the round-5 handoff; still valid for era
context, but this file reflects rounds 5–7). Mission: expand the dole (the archive.aero
chart catalog) with downloadable FULL-RESOLUTION scans of US **sectional aeronautical
charts** (1:500,000). The 1970–2010 window remains the big desert (LOC ends early-70s,
FAA digital starts 2011), but any-era new sectionals count.

**READ FIRST:** `dole_search_log.txt` Parts 1–8 — the complete record of every source
searched across 7 rounds, including dead ends, bot-walls, and zero-yield queries.
Re-searching ANYTHING in it is wasted work. Per-lane detail (recovered file inventories,
barcodes, exact-URL lists for contact emails) is in `dole_search_lane_reports/`.

## Files (repo root: /Users/ryanhemenway/archive.aero)
- `master_dole_v2.csv` — the catalog, 7,247 rows; loader `dole_v2.py`
- `missing_from_dole_online.csv` — 1,490 verified online findings (source,name,url,date_or_edition,note)
- `dole_search_log.txt` — Parts 1–8; the authority on what is closed
- `dole_search_lane_reports/` — round 5–7 lane reports + `glideplan_macmaps_inventory.txt`
  + `asu_thumbnails_record_only.csv`
- `missing_from_dole.csv` + `missing_from_dole/` — local-disk audit (separate task)

## STATE OF THE HUNT (be honest with yourself before searching more)
Seven rounds have exhausted the public-internet search frontier: national archives (NARA
fully enumerated — 1.45M cartographic keys), LOC (both sectional sets probed number-by-
number), all US library aggregators, a complete numeric CONTENTdm census (1,310 instances),
OpenGeoMetadata (all 36 repos cloned+grepped), the whole DOI system (DataCite corpus
negative), Trove/Gallica/DigitalNZ/Canadiana, wayback CDX of ~100 hosts, Common Crawl +
arquivo.pt resurrection attempts, Usenet mboxes AND pre-2003 usenetarchives bodies,
abandonware/disc databases, FTP indexers (ecosystem dead), map dealers, and runet hoards.
Round 6 netted 1 row; round 7 netted 46 only because of one bot-wall bypass (raremaps GCS).
**The remaining recovery paths are mostly ACTIONS, not searches** — see the ranked list.

## RANKED ACTION LEADS (the actual round-8 work)

### A. Download-phase (URGENT, scriptable today, no human gatekeeper)
1. **usahas.com ChartGeek pyramids** — 23 full mid-2009 NACO-edition charts as GIF
   superoverlays on a live USAF-contractor leftover dir (~2.3k tiles, few hundred MB).
   Index: `http://www.usahas.com/downloads/Sectionals.kml`; tiles
   `/GE/chartgeek/gifs/<Chart>_kmz_L{l}_{r}_{c}.gif`. 16 years stale — could vanish any day.
2. **simviation.com realcharts set** — 42 zips, complete US sectional set as ca-2010
   full-chart PDFs. 2-step cookie download (`/1/download-file?...` for session cookie, then
   `/1/download?...`), server ignores Range, ~80KB/s — budget hours. URLs in the CSV.
3. The rest of `missing_from_dole_online.csv` (1,490 rows) is download-ready generally;
   wayback rows: use the largest capture, `id_` form.

### B. Contact class (files confirmed to exist; owners findable; write emails)
1. **ASU Library Map & Geospatial Hub** (lib.asu.edu/geo) — holds 600dpi 2022-23 scans of
   22 sectionals, SIX editions the dole lacks entirely incl. **Juneau 1971** (nowhere else
   on earth), Cape Lisburne/Anchorage/Albuquerque/Hawaiian-Marianas-Samoan 1971, Prescott
   1964 f+b. Only 800px thumbnails public. Exact FILE_NAMEs/SuDoc/barcodes: lane_j_report.md.
2. **GlidePlan (company alive, contact-form live)** — their dead free tree had 117
   city+edition seamless mosaics, eds 75–89 (2008–2011). Precise ask inventory:
   lane_g_report.md + glideplan_macmaps_inventory.txt.
3. **Ivo Welch** (UCLA Anderson; site invites contact) — complete Jan-2006 NACO eastern-US
   set: 38 half-sheet JPEGs + GeoTIFFs (+ maybe the source NACO DVDs). Filenames in
   lane_g_report.md.
4. **Matt Fox** (Forkboy2/topomatt; gelib.com owner) — 2005–09 full US+AK sectional overlay
   source rasters; the 53-zip fox-fam Offline set incl. all 15 Alaska charts.
5. **ChartGeek author** — 50-city free exe set (2012) + the ~$90 DVD (all sectionals+TACs).
6. Neil Neilson (nlneilson, WorldWind packs), Lynn Alley (soaringdata.info Feb/Apr-2010 CMR
   sets), Jay Honeck (alexisparkinn "hiQuality" Iowa City Des Moines scans 1939–68),
   Paul Tomblin (xcski.com, Dighera scans), Sean Morrissey (full-size 2010 masters?).

### C. Scan requests (institutions with confirmed undigitized in-era paper)
1. **NLA Australia Bib 1030946** — near-complete ~1970s CONUS+AK+HI National Ocean Survey
   set: THE DESERT ERA IN ONE PLACE; NLA does digitisation-on-demand. Also Bib 7796679
   (C&GS/USAF 1940–66).
2. Harvard Gray Herbarium Orlando (O-8) Jan-1943 Restricted (HOLLIS 990147545500203941).
3. Fort Collins History Connection Cheyenne base-1955/rev-1959 (cdm17204, acc 1979.065.0001).
4. Archives West/OAC personal papers (Spokane 1965, SF+Seattle 1941–45, Boston 1945,
   Detroit 1936, El Paso/San Antonio/Del Rio, Pangborn SE-US set) — list in log Part 6.
5. ASU unscanned Phoenix 1968-01-04 paper sheet (piggyback on the B.1 contact).

### D. Free-login manual tasks (a human with a browser + free account)
1. **AVSIM library**: Matt Fox 2005 "Sectional_Charts_of_the_US_Pack_1..11" (Pack 11 = 11
   Alaska charts; Point Barrow/Seward/W-Aleutian/Whitehorse exist NOWHERE else for
   1970–2010) + David Myers Mar-2010 Hawaii/Guam/Samoa (Aug-2009 eds) + 250k-inserts pack.
   Download IDs in log Part 6 / lane_d_report.md.
2. **pilotsofamerica.com** thread 141250 attachment 113812 — full scanned 1943 Cleveland PDF.
3. **x-plane.org** file 38646 (Bob Denny) — 1940s WESTERN sectional PDF scans + mbtiles.
4. Browser-only residuals (low prior): OldMapsOnline (fully walled), IU Virtual Disk
   Library full enumeration, Arizona Memory Prescott R-3 1955, LOC webarchive, archive.today.

### E. Physical media (used market; no rips exist anywhere online — proven)
MapTech "VFR/IFR AeroPack" regional CDs (~1999–2006, BSB rasters); ChartGeek DVD (~2009);
RMS Technology "VISTA" sectional CD-ROMs (1997+, the product behind 1998's "scanned
sectionals of the whole USA"); Air Chart Systems VFR Sectional Atlases (~1962–2013; one
atlas = half-US coverage at print res; ASINs B003K2XBJC, B004HUERDK); NACO subscription
raster CDs (eds ~60–85). PC Pilot cover CDs not on IA (issues 22–81, 84+) may hide
one-offs like the Issue-82 Ketchikan.

### F. Remaining genuine SEARCH frontiers (thin; only if A–E are done)
- Custom-domain-only CONTENTdm instances (escaped the numeric census — no enumeration
  method found; per-city dorks are the only probe).
- raremaps.com NEW listings over time (GCS pattern is durable; re-run the CDX slug sweep).
- usenetarchives.com bulk pre-2003 sweep (bodies now scriptable via base64 unlock) — the
  targeted reads all led to known hosts; diminishing returns expected.
- Periodic re-checks: FAA next-edition dirs, new archive.org uploads (uploader saved-
  searches for "sectional"), Wikimedia new uploads, eBay→dealer-scan pipeline (raremaps
  adds inventory continuously).

## RULES & CONVENTIONS (unchanged; match rounds 1–7 exactly)
- Sectionals only (1:500k series, incl. USAF editions; versos and half-sheets COUNT;
  distinct scans of dole-held editions COUNT — say so in the note). EXCLUDE: TAC, WAC,
  ONC/JOG, planning, helicopter, Grand Canyon, Caribbean VFR, IFR/enroute, airway strips,
  obstruction charts, state aeronautical charts (any scale), crops/excerpts/thumbnails.
- POLICY (settled round 5–7): tile pyramids/mosaics that assemble a COMPLETE chart count
  as findings with the artifact class stated in the note (usahas ChartGeek, GlidePlan
  mosaics, raremaps DZI). Sub-~1000px thumbnails are record-only (ASU precedent) even if
  the edition is otherwise absent.
- Verify every URL: `curl -sIL` → 200 + plausible size. Wayback: `web.archive.org/web/
  <TS>id_/<url>`, pick the largest capture from full CDX history; ~1MB content-length on a
  big file = truncated capture, reject. Common Crawl pre-2013: ARC format, truncates at
  512KB with NO flag (captured < Content-Length is the tell).
- Dedupe against BOTH `master_dole_v2.csv` (download_link + filename; dole flattens URLs:
  `/` and spaces → `_`) and `missing_from_dole_online.csv`.
- Findings: append (source,name,url,date_or_edition,note) rows; validate with python csv
  (5 cols, no HTML entities, no dup URLs). Record-only items go in the log, not the CSV.
- Log: append the next Part to `dole_search_log.txt` — positives AND zero-yields.
- Python: `~/venv/bin/python` (3.14), NOT ./.venv.
- Parallel fan-out subagents with strict lanes + a coordinator merge/re-verify pass has
  been the working pattern for rounds 5–7 (merge/verify scripts are trivial to rewrite:
  dedupe on URL + wayback-original + basename-vs-dole-flattened-filename, then parallel
  curl HEAD with ranged-GET fallback).

## TECHNIQUE CHEAT-SHEET (hard-won; details in log + lane reports)
- Wayback CDX: `web.archive.org/cdx/search/cdx?url=HOST/*&output=json&collapse=urlkey`
  (+filter/limit). Gotchas: unencoded regex filters silently return empty (use
  matchType=domain + --data-urlencode); big domains 504 (fall back to subdir prefixes);
  bursts get 503s (cool down 30–60s; retry-until loop).
- archive.org: advancedsearch bare `q=` is now FULL-TEXT-only — use explicit `title:()`/
  `creator:()` fields. On-the-fly archive listing: `archive.org/download/<item>/<file.iso>/`
  (trailing slash) lists/extracts inside ISOs/zips. Uploader enumeration via
  `uploader:"..."` works.
- **raremaps GCS bypass**: `storage.googleapis.com/raremaps/img/dzi/img_<id>.dzi` (+
  `_files/<level>/<c>_<r>.jpg` tiles, `_1.dzi` verso, `img/xlarge/<id>.jpg` fallback).
- **usenetarchives.com**: age-gate POST once for cookie; `api/search.php` find_posts is
  open; get_posts wants search_term = base64(message-id-with-angle-brackets) with `=`
  padding stripped, plus search_group.
- **Trove keyless API**: apikey = md5("Wonder"+cookie[x-ctx]), strip leading zeros;
  `/api/search/-1?terms=...&sortBy=relevance` (sortBy mandatory).
- CONTENTdm JSON API (unauthenticated on most instances):
  `/digital/api/search/searchterm/<terms>/field/all/mode/all/maxRecords/50`; IIIF at
  `/iiif/2/<coll>:<id>/info.json`. Numeric census method: probe cdm15000–18299.
- NLA: `nla.gov.au/nla.obj-<id>/image` serves 5000px openly; `/dzi` gives native size;
  tile service is Anubis-walled.
- Harvard LibraryCloud: `api.lib.harvard.edu/v2/items.json?q=` open, unquoted keywords.
- DataCite class-closure re-check: `api.datacite.org/dois?query="sectional aeronautical"`.
- NARA (closed, but for reference): S3 `NARAprodstorage` public listing; catalog proxy
  search unquoted-q only; `lz/cartographic/reference/` needs direct S3 (Beaumont masters).
- WordPress originals: strip `-scaled` from upload filenames (curtiswrightmaps).
- Simviation 2-step cookie dance: see B/download notes above or CSV row notes.
- DEAD/GONE (don't plan around): Memento TimeTravel aggregator, FTP search engines
  (filesearch.ru/filemare/searchftps/filepursuit), Bibliotheca Alexandrina mirror,
  aviationtoolbox.org DNS, gelib/fox-fam/chartgeek-GE-era content.
- BOT-WALLS (curl+WebFetch blocked; browser-only): NYPL, Stanford SearchWorks, Florida
  Memory, DLG Georgia, Arizona Memory, WorldCat, dp.la, TAMU, OldMapsOnline, HathiTrust
  catalog, vetusware, myabandonware, IU VDL (Turnstile), curiosity.lib.harvard.edu, Yale,
  Cornell digital, Oregon Digital, x-plane.org, flightsim.com, airliners.net,
  shortwingpipers.org, Free Library Philadelphia, Dig DC, Louisiana Digital Library,
  kchistory.org (JS), loadmap.net (JWT), LOC webarchive, archive.today (429).

## ERA CONTEXT (why 1970–2010 matters; unchanged from round-5 handoff)
LOC set gct00089 (the dole's backbone) tapers off ~1974 (ends ca004091). FAA digital
coverage starts 2011. In-era holdings recovered so far: WASP rows 1971–82, the complete
2004 archive.org edition set, NARA Juneau eds 34–56 (1994–2016), usahas ChartGeek mid-2009
set (23 charts), simviation ca-2010 set (complete US), GlidePlan/aviationtoolbox mosaics
(2003–06), scattered one-offs (SF 66/78, Seattle 2001 training chart, ESRI SF 2008,
LA 1976 @ raremaps, NLA Whiting NY 1957). For most cities, ~1975–1999 is still ZERO — the
ASU/NLA/GlidePlan/AVSIM leads above are the only known ways in.
