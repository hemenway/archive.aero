# Worklist 03 — Online sources already searched (do not duplicate effort)

**Reviewed 2026-09-21 against local records only.** This is the search-history
index, not a fresh verification of external availability. The last recorded broad
search round here is **2026-07-23**. Contacts, login-only leads and scheduled
rechecks have no later completion evidence in this record. No outreach or new
online search was performed during this cleanup.

The findings CSV still has **1,517 rows**. The full queue was attempted and
analyzed in July; use [07](07_download_queue.md) for its deduplicated acquisition
state, not the original 1,309-row download count. Local source availability now
needs reconciliation separately ([02](02_disk_vs_dole.md)).

*Generated 2026-07-16, condensed from `search_archive/dole_search_log.txt` Parts 1–8,
`dole_search_handoff_round8.md`, and all 13 lane reports. Those remain the authoritative
record — this is the quick-reference index. Rounds 1–7 ran 2026-07-07 → 07-08.*

**Findings CSV:** `search_archive/missing_from_dole_online.csv` — 1,517 rows
(`source,name,url,date_or_edition,note`; +27 from round 9, 2026-07-23). The
**original 1,490-row** era breakdown was: 87×1930s, 118×1940s, 19×1950s, 20×1960s,
**176×1970–2010 (the desert)**, 1,060×2011+, 10 undated.

> **Round-9 addendum (2026-07-23 — full detail: search log Part 9).** +27 findings:
> 23 GPO microfiche findings (mostly 1981–1991, plus Jacksonville 1951) from
> archive.org's NEW GPO-depository microfiche ingest (Readex microfilm, 400dpi panels;
> active channel — **recheck monthly**), 4 raremaps (Washington 1945, Douglas Restricted
> 1944 +verso, LA R-2 NEW STOCK +verso, Fairbanks ca.1941-42). Verdict changes: FAA
> `chart_sample_files` DEAD, faacharts.faa.gov DEAD (the suggested ~2026-08-13 recheck is overdue/unrecorded here);
> raremaps re-sweep via **live sitemap** (not CDX — sitemap is not CF-walled);
> university-ArcGIS frontier EXH (ASU remains unique); usenetarchives bulk sweep EXH
> (closed, diminishing returns confirmed); usahas EXH (missing tiles deleted wholesale —
> the 23 downloaded pyramids are all that exists); oldimprints DEAD. New walled leads:
> backcountrypilot thread 14537, shortwingpipers t-13955. New technique: XenForo sitemaps
> public (full POA forum sweep without login). Rumsey 9911.000 is mislabeled in LUNA —
> the scan is Roswell (Q-4), not Phoenix; search Rumsey by media ID.

**Verdict key (as of the recorded search):** EXH = searched with no further
known yield (avoid repeating the same sweep without a new lead) · EXH+ = exhausted, findings recovered ·
LOGIN = free login needed (manual) · CONTACT = files exist, only the owner has them ·
WALL = bot-wall, browser-only residual · DEAD = host/service gone · RECHECK = periodic re-check worthwhile.

## 1. Historical source-search table (July 2026)

### US federal institutions

| Venue | What was searched | Verdict | Log ref |
|---|---|---|---|
| Library of Congress (loc.gov) | Both sectional sets: gct00089 (dole backbone, ends ~1974) probed number-by-number (17 gaps; 3 versos, 8 dead numbers); gct00498 "Base & Duplicates" ca000001–086 → 72 new tifs. All-maps searches, finding aids, 1990 sampler CD (text-only) | **EXH+** (75 rows) | Parts 2, 5 |
| NARA (catalog + S3) | Full S3 enumeration: rg-237 (18,183 keys) AND `lz/cartographic/` (**1,451,105 keys**); catalog fully paginated (711 + 5,686 hits); series 305976 fully digitized → 78 findings (Juneau eds 34–56, 1940s TX, Tulsa 1950, Seattle 1945); RG 342 USAF = 0 digitized. Bonus: uncataloged Beaumont TIF masters at `lz/cartographic/reference/Batch0001/` | **EXH+** (81) — "definitively closed" | Parts 2, 4, 5 |
| NOAA historicalcharts | All 64 "Aeronautical Chart" records; 19 true sectionals (1932–35 CAA, 1950 batch, Anchorage 1970×2) | **EXH+** (19) | Part 3 |
| NOAA other (IR, libraries, NCEI) | Text pubs only; OneStop dead | **EXH** | Parts 3, 8 |
| FAA live (aeronav.faa.gov) | `chart_sample_files/`: AOD LA zip (317MB 2400dpi plates) + 10 sample tifs; next-edition dir 09-03-2026 still 404 | **EXH+** / **RECHECK** | Parts 5, 7 |
| GovInfo / ROSA-P / FDLP-GPO | Text only; GPO never deposited NACO raster CDs | **EXH** | Parts 4, 6 |
| USGS, Smithsonian, DPLA, HathiTrust | Topo/record-only/zero | **EXH** | Part 3 |

### Universities, public libraries, state hubs, museums

| Venue | What was searched | Verdict | Log ref |
|---|---|---|---|
| UNT Portal to Texas History | **Full OAI sweep, 2,412,477 records**: 13 sectionals (2 UTA + 11 WASP = dole's WASP rows) | **EXH+** | Parts 3, 5 |
| Gateway to Oklahoma History | Full OAI sweep, 1,262,175 records: zero | **EXH** | Part 5 |
| OK State CONTENTdm | Tulsa 1945 + OKC 1945 IIIF | **EXH+** (2) | Part 4 |
| UWM AGS Library | 3 Seattle 1940–41 printing proofs | **EXH+** (3) | Part 3 |
| U. Alabama cartweb | Full 50-state sweep (~29,950 MrSID items) | **EXH+** (19) | Part 5 |
| Leventhal / Digital Commonwealth | Boston ed 47 1961 + Dallas ed 53 1961 (gap fills) + SF RS-1 1943 | **EXH+** (5) | Part 4 |
| Ohio Memory | Cleveland Jul-1942 gap fill | **EXH+** (1) | Part 4 |
| Harvard (LibraryCloud + HGL) | One chart: Orlando O-8 1943, physical only → scan request | **EXH** | Lanes F, J |
| ~30 FDLP universities + ~22 state digital hubs + TX regionals + AK/HI/Pacific | Dorks/APIs where open; zero | **EXH** | Lane F, Parts 4–6 |
| **CONTENTdm numeric census** | **All 1,310 live OCLC-hosted instances** — complete census. Finds: BPL Birmingham USAF 1948/51, El Paso PL Apr-1941. Custom-domain instances escape the sweep | **EXH+** (2) | Part 7, lane H |
| ~25 city library platforms | Probed clean | **EXH** | Lane H |
| Fort Collins, TEVA Nashville, Arizona Memory Prescott, kchistory | Record-only / dead stubs / bot-walls | record-only / **WALL** | Parts 4–5, lane H |
| Museums (10+, incl. Museum of Flight, EAA, College Park) | Record-only or nothing | **EXH** | Part 5 |
| **ASU Map & Geospatial Hub** | Inventory-as-ArcGIS-feature-service: 22 sectionals at 600 dpi 2022-23, **6 editions the dole lacks entirely incl. Juneau 1971 (nowhere else on earth)**; only 800px thumbnails public | **CONTACT** — top lead | Part 8, lane J |
| GIS geoportals / OpenGeoMetadata | All 36 repos cloned+grepped (5GB); ArcGIS Online global search | **EXH** | Lane J |
| Research-data repos (DataCite, Zenodo, Figshare, all Dataverses…) | **0 records in the entire DOI system** for the phrase — closes them all transitively | **EXH** | Lane K |

### International

| Venue | What was searched | Verdict | Log ref |
|---|---|---|---|
| NLA Australia / Trove | Sheila Scott collection (10 annotated US sectionals 1965–71, 5000px open) + Whiting NY 1957; **two undigitized physical sets: Bib 1030946 (~1970s NOS near-complete CONUS+AK+HI — "the desert era in one institution") + Bib 7796679 (C&GS/USAF 1940–66)** — digitisation-on-demand offered. Full keyless-API Trove sweep: 127 records, rest physical | **EXH+** (11) + **scan-request lead #1** | Parts 5, 8, lane L |
| DigitalNZ, Gallica, Canadiana, BC/Yukon, IWM, RAF Museum, BL, Japan, Ireland, South Africa, Europeana, LAC, McMaster | Foreign-theatre AAF/WAC/planning only | **EXH** | Lane L, Parts 4–5 |
| Russian/E-European hoards (loadmap, mapstor, poehali, retromap, radioscanner…) | Topo-only; "no runet hoard of US sectionals exists" | **EXH** | Lane M |

### Web archives

| Venue | What was searched | Verdict | Log ref |
|---|---|---|---|
| **Wayback — aeronav.faa.gov domain-wide** (1.22M urlkeys) | Old layout `sectional_files/*` → **280 edition zips 2011–19**; `visual/<date>/PDFs/` → **760 print PDFs** (13 whole + 11 partial editions; 407 URLs 1MB-truncated → editions 08-2021, 5×2022, 3×2023 unrecoverable); Seattle 2001 NACO training GeoTIFF | **EXH+** (1,044 rows) | Parts 2, 5 |
| Wayback — positive hosts | aviationtoolbox.org (3 Chicago GeoTIFFs), glideplan.com (2 Experimental mosaics; free tree never captured) | **EXH+** | Parts 3, 6 |
| Wayback — ~100 zero-yield hosts | chartbundle, vfrmap, skyvector, aeroplanner, maptech(+ftp), airnav, gelib, gearthhacks, bbs.keyhole, welch.econ.brown.edu, soaringdata.info, ~60 state DOT domains, 12 GIS clearinghouses, IU VDL, many more (full list log Parts 3–7) | **EXH** | Parts 3–7 |
| Common Crawl | 125 indexes per target host; pre-2013 ARC truncates at 512KB unflagged. Only glideplan Mac zips captured (all truncated; headers → `glideplan_macmaps_inventory.txt`) | **EXH** for known targets | Lane G |
| arquivo.pt | All target hosts: zero | **EXH** | Lane G |
| Memento TimeTravel, Bibliotheca Alexandrina | Gone / down | **DEAD** | Lane G |
| LOC webarchive, archive.today, Stanford SWAP, UKWA | Walled | **WALL** (low prior) | Lane G |

**Net verdict:** IA+CC+arquivo all negative ⇒ the chartgeek / fox-fam / welch / nlneilson /
soaringdata binaries are **archive-extinct — only the original authors have them** (hence the
CONTACT list).

### archive.org as content host

| Angle | Verdict | Log ref |
|---|---|---|
| Chart items: `sectional-charts` 2004 set (79), `noaa-aeronautical-charts` (6 mid-1930s), 5 Rumsey mirrors, SF 1966/1978, San Diego 1948, PC Pilot Issue 82 ISO | **EXH+** (~100) | Parts 2–4, lane A |
| Dozens of query campaigns (every product name: MapTech, FliteStar, JeppView, ChartGeek, Air Chart Systems…) + uploader enumerations. NB: bare `q=` is now full-text-only | **EXH** / **RECHECK** new uploads | Parts 3–7 |
| CD-ROM/ISO inspections (9 discs) + **discmaster.textfiles.com (1.7B files)**: no NACO/Jeppesen/MapTech chart disc ever ingested | **EXH** | Lanes A, I |
| Usenet: 21 rec.aviation.* mboxes grepped (~8,600 URL hits, ~700 hosts triaged) | **EXH** | Lane E |

### Commercial / marketplace

| Venue | What was searched | Verdict | Log ref |
|---|---|---|---|
| **raremaps.com** | CF-walled but scans on open GCS (`storage.googleapis.com/raremaps/img/dzi/img_<id>.dzi`, native 20–33k px, versos `_1.dzi`); enumerated via Wayback CDX slug filter → 43 findings incl. **LA 1976 r+v (desert era)**, ~30 WWII Restricted overprints | **EXH+** (43) + **RECHECK** new stock | Part 8, lane M |
| curtiswrightmaps.com | Dallas 1942 + Austin 1942 at 10,000px (strip `-scaled` from WP filenames) | **EXH+** (2) | Lane M |
| Auction/dealer sweep (12 houses) | Zero sectional scans | **EXH** | Lane M |
| Air Chart Systems / Sky Prints / Tri-Nav | Never digitized anywhere; **physical purchase only** (ASINs B003K2XBJC, B004HUERDK) | **EXH** record-only | Lane B |
| eBay/WorthPoint | Listing photos ruled out (low-res/watermarked); keep as physical-media channel | policy | Handoff r5 |
| Abandonware/disc DBs (winworld, macgarden, redump, vetusware) | Negative | **EXH** | Lanes A, I |

### Forums, communities, flight-sim

| Venue | What was searched | Verdict | Log ref |
|---|---|---|---|
| **simviation.com** | Sean Morrissey realcharts: 42 zips, complete ca-2010 US set — **downloaded 07-09** | **EXH+** (42) | Part 6, lane D |
| **usahas.com** | 23 ChartGeek mid-2009 superoverlay pyramids — **downloaded + mosaicked 07-08/09**; ~13 CONUS + AK KMLs 404 | **EXH+** (24) | Part 6, lane C |
| AVSIM library | **LOGIN**: Matt Fox 2005 Packs 1–11 (**Pack 11 = 11 AK charts; Point Barrow/Seward/W-Aleutian/Whitehorse exist in no other 1971–2010 source**) + Myers Mar-2010 HI/Guam/Samoa. Download IDs in lane_d_report | **LOGIN** | Part 6, lane D |
| pilotsofamerica thread 141250 | Attachment 113812 = full 1943 Cleveland PDF, login 403 | **LOGIN** (1 item) | Lane D |
| x-plane.org file 38646 | Bob Denny 1940s WEST sectional PDFs + mbtiles | **LOGIN** | Lanes C, I |
| flightsim.com / flightsim.to | CF + login, not enumerable | **LOGIN** (unverified) | Parts 3, 6 |
| GE Community ecosystem | GEC → fox-fam.com 53-zip Offline set (exact list recovered) → gelib/chartgeek storefront + 50 free city exes — binaries never crawled anywhere | **CONTACT** (Fox, ChartGeek) | Lanes C, G |
| nlneilson.com, welch.econ.brown.edu, soaringdata.info, alexisparkinn.com (Honeck) | Inventories/manifests recovered from CC/wayback; binaries never crawled | **CONTACT** each | Lanes E, G |
| glideplan.com | Unknown free tree `download/win_maps/<City>_<ed>.exe` — **117 city+edition mosaics eds 75–89 (2008–2011)**; company alive, contact form live | **CONTACT** — top lead | Lane G |
| 10+ other forums (backcountrypilot, supercub, PPRuNe, vansairforce, reddit…) | All point to known sources | **EXH** | Parts 4, 6 |
| airliners.net, shortwingpipers.org | Paywalled/403, no wayback | **WALL** (low prior) | Lane D |
| Usenet pre-2003 (usenetarchives.com) | Unlocked (base64 message-id trick, scriptable); targeted reads all led to known storefronts | **EXH** targeted; bulk sweep = diminishing returns | Part 7 |

### FTP / open-dir / dork campaigns

| Angle | Verdict | Log ref |
|---|---|---|
| FTP search engines (6) | **DEAD** ecosystem — never re-run | Lane I |
| Open-dir engines (ODCrawler ES backend direct: chart-negative) | **EXH** (re-query only if index grows) | Lane I |
| NACO internal filenames (seclosa/secsea98/…) across web+IA+CC+discmaster+govdocs1 | Indexed NOWHERE — filenames never leaked | **EXH** | Lane I |
| ~18 search-engine dork variants, S3/GCS bucket probes, GeoCities graveyards, photo hosts | Converge on known sources | **EXH** | Parts 4, 6, lanes D, I |
| Wikimedia Commons | Exhaustive 274-file inventory: Newberry Chicago 1955 ×2, ESRI SF 2008 | **EXH+** (3) / **RECHECK** uploads | Parts 3, 4 |
| GitHub (~90 repos) | Code only, no chart binaries | **EXH** | Part 4 |

## 2. Remaining acquisition lanes

1. **Owner/master-file leads:** ASU Hub, GlidePlan, Ivo Welch, Matt Fox,
   ChartGeek/DeTect, Neilson, Alley, Honeck, Tomblin and Morrissey. The lane reports
   hold contact context, ASU barcodes and inventories. Outreach status is unknown;
   check prior correspondence before contacting anyone.
2. **Scan requests:** NLA Bib 1030946 and 7796679, Fort Collins Cheyenne 1955/59,
   Harvard Orlando 1943, Archives West/OAC papers, ASU Phoenix 1968-01-04. Historical
   holdings claims require verification when resuming the lane.
3. **Login access:** AVSIM Fox/Myers packs, POA 141250 and x-plane.org 38646.
   Pack 12 was already cataloged; reconcile it via 02 instead of counting it as a
   new find. Other access/download completion is unrecorded here.
4. **Physical media:** MapTech AeroPack, ChartGeek, RMS VISTA, Air Chart Systems,
   NACO subscription and PC Pilot cover discs. Cross-check the later recoveries
   in [11](11_acasis_img_recovery.md) and [12](12_sdcard_batch.md) before buying
   or soliciting duplicates. No saved-search automation is established by this doc.
5. **Candidate preparation:** [07](07_download_queue.md) owns the 258 deferred
   HUNT26 candidate files and their historical import plan. Seward 93 needs a
   different source; its known captures were exhausted in July.
6. **Targeted rechecks:** new archive.org/GPO microfiche and Wikimedia uploads,
   raremaps **live sitemap** (round 9 superseded the CDX suggestion), current FAA
   cycle directories, and specific browser-only/custom-domain institutional leads.
   No later sweep is recorded. The university ArcGIS and usenet bulk lanes were
   exhausted in round 9; reopen only with a concrete new lead.

## 3. Rules of the hunt (summary — full text in `dole_search_handoff_round8.md`)

- **Scope**: 1:500k sectionals only (USAF editions, versos, half-sheets, and distinct scans of dole-held editions COUNT); exclude TAC/WAC/ONC/planning/helicopter/IFR/strips/state charts/crops.
- **Pyramids/mosaics** assembling a complete chart count as findings (artifact class in note); sub-~1000px thumbnails are record-only.
- **Verify** every URL (`curl -sIL` → 200 + plausible size); Wayback `id_` + largest CDX capture; ~1MB on a big file = truncated, reject; CC pre-2013 ARC silently truncates at 512KB.
- **Dedupe** against BOTH the catalog (loaded with `scripts/dole_v2.py`) and
  missing_from_dole_online.csv, including **location + edition + side**, not only
  download_link/filename or flattened URLs. July's URL-only check overcounted
  1,021 already-held editions (07).
- **Record** findings as CSV rows; record-only items in the log; log zero-yields too, as the next log Part.
- Technique cheat-sheet (raremaps GCS bypass, Trove keyless API, usenetarchives unlock, NARA WAF quirks, CDX gotchas, bot-wall master list): handoff §TECHNIQUE.
