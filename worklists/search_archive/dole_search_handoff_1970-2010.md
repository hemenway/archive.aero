# HANDOFF — Dole internet search, round 5: focus 1970–2010

Written 2026-07-08 at the end of round 4. Your job: continue searching the internet for
downloadable full-resolution scans of US **sectional aeronautical charts** (1:500,000,
NOAA/FAA era) **published ~1970–2010** that are missing from the dole. This era is the
dole's biggest desert and rounds 1–4 barely dented it.

## Files (repo root: /Users/ryanhemenway/archive.aero)
- `master_dole_v2.csv` — the catalog ("dole"), 7,247 rows, one per chart scan; loader `dole_v2.py`
- `missing_from_dole_online.csv` — 1,373 verified online findings (source,name,url,date_or_edition,note)
- `dole_search_log.txt` — **READ PARTS 1–5 FIRST.** Every source already searched, including
  dead ends, bot-walls, and zero-yield queries. Re-searching anything in it is wasted work.
- `missing_from_dole.csv` + `missing_from_dole/` — the local-disk audit (not your task)

## Why 1970–2010 is the desert
- LOC set gct00089 (the dole's backbone) tapers off in the early/mid-1970s (ends ca004091).
- FAA digital coverage (edition zips) starts 2011.
- In-era holdings are a handful of one-offs, mostly recorded as findings, not yet downloaded:
  WASP rows 1971–1982 (SF 71, Denver 72/75, Wichita 72, DFW 81/82), the complete 2004 FAA
  edition (archive.org `sectional-charts`, 79 half-sheets), NARA paper Juneau eds 34–56
  (1994–2016), aviationtoolbox Chicago 2003/2005 GeoTIFFs, SF 2008 ESRI sample PDF,
  Seattle 2001 NACO training chart, SF 1966/1978 archive.org scans, NOAA Anchorage 1970,
  NLA Sheila Scott 1965–71 annotated charts.
- Net: for most cities, the mid-1970s through the 1990s have ZERO coverage. Any full-chart
  scan from this window is valuable, including odd editions, USAF editions, and versos.

## Definitively closed — do NOT revisit (details in log Parts 2–5)
- **NARA**: all 1.45M `lz/cartographic/` S3 keys enumerated + catalog fully paginated.
  Series 305976 (78 objects) is complete; RG 342 AF editions undigitized. Closed.
- **LOC**: nothing outside gct00089/gct00498; both fully probed.
- **UNT Portal + Gateway to Oklahoma**: full OAI sweeps (3.67M records). Closed.
- **NACO CD-ROM raster era (eds ~60–85, 1998–2011)**: subscription CD products, never freely
  hosted; wayback shows nothing beyond the Chicago/ESRI/Seattle-training exceptions above.
  (But see lead #1 below — the CDs themselves may survive as ISOs.)
- Wayback CDX zero-yield hosts (25+): skyvector, soaringweb, wingsandwheels, aeroplanner,
  fltplan, airnav, chartbundle, vfrmap, runwayfinder, aircharts, sectionalcharts.com,
  landings.com, vfrcharts.com, naco/avn.faa.gov, etc. — full list in log.
- aeronav.faa.gov domain-wide CDX done (1.22M urlkeys); www.faa.gov likely-path subtrees done.
- Wikimedia Commons, Flickr (incl. SDASM), GitHub, WorldCat, Europeana, DPLA, HathiTrust,
  museums (10+ checked), and ~30 university/state digital collections: all swept, log has
  the per-source verdicts.

## Suggested leads for 1970–2010 (untried; verify, expand, follow your nose)
1. **Commercial chart CD-ROM/atlas products as ISOs or scans** — the era's charts survived
   commercially, not on the web. Hunt archive.org software/CD-ROM collections and
   abandonware sites (vetusware etc.) by PRODUCT NAME (prior negative queries were only
   generic "NACO"/"CD-ROM" phrasings): MapTech aviation/Pocket Sectional CDs, RMS Technology
   FliteSoft, Jeppesen FliteStar / JeppView VFR raster CDs, Control Vision Anywhere Map,
   "VFR RasterPlus", Sporty's chart CDs. Any ISO with full-chart rasters is a finding.
2. **Air Chart Systems (Howie Keefe) atlases** — spiral-bound books reproducing full
   sectionals with update stickers, sold 1970s–2000s. archive.org texts collection, Google
   Books full-view, used-book scan sites. A scanned atlas = dozens of in-era charts.
3. **Google Earth Community / gearthhacks KMZ overlays (2006–2010)** — hobbyists overlaid
   scanned old sectionals; forum archives + wayback of linked file hosts.
4. **Personal pilot sites/blogs** hosting scans of 70s–90s charts — deep search-engine
   passes ("sectional chart 1978" scan, "old sectional" jpg, city+year phrasings), then
   wayback the dead hosts.
5. **FDLP depository libraries** — GPO distributed sectionals to depository libraries;
   look for digitized "superseded charts" collections (search FDLP/GPO + library sites).
6. **Usenet archives** (rec.aviation.* via Google Groups / archive.org UTZOO-style dumps)
   for chart-scan site URLs to feed into wayback.
7. **State aviation agencies / DOT aeronautics divisions** that scanned historical state
   chart sets (some published annual chart books containing the sectional).
8. **archive.org uploader probes** — find users uploading aviation CD ISOs / chart scans
   and enumerate their items (technique worked in round 3: peron@sdf.org).
NOT worth it: eBay/WorthPoint listing photos (low-res/watermarked), tile pyramids, login-
walled sim libraries (avsim/flightsim.to — already noted).

## Rules & conventions (match rounds 1–4 exactly)
- Sectionals only. EXCLUDE: TAC, WAC, ONC/JOG, planning, helicopter, Grand Canyon,
  Caribbean VFR, IFR/enroute, airway strip maps, obstruction charts, state 1:1M charts,
  crops/excerpts. Half-sheets and versos COUNT. Distinct scans of editions the dole already
  holds COUNT (say so in the note).
- Verify every URL: `curl -sIL` → 200 + plausible size. Wayback: use the
  `web.archive.org/web/<TS>id_/<url>` form; pick the largest capture per URL from full CDX
  history; beware 1MB-truncated captures (check content-length).
- Dedupe against BOTH `master_dole_v2.csv` (download_link + filename; note the dole
  flattens URLs: `/` and spaces → `_`) and `missing_from_dole_online.csv`.
- Findings: append rows (source,name,url,date_or_edition,note) to
  `missing_from_dole_online.csv`. Validate with python csv (5 columns, no HTML entities).
- Log: append **Part 6** to `dole_search_log.txt` — positives AND zero-yields, same style.
- Record-only items (no download) go in the log, not the CSV.
- Python: `~/venv/bin/python` (3.14), NOT `./.venv`.
- Parallel fan-out subagents with strict lane assignments worked well in rounds 3–4;
  give each agent the exclusion list and the "read the log first" instruction.

## Technique cheat-sheet (hard-won)
- Wayback CDX: `web.archive.org/cdx/search/cdx?url=HOST/*&output=json&collapse=urlkey`
  (+ filter/limit/pagination). Domain-wide sweeps are feasible up to ~1M urlkeys.
- archive.org: `archive.org/advancedsearch.php?q=...&output=json`; item files via
  `archive.org/metadata/<item>`.
- NARA proxy search (if ever needed): unquoted q only, numeric-only q trips WAF, 45–60s
  cooldowns, `ancestorNaId=` reliable.
- Known bot-walls (curl+WebFetch blocked; need manual browser): NYPL, Stanford, Florida
  Memory, DLG Georgia, Arizona Memory (Recollect), WorldCat (Turnstile), dp.la, TAMU.
- NLA: `nla.gov.au/nla.obj-<id>/image` is open at 5000px; `/dzi` descriptor gives native size.

## Unrelated leftovers (out of era focus; optional manual-browser tasks)
- Arizona Memory `nodes/view/150290`: Prescott R-3 1955 USAF edition (Sharlot Hall Museum) —
  record confirmed, download never verified (Cloudflare).
- TEVA Tennessee ids 6455/6456/8971: Nashville 1944/45/51 — records are dead `.url` stubs.
