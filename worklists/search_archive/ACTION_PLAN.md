# ACTION PLAN — consolidated "things to look into" (written 2026-07-08)

One-page index of everything actionable in this folder, ordered by impact on the
**1971–2010 gap** (the dole's desert: LOC ends ~1974, FAA digital starts 2011).
Deep detail lives in `dole_search_handoff_round8.md` (authoritative, supersedes the
round-5 handoff), `dole_search_log.txt` Parts 1–8, and `dole_search_lane_reports/`.

Era math on the online findings: of 1,490 rows in `missing_from_dole_online.csv`,
**~172 are 1971–2010** (79 = archive.org 2004 set, 42 = simviation ca-2010,
24 = usahas 2009, 17 = NARA Juneau, rest one-offs). Everything else is pre-1971
or post-2010. The desert is filled by ACTIONS below, not more searching.

---

## TIER 0 — Already on your disk; just needs cataloging (hours, free)

**CLOSED 2026-07-14** — every row of `missing_from_dole.csv` reconciled; dates read off
the actual sheets (see `catalog_additions_report.csv` for the row-by-row disposition).
**Headline correction: the "SEC NN" filename numbers are EDITION numbers, not years.**
None of these fill the 1975–1999 desert:

- [x] **Dallas-Ft Worth SEC 92 / 94 / 96** = 92nd/94th/96th editions (2014/2015/2016) —
      already in dole from FAA wayback zips; no rows added. `SEC_output.tif` = the
      2025-11-27 cycle, also already in dole.
- [x] **Seattle SEC 86** = 86th ed 2013-12-12 (NOT-FOR-NAVIGATION watermark) — already in dole.
- [x] **Wichita SEC 95** = 95th ed 2015-07-23 — already in dole.
- [x] **Honolulu / Mariana / Samoan Islands insets SEC 95** = Hawaiian Islands **95th-ed
      insets (2016-10-13)**, no printed date on the sheets — 3 rows ADDED (NARA S3 links).
- [x] **us_sectionals_pack_12 JPGs** — 11 rows ADDED with DATE-APPROX 2004-07-01 (margins
      cropped, no edition panels; ca. 2004-05 per AVSIM). Point Barrow halves are the
      unique payload.
- [ ] **Seward_93.zip repair** — row ADDED (ed 93, 2013-11-14→2014-05-29, from FAA cadence
      + wayback URL) but the local zip is still the truncated 1.2 MB capture; re-pull the
      largest CDX capture of the same URL.
- [x] Out-of-era chores: Whitehorse SEC 54 = 54th ed 2014 (dup), W. Aleutian 47 PDFs =
      47th ed 2014 halves (dup of whole-chart row). 7 LOC `ca######` tiffs: **4 Portland
      scans fill the dole's 1947-10→1949-10 hole exactly** + Portland 32nd ed 1954-03-23
      (ca003009r's end split accordingly) + Milwaukee ca002354 (DATE-APPROX Oct 1941) —
      6 rows ADDED (ca003008v verso skipped). 2 WASP tiffs = blank versos of rectos already
      in dole. 3 archive.org FAA PDFs = 2026-03-19 editions, already in dole. 7+1 modern
      `*_Sectional.zip` bundles = local copies of cycles already cataloged (2024-12-26
      through 2025-11-27 + 2026-05-14 All-Files) — no rows needed.
- [x] **87 `sectionalproject_renamed/` Dallas files** — all reconciled: 83 by year-month,
      4 read off the sheets (193302 = Jan 1933 airway map; 194407 = AUG 24 1944;
      194507 = SEPT 25 1945; 202107 = 2021-06-17 Dallas-Ft Worth). Zero new editions.

## TIER 1 — Downloads you can script TODAY (no gatekeeper) — in-era payload

**DONE 2026-07-08/09 → everything staged in `/Volumes/projects/rawtiffs/dole_gap_2026-07/`**
(see its README.txt; manifest_*.json files map each file to its CSV row):

- [x] **usahas.com ChartGeek pyramids** — archival grab complete: all 24 KMLs + 2,603
      tiles (61 KML-listed tiles are server-404 = never-generated padding). 23 mosaics
      assembled as georeferenced LZW GeoTIFFs (EPSG:4326 from the KML LatLonBoxes),
      ~10-12k × 8k px; regions with no deep tile carry upscaled L4/L3 content
      (Cheyenne row 4, one Seattle area). QA'd visually — no holes.
- [x] **simviation.com realcharts** — all 42 zips (1.3 GB), integrity-verified;
      94 chart PDFs inside (complete ca-2010 US set incl. the 4 Alaska packs).
- [x] **archive.org `sectional-charts` 2004 set** — 79/79 half-sheets (294 MB).
- [x] In-era one-offs (32 files + 2 SF PDFs): NARA Juneau eds 34–56 (23 scans ~22k px),
      aviationtoolbox Chicago 67/70 GeoTIFFs, GlidePlan Reno_Whites + Mt Shasta zips,
      ESRI SF 2008 PDF, Seattle 2001 NACO training GeoTIFF (= the wayback faa.gov
      sample.tif), NOAA Anchorage 1970 r+v, SF 1966 + 1978 archive.org PDFs.
- [x] **Catalog the downloads** — DONE 2026-07-14: 189 rows added to master_dole_v2.csv
      (78 archive2004 half-sheets + 54 simviation charts, edition dates read off every
      sheet; 23 usahas mosaics DATE-APPROX mid-2009; 23 NARA Juneau eds 34–56, each
      sheet read individually; 11 oneoffs incl. NOAA Anchorage 1970 base printing,
      Chicago 67/70, SF 1966+1978, Seattle 2001 training, ESRI SF 2008, GlidePlan ×2).
      Row-level log: `catalog_additions_report.csv`. Slicer pipeline still to run.
- [ ] Push the new rows through the slicer pipeline (GCPs/georef via georeftool first).
- [ ] Then sweep the rest of the 1,490 findings (pre-71 + post-2010) as a background batch.

## TIER 2 — Emails to write (files CONFIRMED to exist; biggest in-era unlock per hour)

1. [ ] **ASU Map & Geospatial Hub** (lib.asu.edu/geo) — holds unpublished 600-dpi 2022-23
       masters of 22 sectionals incl. SIX dole-absent editions: **Juneau 1971 (exists
       nowhere else on earth)**, Cape Lisburne 1971, Anchorage 1971, Albuquerque 1971,
       Hawaiian/Marianas/Samoan 1971, Prescott 1964 f+b. Exact FILE_NAMEs, SuDoc, barcodes:
       `lane_j_report.md`. Piggyback: ask them to scan their unscanned Phoenix 1968-01-04.
2. [ ] **GlidePlan** (company alive, contact form live) — 117 city+edition seamless mosaics,
       eds 75–89 (2008-2011), + 2010 CMR sets. Precise ask: `lane_g_report.md` +
       `glideplan_macmaps_inventory.txt` (filenames, sizes, dates from Common Crawl headers).
3. [ ] **Ivo Welch** (UCLA Anderson; site invites contact) — complete Jan-2006 NACO eastern
       US set: 38 half-sheet JPEGs + GeoTIFF twins (maybe the source NACO DVDs too).
       Exact filenames: `lane_g_report.md`.
4. [ ] **Matt Fox** (Forkboy2 / gelib.com) — 2005-09 overlay source rasters; 53-zip Offline
       set incl. all 15 Alaska charts (exact list recovered, `lane_g_report.md`).
5. [ ] **ChartGeek author** — 50 free city exes (2011-12 eds, incl. all AK) + the ~$90 DVD
       (all sectionals+TACs). Also DeTect/usahas re: the 13 CONUS + AK KMLs that 404.
6. [ ] Second string: Neil Neilson (WorldWind packs), Lynn Alley (soaringdata 2010 CMR),
       Jay Honeck (7 Des Moines scans 1939-68, exact URLs known), Paul Tomblin,
       Sean Morrissey (rottydaddy@msn.com — full-size 2010 masters?).

## TIER 3 — Free-login browser afternoons (manual, cheap)

- [ ] **AVSIM account** — unlocks Matt Fox 2005 Packs 1-11 (ca. 2004-05 editions;
      **Pack 11 = 11 AK charts — Point Barrow, Seward, W. Aleutian, Whitehorse exist in NO
      other 1971-2010 source**) + David Myers Mar-2010 Hawaii/Guam/Samoa (Aug-2009 eds)
      + 250k-inserts pack. Download IDs: `lane_d_report.md` / log Part 6.
- [ ] pilotsofamerica.com thread 141250, attachment 113812 — 1943 Cleveland PDF (out of era).
- [ ] x-plane.org file 38646 — 1940s western sectional scans (out of era).
- [ ] Low-prior browser-only residuals: Arizona Memory Prescott R-3 1955, OldMapsOnline,
      IU Virtual Disk Library, kchistory.org, LOC webarchive, archive.today.

## TIER 4 — Scan requests to institutions (slow, some cost, unique payoff)

1. [ ] **NLA Australia Bib 1030946 — the single biggest desert lever.** Near-complete
       ~1970s CONUS+AK+HI National Ocean Survey sectional set in one collection; NLA does
       digitisation-on-demand. Also Bib 7796679 (C&GS/USAF 1940-66).
2. [ ] Fort Collins History Connection — Cheyenne base-1955/rev-1959 (acc 1979.065.0001).
3. [ ] Harvard Gray Herbarium — Orlando O-8 Jan-1943 Restricted (out of era).
4. [ ] Archives West / OAC personal papers (Spokane 1965, SF/Seattle 1941-45, etc. — log Part 6).

## TIER 5 — Physical media on the used market (money; proven no rips exist online)

- [ ] **Air Chart Systems VFR Sectional Atlas** (~1962-2013; one atlas = half-US coverage;
      ASINs B003K2XBJC, B004HUERDK) — a single 1980s/90s atlas would carpet the desert.
- [ ] MapTech "VFR/IFR AeroPack" regional CDs (~1999-2006, BSB rasters)
- [ ] RMS Technology "VISTA" sectional CD-ROMs (1997+)
- [ ] ChartGeek DVD (~2009), NACO subscription raster CDs (eds ~60-85)
- [ ] PC Pilot cover discs not on IA (issues 22-81, 84+)
- Set up eBay saved searches for the product names above.

## TIER 6 — Residual search frontiers (thin; only after Tiers 0-5)

Custom-domain CONTENTdm dorks, raremaps.com GCS re-sweeps (durable bypass), usenetarchives
bulk pre-2003, periodic re-checks (FAA next-edition dirs, new archive.org/Wikimedia
uploads, eBay→dealer-scan pipeline). Everything else is CLOSED — see log Parts 1-8
before searching anything.

---

### Folder map
| Path | What it is |
|---|---|
| `ACTION_PLAN.md` | this file |
| `dole_search_handoff_round8.md` | authoritative hunt state + rules (supersedes 1970-2010 handoff) |
| `dole_search_log.txt` | Parts 1-8: every source ever searched, incl. dead ends |
| `dole_search_lane_reports/` | per-lane detail: inventories, barcodes, exact URLs for emails |
| `missing_from_dole_online.csv` | 1,490 verified downloadable findings |
| `missing_from_dole.csv` | 136-row local-disk audit (files here but not in dole) — CLOSED, see report below |
| `catalog_additions_report.csv` | 2026-07-14 cataloging: per-file ADD/SKIP/FIX disposition (210 rows added) |
| `end_date_fill_report.csv` | 2026-07-14 end-date cleanup: all 294 date/end_date changes with method |
| `loc_ca_scan/` `sec_chart/` `sectional_named/` `sectionalproject_renamed/` `wasp_scan/` | symlinks to the local-audit files, grouped |
