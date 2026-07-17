# LANE F REPORT — round 5 (FDLP/depository-library digitizations + state aviation agency / DOT aeronautics archives)
Date: 2026-07-08. Verified findings: see lane_f_findings.csv (1 row).

## POSITIVES

Birmingham Public Library CONTENTdm (bplonline.contentdm.oclc.org, "Maps Project" p15099coll3):
  Birmingham (Q-7) Sectional Aeronautical Chart, U.S. Air Force Edition — "Edition of Sept. 1948
  Revised July 1951", print date Aug 16 1951, USC&GS, verso text/illustrations noted in metadata
  (only recto digitized; single-image item, no compound pages). IIIF native 5862x2788px; full JPEG
  https://bplonline.contentdm.oclc.org/iiif/2/p15099coll3:1147/full/full/0/default.jpg verified
  200, 3.19MB downloaded. DEDUP: same USAF edition as the cartweb Alabama1951b.sid finding in
  missing_from_dole_online (UA scan is only 3664x1743) — this is a DISTINCT, HIGHER-RES scan;
  dole itself holds only civil 1951 Birmingham eds (ca000360r/ca000361r). BPL "aeronautical"
  search total = 2: the other (id 1472 "Alabama - aeronautical chart") is a state chart, EXCLUDED.

## ZERO-YIELD (FDLP / university / state-hub angle) — all checked this round
  Harvard: LibraryCloud API (api.lib.harvard.edu/v2/items.json) swept "sectional aeronautical"
    (484 keyword hits reviewed): exactly ONE chart record — Orlando (O-8) 1943 "Restricted"
    (Gray Herbarium Map Case 75.9, OCLC 955693114) — PHYSICAL ONLY, no digital object/IIIF;
    record-only. Also a "Sectional raster aeronautical charts" serial record = NACO CD product,
    physical CDs, no scan. curiosity.lib.harvard.edu = 403 bot-wall (LibraryCloud covers it).
  Yale: collections.library.yale.edu bot-challenged (HTTP 202); site: dork shows only AAF cloth
    charts (foreign) — no US sectionals digitized.
  Princeton DPUL: catalog.json API works — 0 results "sectional aeronautical".
  Cornell: digital.library.cornell.edu API 403; dork zero.
  U. Chicago: only VuFind catalog record for current FAA "VFR raster charts" (physical/current);
    no digitized sectionals.
  USC, UC Berkeley (incl. geodata.lib.berkeley.edu), UCLA, OhioLink DRC: dorks zero.
  Rice, LSU/Louisiana Digital Library, U. Kentucky (exploreuk + kdl.kyvl.org), Historic
    Pittsburgh, Rutgers RUcore, SUNY Buffalo, WVU (wvhistoryonview + wvculture), Auburn
    (CONTENTdm API probed: 51 hits all newspapers/yearbooks): zero.
  ASU keep.lib.asu.edu, UNLV special.library.unlv.edu, U. Idaho, Boise State, Oregon Digital
    (UO+OSU; API bot-challenged, dork zero): zero.
  Notre Dame, Syracuse, Brown, Dartmouth, UMD, UMass Credo, U. Delaware udspace, UT Knoxville,
    Arkansas, Ole Miss (msdiglib), OU (digital.libraries.ou.edu): dorks zero.
  ERAU (commons.erau.edu Scholarly Commons/Data Commons, aviation-specific): dork zero; search UI
    is JS-only. Prescott Aviation Safety Archives = physical finding aids. Low-value lead.
  State hubs: DigitalNC, NY Heritage, SC Digital Library (scmemory), Virginia Memory,
    Michiganology, Mountain West DL, Illinois Digital Archives (idaillinois CONTENTdm API: 4 hits
    all newspapers), Indiana Memory (API: 0), Digital Maryland (API: 0), Maine (digitalmaine),
    NH/VT/WY/ID/AZ state archives dorks: zero. CARLI collections.carli.illinois.edu API: 3 hits,
    only real one = Newberry Chicago (U-7) 1955 nby_chicago id 7036 — SAME item already recorded
    via Wikimedia re-upload; CONTENTdm children 7034/7035 native are the SAME 3600px (no upgrade).
  Cross-platform dorks: quartexcollections.com zero; *.access.preservica.com zero (TSLAC TxDOT
    Preservica pages are records/docs, no charts); Islandora dork zero; "inurl:iiif" dork zero;
    JSTOR zero; "index of" .edu dork → only the known FAA productcatalog Raster sample (already
    in missing_from_dole_online via wayback; live dir is a rebuilt Drupal product page).
  archive.org: collection:fedlink "sectional aeronautical" (4 hits, all NACA/NPS reports);
    "superseded charts" aeronautical = 0.
  FDLP/GPO: fdlp.gov/govinfo dorks — offers lists/CFR text only, no digitized superseded charts.

## IU "Virtual Disk Library" (webapp1.dlib.indiana.edu/virtual_disk_library) — the one real FDLP
  digitization program found (mounted GPO depository CD-ROMs, browsable per barcode). Live site now
  behind Cloudflare Turnstile. Wayback CDX: 10,984 unique captured URLs, 443 disk barcodes seen in
  paths, 212 disk index pages captured; fetched 207/212 index pages + resolved the 5 stragglers'
  titles (census/pavement/DEA-solver/practical-navigator) — CD titles are census/health/defense/
  USGS-text products, ZERO aeronautical chart raster products (title grep aeronaut|chart|
  sectional|raster = 0 across all 212). Path-level audit of ALL 10,984 URLs (covers all 443
  barcodes): 0 .tif/.sid/.jp2, image files are UI icons + oceanexplorer web JPGs. Only
  chart-adjacent disk = barcode 451066 "Aeronautical charting data, sampler II" (FID2498/DATA/
  MOAS.RAW visible) = the 1990 sampler ALREADY CONFIRMED text-only dBASE in round 4 (LOC copy).
  NOAA weather chart CDs (4271534) = weather fax charts, not sectionals. Conclusion: GPO never
  distributed NACO sectional-raster CDs to depositories; VDL closed.
  (Record-only lead: full VDL disk list beyond the 212 captured pages is behind the Turnstile
  wall — a manual-browser session could enumerate the complete list, but title distribution seen
  gives near-zero prior for chart rasters.)

## ZERO-YIELD (state aviation agency / DOT aeronautics angle) — wayback CDX, filter sectional,
  collapse urlkey, domain match (prefix where noted):
  dot.state.mn.us, wsdot.wa.gov (+/aviation*), doav.virginia.gov + doav.state.va.us, oac.ok.gov +
  oac.state.ok.us, mdt.mt.gov (+/aviation*), itd.idaho.gov (+/aero*), dot.ca.gov
  (+/hq/planning/aeronaut*), txdot.gov + dot.state.tx.us (highway cross-section noise only),
  fdot.gov + dot.state.fl.us, ncdot.gov, dot.ga.gov + dot.state.ga.us, azdot.gov + dot.state.az.us,
  codot.gov (only ~190KB AWOS sectional CROPS — excluded), udot.utah.gov (+/aeronautics*,
  aeronautics.utah.gov), dot.nd.gov, dot.sd.gov, dot.state.il.us, aviation.illinois.gov,
  iowadot.gov + dot.state.ia.us, ksdot.org + dot.state.ks.us, modot.org (overhead-door PDFs!),
  aeronautics.nebraska.gov + aero.nebraska.gov + dot.nebraska.gov + aero.state.ne.us, penndot.gov +
  dot.state.pa.us + aviation.state.pa.us, transportation.wv.gov, wisconsindot.gov +
  dot.wisconsin.gov + dot.state.wi.us + flywisconsin.com, dot.state.wy.us, dot.alaska.gov
  (+/stwdav*), hidot.hawaii.gov + airports.hawaii.gov + aviation.hawaii.gov, dot.state.oh.us +
  transportation.ohio.gov, transportation.ky.gov, dotd.la.gov, dot.state.al.us, dot.state.la.us,
  dot.state.co.us, dot.state.nm.us, dot.state.ok.us, mnaero.com, mdot.state.mi.us,
  aeronautics.mt.gov, aeronautics.wa.gov, michigan.gov/aero* + /mdot*, in.gov/indot*,
  tn.gov/aeronautics*, state.nj.us/transportation*, oregon.gov/aviation* + /ODA*.
  NET: no state DOT/aeronautics agency ever hosted scanned sectionals; their "sectional" images
  are airport-vicinity crops (CODOT AWOS) — excluded. State aeronautical charts (WisDOT etc.)
  excluded by rule.
  State GIS clearinghouses (idea: pre-2011 NACO raster mirrors): pasda.psu.edu,
  deli.dnr.state.mn.us, rgis.unm.edu, msdis.missouri.edu, asgdc.state.ak.us, agrc.utah.gov,
  kygeonet.ky.gov, geostor.arkansas.gov, tnris.org, wagda.lib.washington.edu, nris.mt.gov,
  data.wvgis.wvu.edu — CDX filter sectional/aeronaut: all zero. Clearinghouses never mirrored
  FAA sectionals.
  North Dakota SHSND "Aeronautics Commission" state-agency records: paper finding aid only.
  Montana Aeronautics Division records (Archives West): finding aid record-only.

## RECORD-ONLY ITEMS (log, not CSV)
  - Harvard Gray Herbarium: Orlando (O-8) sectional, Jan 14 1943 "Restricted", 50x83cm —
    undigitized (HOLLIS 990147545500203941). Candidate for scan request.
  - IU VDL complete disk list behind Turnstile (low priority, see above).
  - Archives West / OAC finding aids list PAPER sectionals in personal-papers collections
    (all undigitized): Callison Marks (WSU?) Spokane Mar 1965; John E. Rouse SF+Seattle 1941-45;
    Frank Sarris Boston (UV-10) Dec 26 1945 (OAC); Helen Richey Detroit Mar 1936 + 1940s
    restricted eds (OAC); Charles Wallace El Paso/San Antonio/Del Rio (OAC); Pangborn
    Collection SE-US sectionals (WSU MASC). Record-only; possible scan-request targets.

## TECHNIQUE NOTES
  - CDX gotcha: url=HOST/* + unencoded regex filter silently returns empty; use
    matchType=domain + --data-urlencode for the filter. 504s on big domains (wsdot, dot.ca.gov,
    michigan.gov...) — fall back to aviation-subdir prefix probes.
  - CONTENTdm cross-collection JSON API works unauthenticated on most instances:
    /digital/api/search/searchterm/<terms>/field/all/mode/all/maxRecords/30; per-item:
    /digital/api/collections/<coll>/items/<id>/false; IIIF /iiif/2/<coll>:<id>/info.json.
    Instances probed OK: okstate, auburn, carli, idaillinois, indianamemory, digitalmaryland,
    bplonline, kdl.kyvl.org. Bot-blocked/non-CDM: digital.boisestate.edu, d.library.unlv.edu,
    wvhistoryonview.org.
  - Harvard LibraryCloud (api.lib.harvard.edu/v2/items.json?q=) is open, no key — good for any
    future Harvard probing; exact-phrase q returns 0, use unquoted keywords.
  - New bot-walls hit this round: webapp1.dlib.indiana.edu (Cloudflare Turnstile),
    curiosity.lib.harvard.edu (403), collections.library.yale.edu (202 challenge),
    digital.library.cornell.edu (403), oregondigital.org (challenge redirect).

## LEADS FOR FUTURE ROUNDS
  1. Ask-a-librarian scan requests for confirmed undigitized in-era paper holdings (Harvard
     Orlando 1943; U. Chicago current-cycle physicals are not in-era) — out of scope for search.
  2. BPL "Maps Project" has only the one sectional, but other city public libraries with
     CONTENTdm may hide singletons findable ONLY via the general contentdm dork used here
     ("contentdm.oclc.org <chart-city> sectional aeronautical chart" per-city variants could
     surface more one-offs; only Birmingham/Nashville(TEVA dead) indexed today).
  3. IU VDL full enumeration via manual browser if anyone wants to be exhaustive (low prior).
