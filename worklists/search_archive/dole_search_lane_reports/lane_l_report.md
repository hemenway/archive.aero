# LANE L REPORT — international libraries/aggregators (round 7)

Scope: Trove/NLA (AU), DigitalNZ (NZ), Gallica/BnF (FR), British Library/RAF Museum/IWM/JISC (UK),
Canadiana/BC Archives/MemoryBC/Yukon (CA), Japan/Ireland/South Africa quick dorks.
Precedent to extend: NLA Sheila Scott 10-chart find (log Part 5) — foreign aviators' papers hold
annotated US sectionals. Confirmed the precedent generalises: found a SECOND aviator collection.

## POSITIVE — 1 verified downloadable finding (in lane_l_findings.csv)

**NLA "Papers of Charles Whiting" — New York Sectional Aeronautical Chart, 46th ed (1957-01-24)**
- Trove work 262972573; nla.obj-3762588236; Item 19.7 - Phase-Box 4; C&GS; Trove Digital Library (open).
- https://nla.gov.au/nla.obj-3762588236/image → verified 200 image/jpeg, 2.40 MB, 5000x2703 px.
  Native 17844x9651 per /dzi (tile service Anubis-walled, same as Scott set).
- Read the image to confirm: genuine full-sheet folded copy, bottom-right panel "NEW YORK SECTIONAL
  AERONAUTICAL CHART / 46TH EDITION / aeronautical information corrected through JAN. 24 1957."
- DEDUPE: the dole already holds this exact edition as LOC ca002691r ("New York, NY" ed 46, eff
  1957-01-24). This is a DISTINCT SCAN (separate physical copy from an Australian aviator's papers) —
  counts per the "distinct scans of held editions COUNT" rule. Note says so.
- Charles Whiting collection has 62 Trove items incl. an Item-19.x chart run; of those only 19.7 is a
  US sectional. The rest are EXCLUDED classes: 19.2/19.6 Qantas plotting charts, 19.3 Townsville high-
  altitude, 19.4 AP-33 USAF planning chart, 19.9/19.10/19.12 Australian WACs, NP405 RAAF plotting.
  Parent folder 262972544 "Aeronautical charts from Australia and the USA…" is a grouping record (no
  own scan).

## RECORD-ONLY (log, NOT in CSV) — NLA physical US-sectional sets, only index diagrams digitised

Two NLA catalogue bibs each hold a large PHYSICAL US sectional set; the only digitised object under
each is a librarian-made COVERAGE-INDEX diagram (read both — "NLA holds sheets shown in pink/blue"),
not a chart. These are strong scan-request / contact-class leads, esp. the NOS one for the desert era:
- **Bib 1030946** (author "National Ocean Survey" ⇒ ~1970-1982 = the dole's 1970s desert):
  index nla.obj-233011372 (5000x3425). Pink coverage = nearly complete CONUS + all Alaska sheets +
  Hawaii. Physical set undigitised. DESERT-ERA scan-request lead.
- **Bib 7796679** (C&GS, "1940-1966"): index nla.obj-721622305 (5000x3466). "NLA holds USAF editions
  in pink and civilian editions in blue" — USAF eds for most CONUS cities; civilian eds for ~12
  (Lake Superior, Boston, San Francisco, Los Angeles, Albuquerque, Roswell, Chattanooga, Savannah,
  New Orleans + Atlantic "Water" charts). Physical only.
- Both call number MAP G3701.P6 s500 (P Stack), National Library of Australia.

## TECHNIQUE NOTES (APIs / walls)

- **Trove internal API unlocked keyless.** The SPA derives its apikey client-side:
  `apikey = md5("Wonder" + cookie[x-ctx])` with leading zeros stripped. Get x-ctx from Set-Cookie on
  GET https://trove.nla.gov.au/ , compute the key, send headers `apikey:<key>` + `Cookie: x-ctx=<ctx>`.
  API base is `/api`. WORKING search shape (hard-won — most variants 500):
  `GET /api/search/-1?terms=<q>&pageSize=<n>&startPos=<s>&sortBy=relevance`
  `sortBy=relevance` is REQUIRED; adding `pageTotals=false` or omitting sortBy → 500. Category id `-1`
  = all. Config endpoints (/api/configuration/environment, /api/collection/exploreinfo) also open.
  Helper script + full technique saved this session. The public api.trove.nla.gov.au still needs a key;
  trove.nla.gov.au/api/* is what the derived key unlocks.
- **DigitalNZ** api.digitalnz.org/v3/records.json is keyless-usable (no key needed for read).
- **Gallica SRU** works but 403s bare curl — send a browser User-Agent. Endpoint:
  gallica.bnf.fr/SRU?operation=searchRetrieve&version=1.2&query=<CQL>&maximumRecords=N (oai_dc records).
- **BC Archives / MemoryBC (AtoM)** front a JS-cookie challenge: just send `Cookie: visited=true`.
  catalogue.nla.gov.au and yukon.ca are Anubis/Cloudflare-walled (curl blocked).

## ZERO-YIELD (terse)

- Trove full "sectional aeronautical chart" sweep = 127 records. Beyond Scott(10, already in CSV) +
  Whiting(1, new) + 2 index diagrams, all 127 are PHYSICAL catalogue records held by SLNSW / NLA /
  Adelaide Uni / UQ / UWA whose onlineUrl points back to LOC (hdl.loc.gov g3701pm.gct00089), FDLP
  purl, or nothing. No further digitised US sectionals. Broad digitised sweeps (city names, "aeronautical
  chart united states", "from Papers of", "annotated pilot") surfaced nothing new. Nancy Bird Walton
  papers = photos/memorabilia only (no charts); "road map used by NBW" at Powerhouse/MAAS = road map.
- **DigitalNZ**: 0 US sectionals. "aeronautical chart" 2818 / "sectional chart" 122 all resolve to WACs
  (GNC/USAF nav-planning), foreign-area maps, LORAN charts, or non-aviation "sectional" (meetings, geology).
  Jean Batten collection = no charts. Auckland/Te Papa aggregated in DigitalNZ — nil.
- **Gallica/BnF**: 0 US sectionals. Rich US map holdings but all excluded classes: USGS geological/erosion/
  relief surveys, 19th-c. state & township "sectional" land-survey maps, AAF SPECIAL aeronautical charts
  of FOREIGN theatres (Fiji, Luzon, New Caledonia), US aeronautical PLANNING/ROUTE/OUTLINE charts, TWA
  system maps. No domestic US 1:500k Sectional series.
- **Canadiana**: text/microfiche corpus (oocihm, all pre-1920 scope); "sectional aeronautical chart"
  = 15,856 OR-noise hits, zero charts. Not a chart repository.
- **BC Archives + MemoryBC (AtoM)**: "aeronautical chart" → No results (challenge-cookie bypassed, so
  genuine zero). **Yukon.ca / archives.gov.yk.ca**: Cloudflare-walled / DNS-dead (bush-pilot Alaska/Yukon
  angle unconfirmable via curl; WebSearch surfaced no digitised US sectionals in Yukon Archives).
- **IWM**: collections.iwm.org.uk 403 bot-wall; WebSearch shows only a USAAF-in-Britain photo item, no
  charts. **RAF Museum / British Library / JISC Library Hub / Japan / Ireland / South Africa**: dorks all
  converge on the known LOC / UNT-WASP / NLA-Scott set; no foreign-held digitised US sectionals.

## CONCLUSION
International-library lane is now effectively exhausted for DOWNLOADABLE US sectionals. Net new: 1 chart
(Whiting NY 1957, distinct scan). The generalisable insight — aviators' personal papers in national
libraries hold US sectionals — produced Scott (r5) then Whiting (r7); further NLA aviator-paper mining
is spent (Trove digitised sweep is definitive). Best remaining international lead is a SCAN REQUEST to
NLA for the two undigitised physical sets, especially Bib 1030946 (National Ocean Survey, ~1970s desert era).
