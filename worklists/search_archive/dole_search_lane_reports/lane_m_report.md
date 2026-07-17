# LANE M report — round 7: professional map dealers + Russian/EE map hoards + WARC mirrors

45 verified findings in `scratchpad/lane_m_findings.csv` (all URLs curl-verified 200 in a
single final pass). Deduped against master_dole_v2.csv and missing_from_dole_online.csv
(zero hits for raremaps/googleapis-raremaps/curtiswright in either).

## POSITIVE 1 — raremaps.com (Barry Lawrence Ruderman) — THE LANE FIND (43 rows)

**The unlock:** raremaps.com itself is double-walled (Cloudflare vs curl AND WebFetch 403),
but every listing's full scan lives on a **public Google Cloud Storage bucket with no
referer/auth check**:
- single-file: `https://storage.googleapis.com/raremaps/img/{small,large,xlarge}/<id>.jpg`
  (xlarge is modest, 0.5–7 MB)
- **native scan: `https://storage.googleapis.com/raremaps/img/dzi/img_<id>.dzi`** (OpenSeadragon
  Deep Zoom; TileSize 254, Overlap 1, jpg) with tiles at
  `img_<id>_files/<level>/<col>_<row>.jpg`. Bucket object-GET is anonymous; object-LIST denied.
  Versos, where scanned, are `img_<id>_1.dzi`.
- Tile fetchability verified end-to-end (level-15 corner tiles 200 on item 84827), so per the
  deep-zoom rule these count as findings, not record-only.
- Native sizes are archival: **~20,000–33,000 px on the long side** (~400+ dpi on a 47x20.5in
  sheet) — higher than most LOC gct00089 masters.

**Enumeration method** (site search is walled): Wayback CDX of
`raremaps.com/gallery/detail/*` filtered on slug keywords `sectional`/`aeronatical` →
178 captured URLs → 111 unique ids → 42 sectional-aeronautical items; titles +
"Publication Place / Date" + sheet dims parsed from wayback captures of each listing.
IDs: 46941, 55364, 55379, 55403, 55415, 55453, 55476, 55526, 55852, 55911, 56071, 56455,
56617, 56634, 56655, 57793, 57852, 57879, 57906, 57979, 58020, 58044, 58199, 58242, 58345,
68461, 68496, 68581, 68612, 69172, 69693, 70376, 70977, 81354, 81355, 83757, 84100, 84827,
95104, 95759, 97173, 98748 (+78543 Alaska, below).

Highlights:
- **Los Angeles 1976 (id 70376), recto 32926x12329 + verso 32929px — squarely in the
  1970–2010 desert**, with the verso scanned (dole-relevant half-sheet era format 55x20.5in).
- 1943–45 wartime **[Restricted] military-overprint editions** for ~30 cities (New Orleans,
  Shreveport, El Paso, Winston-Salem, Nashville, Chattanooga, Charlotte, Birmingham, Mobile,
  Bellingham, Boise, Cheyenne, Mt. Whitney, Portland, Seattle, Yellowstone Park, San Antonio,
  Oklahoma City, Little Rock x2, Corpus Christi, Beaumont, Sioux City, Savannah, Miami/South
  Florida x5, Austin, Prescott, Phoenix Q-3) — distinct dealer scans; many dole 1943-45
  holdings are civil editions, and the notes say which are duplicates-by-edition.
- Phoenix "Edition of October 1935" (83757, 28865x14080 + verso).
- Texas-Louisiana-Mississippi Restricted sectional (68612, 1943, 23954x13711) — an odd
  regional-format sheet worth eyeballing at download time.
- San Diego Q-2 run: 1943, Mar-2-1944 (listing pub 1948 — check on image), Mar-8-1945,
  June 1948 (46941 is the one LOW-RES outlier, 3516px).
- **Pt. Barrow Alaska Aug 1941** (78543, 18964x17022) — USC&GS Alaska 1:500k series
  (sectional lineage); flagged CAVEAT in CSV for coordinator judgment.
- 42 of 43 items ≥17,000 px long side. 10 items have scanned versos (7000px class).

**Intel for future rounds:** ANY raremaps item id → full native scan via the GCS pattern
above, no wall. If new sectional stock appears, only the id is needed (Google dork or CDX).
Non-sectional aeronautical stock checked and excluded: 54 ids = WACs, Regional (11M/15M/17M),
AAF radar charts, DF charts, AAF long-range cloth charts, NASA/ACIC space charts.

## POSITIVE 2 — curtiswrightmaps.com (2 rows)

Dallas 1942 and Austin 1942 restricted-era sectionals, dealer scans 10000x5027 / 10000x4883 px
(3.5/2.9 MB). WordPress trick: strip `-scaled` from the displayed image filename to get the
full-res original in wp-content/uploads. Full product-catalog sweep for
sectional/aeronautical: nothing else in scope (AAF 1:1M river charts, Regional PNW, Aleutian
flight charts, world airways only).

## Zero-yield / closed (lane A dealers)

- oldworldauctions.com: archive search is a React SPA (state blob, no SSR results); Google
  dorks "sectional aeronautical" + "aeronautical" → only AAF cloth charts, 1920s landing-field
  maps, Coast Survey letters. No US sectionals ever auctioned there. Closed.
- swanngalleries.com, pbagalleries.com: dork zero (ephemera class absent). Closed.
- neatlinemaps.com: Cloudflare vs curl; Google-index negative for sectionals. Closed.
- georgeglazer.com, bostonraremaps.com, mapsofantiquity.com, murrayhudson.com,
  antiquemapsandprints.com, oldmapgallery.com: dork zero. Closed.
- oldimprints.com: searchResults.php endpoint now 404 (platform gone); Google-index negative.
  Closed.
- liveauctioneers.com / invaluable.com: dork zero for "sectional aeronautical chart"; class is
  listing-photo (eBay-equivalent, watermark/low-res) anyway. Closed.

## Zero-yield / closed (lane B — Russian/Eastern-European hoards)

- loadmap.net: JS-challenge (JWT redirect → survey-smiles.com) resists curl cookie-dance;
  Google index of the domain shows topo/AMS sheet DB only, no aeronautical class. Closed
  (manual-browser candidate only if someone insists; prior low — DB is topographic).
- mapstor.com (same ecosystem): topo series only. poehali.org/poehali.net: forum + Soviet GS
  topo archive, no US aeronautical. retromap.ru: city plans/RKKA/aerial. meridian.com.ua: zero.
  All closed via dorks.
- radioscanner.ru/files/aviamaps/ (surfaced by Russian-language search): 205-file category
  enumerated (cp1251) — Ukrainian VFR 1:500k sets 2008-2012, RF charts, frequency data; the
  only US-made items are ONC-class E.Europe/Asia charts and a DoD airfield directory. No US
  sectionals. Closed.
- avsim.su (Russian flight-sim library — the plausible US-chart-pack host): server unreachable
  (connection fails, no DNS/TCP response). Record as dead/unprobeable this round.
- Russian-language WebSearch passes ("аэронавигационная карта США 1:500000 скачать",
  "сканы карт" + forum terms): all roads lead to FAA/LOC/skyvector/usahas (the round-5 find
  is even indexed on runet). aviaforum.ru thread 13266 = RF/CIS chart legality talk, no US
  scans. No runet hoard of US sectionals exists in the indexed web. Closed.

## Zero-yield (lane C — archive.org WARC/mirror probes)

advancedsearch: identifier:(glideplan*|chartgeek*|foxfam*|fox-fam*|gelib*|nlneilson*) → only
Yiddish books and Fox Family TV; title:(warc)+aviation/sectional → ArchiveTeam single-site
WARCs of unrelated aviation sites; "sectional chart" warc → 0; title:"site mirror" → 0.
Confirms Part-7 lane-G verdict: those binaries are archive-extinct. Closed.

## Record-only / leads

- raremaps GCS pattern (see intel above) — reusable extraction recipe.
- 46941 San Diego June-1948: only low-res scan in the raremaps set (3516px) — included in CSV
  with LOW-RES flag; drop if quality floor is enforced strictly.
- 56455 San Diego "March 2, 1944" title vs listing pub-date 1948 — verify edition on image at
  download time (raremaps Dallas-style title/date permutation seen before at NARA).
- Miami/South Florida x5 and New Orleans x2 / Little Rock x2 raremaps listings may be
  relistings of the same physical sheets (dims match within an inch) — but the scan files are
  distinct ids; download all, dedupe visually.
- Download-phase note: full DZI reassembly ≈ (W/254+1)x(H/254+1) tiles ≈ 6,700 tiles for a
  29000x14700 sheet; or take the max level only. Plan ~290k tiles for all 43 items, or
  xlarge jpgs (0.5-7MB) as quick placeholders.
