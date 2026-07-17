# LANE B — Air Chart Systems (Howie Keefe) atlases & scanned chart-atlas books (round 5)

**Verified findings: 0.** Entire lane is record-only: the Keefe atlas products and every other
chart-atlas book identified are in-copyright commercial compilations that were never digitized
anywhere public. Log everything below so future rounds don't retry.

## Company/product intelligence (established this round)
- Air Chart Systems, Venice/Santa Monica CA, domain **www.airchart.com** (confirmed via AVweb
  review Dec 1998 + shutdown notice). Ceased atlas production **March 2013** ("adverse business
  conditions... electronic charts") after 50+ years (product-lineage photo on trinavcharts.org is
  titled "Atlases-1962-2010.jpg" → atlas line ran from ~1962).
- Product line (11"x11" spiral atlases): **VFR Sectional Atlas** (East + West US editions, annual,
  e.g. "2007-2008 Eastern US" on Amazon ASIN B003K2XBJC), **Aviation Topographic Atlas**
  (Continental US + Bahamas + Baja; "full-color reproductions of all the WACs and Sectionals
  (enhanced to serve as TACs)", 76 two-page spreads of ~150x300nm; ASIN B004HUERDK), **VFR
  Enroute Atlas**, **IFR Atlas** (excluded classes), plus 28-day cumulative update sheets.
  A scanned VFR Sectional Atlas would be dozens of in-era (1970s-2013) sectional panels —
  none exists online.
- Keefe's earlier product: **Sky Prints Aviation Atlas** (Sky Prints Corp., 1960s-80s; Open
  Library records: "Sky Prints Aviation Atlas, 1984", "Sky Prints 1988 Aviation Navigation
  Atlas" — both `no_ebook`). His post-sale venture: **Tri-Nav Charts** (trinavcharts.org,
  ~2010-2017) = IFR low-altitude enroute charts + VFR data → excluded class anyway.

## Wayback CDX sweeps (all zero-yield for full-res sectionals)
- **airchart.com** domain-wide: 904 unique urlkeys, 1997-2013. Content = marketing GIFs/SWFs and
  tiny sample images. Only PDFs: `Hi-Res_Samples/ifr.hi altitude.pdf` (1.9MB, IFR — excluded),
  `Hi-Res_Samples\web.sample.update.pdf` (318KB update sheet), `Pages/pdf/smpl_update.pdf` (55KB),
  update-sheet PDFs (23-48KB). VFR Sectional Atlas sample page images
  (`pages/ac_vfr_dtl/vfr_01a-c,02a.gif`) are 100-150KB web GIFs = crops/thumbnails, rejected.
  No full chart ever hosted. Hi-Res_Samples dir has only those 2 unique captures.
- **trinavcharts.org**: 375 urlkeys, 2011-2017. All small page JPGs (skynav*.jpg 2-450KB promo/
  page-sample photos); largest files are booth/product photos (three.jpg 730KB, twelve.jpg 759KB,
  Atlases-1962-2010.jpg 225KB product-history photo). Product is IFR-enroute-based (excluded)
  and nothing approaches full-res. Zero.
- **trinavcharts.com**: 0 captures. **skyprints.com**: 12 urlkeys — unrelated current site
  (a parking-lot photo), zero. **airchartsystems.com**: 2 captures (parked page, 390B).
- **howiekeefe.com**: 302 → geocities.com/howiekeefe; GeoCities path has a single 410 capture.
  Dead end. **tri-nav.com** = architecture firm; **trinav.com** = TriNav Consultants (Newfoundland
  marine brokerage) — both unrelated (verified on captures).

## Repository/book-site sweeps (all zero for scans; records only)
- **archive.org advancedsearch** (NOTE: bare `q=` now maps to FULL-TEXT search only — must use
  explicit metadata fields like `title:()`, `creator:()`, `publisher:()`): zero hits for
  title:("air chart"), title:(airchart), creator/publisher Keefe/"Air Chart"/"Sky Prints",
  title:("sky prints"), title:("sectional atlas"|"VFR atlas"|"atlas of sectional") (only medical
  sectional-anatomy atlases), title:("aviation atlas") (Smithsonian history book only),
  title:("aeronautical")+title:(atlas) (Rumsey Geographia world atlas pages), title:("chart
  atlas") (1992 "Final Approach: Electronic Approach Chart Atlas" software = IFR approach plates,
  excluded), title:(sectional)+mediatype:texts (medical), title:("flight guide") (nothing
  aviation-chart), creator:(AOPA/"Aircraft Owners and Pilots Association") (podcast audio only,
  no Aviation USA scans), full-text `text:("air chart systems")` and `text:("sky prints")` = 0.
- **Google Books API** (googleapis.com/books/v1): **HTTP 429 all day 2026-07-08** — anonymous
  shared-project daily quota exhausted (project 624717413613); country=US param doesn't bypass.
  Fallback WebSearch restricted to books.google.com: no Air Chart Systems / Sky Prints / sectional-
  atlas volumes indexed at all (only FAA Chart User's Guides, testing supplements = excerpt
  crops, excluded). → No full-view digitization exists.
- **HathiTrust**: catalog.hathitrust.org now 403s curl even with browser UA (add to bot-wall
  list). Domain-filtered WebSearch over hathitrust.org: no Air Chart / Sky Prints records
  surface; consistent with Part-3 verdict (HathiTrust has no chart-plate holdings).
- **Open Library**: records only, all `no_ebook`: "Air Chart Systems" (1996), "Sky Prints
  Aviation Atlas, 1984", "Sky Prints 1988 Aviation Navigation Atlas". Record-only.
- **Doc-scan sites** (dokumen.pub, vdoc.pub, manualslib, yumpu, scribd via WebSearch): no ACS or
  Sky Prints uploads; only medical sectional-anatomy atlases and an INDOAVIS (Indonesian) SAC
  doc (not US). Scribd hosts current FAA legend excerpts only.
- **LOC re-check**: a search-snippet mention of a "handmade aeronautical flight composite atlas
  (atlas factice) of sectional charts" resolves to the gct00089 set description itself — nothing
  new; maps searches "atlas factice aeronautical" / "composite atlas sectional" = 0. LOC stays
  closed.
- **Howie Keefe personal collections**: no museum/library holds his papers per searches (his
  P-51 "Miss America" material is scattered: Oklahoma Museum of Flying page, book "Galloping on
  Wings" — not digitized, title search on archive.org = 0). SDASM Flickr already closed in
  Part 3. vansairforce.net thread 403-walled to WebFetch (forum chatter, no scan links per
  search snippets).

## Other atlas-book products triaged (none reproduce full sectionals / none scanned)
- **AOPA "Aviation USA"**: airport-directory format, no full sectional reproductions; no scans
  on archive.org (creator:AOPA = podcasts).
- **"Flight Guide"** (Airguide Publications): airport diagrams only → excluded class; nothing
  scanned anyway.
- **Sporty's / ASA**: no atlas product reproducing sectionals identified (only chart CD-ROMs →
  Lane A territory / handoff lead #1).
- **Foreign reprints of US sectionals**: nothing found (searches return LOC/FAA only).
- Nautical "chart atlas" products (Maryland Nautical / Paradise Cay etc.) = marine, excluded.

## Leads for future rounds
1. **Physical-copy acquisition is the only route** to Air Chart Systems VFR Sectional Atlases
   (used copies cheap on Amazon/Alibris/eBay: ASINs B003K2XBJC East 2007-08, B004HUERDK Topo).
   One atlas ≈ full East- or West-US sectional coverage for its year at print resolution —
   arguably the highest-density 1990s-2000s recovery object that exists. Editions ran ~1962-2013.
2. Google Books API retry on a later day (quota was exhausted today, possibly by parallel lanes)
   — though books.google.com web index suggests the titles simply aren't digitized.
3. vansairforce.net thread 2910 ("Howie Keefe's Air Chart System") — manual browser read for
   possible scan links (403 to tools).
4. Keefe estate/family (d. 2011?) — his Tri-Nav site persisted to ~2017; no archival donation
   traced. Low probability.
