# LANE K — research-data repositories (round 7)

Verdict: ZERO findings. Lane definitively closed. lane_k_findings.csv = header only.
No researcher has ever deposited scanned/georeferenced US sectional charts in any
DOI-minting or federated research-data repository.

## The master negative (closes most of the lane in one stroke)
DataCite API (api.datacite.org/dois?query=...) sweeps DOI metadata across ALL member
repositories (Zenodo 10.5281, Figshare 10.6084 + institutional, Dryad 10.5061, Mendeley
Data 10.17632, IEEE DataPort 10.21227, OSF 10.17605, Pangaea 10.1594, Harvard Dataverse
10.7910, TDL 10.18738, UNC 10.15139, and every other Dataverse installation):
- "sectional aeronautical" (phrase): **0 records in the entire DOI corpus**
- sectional+aeronautical+chart (AND): 0
- "sectional chart" (101) / "sectional charts" (9): all Mars quadrangle atlas (Zenodo
  Gangale "Atlas of Martian Geography"), cross-sectional medicine, one arXiv LLM/ATC
  paper (current-era charts, no scan deposit)
- "aeronautical chart(s)" (11/20): Swiss-VFR eye-tracking HCI studies (ETH/UZH), chart-
  design papers, FAA/NTL ROSA-P text docs (10.21949 — advisory circulars, known class),
  1 Zenodo geodesy text set. Zero rasters.
- "VFR chart" 1 (Swiss), VFR+chart+FAA 0, aeronautical+georeferenced 0.
This transitively closes the bot-walled Dataverses too (TDL CloudFront-403, UNC auth-only
search API, JHU Cloudflare) — their DOIs would still surface in DataCite.

## Per-repo zero-yields (all queried directly anyway)
- Zenodo API (size cap 25 unauth): "sectional aeronautical chart" 0; "sectional chart" 28
  = Mars atlas only; "aeronautical chart" 3 = 1960s geodesy translations. CLOSED.
- Figshare API (POST /v2/articles/search; quoted phrase honored): "sectional aeronautical
  chart" 0, "aeronautical chart" 0 (unquoted queries return biomed noise = OR tokens).
  CLOSED.
- Harvard Dataverse API: "sectional aeronautical chart" 0, "sectional chart" 0.
  "aeronautical chart" 174 = Harvard Geospatial Library Africa raster set (ONC/TPC/AMS,
  hdl 1902.6/...) + Executive Agreements text DB — ALL excluded classes, no US sectionals
  in HGL. title:sectional 587 = cross-sectional studies. CLOSED.
- Other Dataverses probed live: ASU, Borealis (Canada), UVA, UCLA, QDR Syracuse = 0;
  TDL 403 CloudFront (curl+WebFetch), UNC search API auth-walled, JHU Cloudflare —
  covered via DataCite negative. CLOSED.
- OSF: api.osf.io/v2/nodes filter[title]=aeronautical → 4 irrelevant (CFD classes,
  Indonesian admin theses); /v2/search 502 (dead endpoint); share.osf.io v3
  index-card-search "sectional aeronautical chart" 0; SHARE v2 ES: 2 hits = Zenodo SPAM
  records ("Pilot charts pdf" clickbait). site:osf.io dork 0. CLOSED.
- USGS ScienceBase: "sectional aeronautical chart" 0; "aeronautical chart" 3 = 1955
  Antarctica maps (USS Atka). CLOSED.
- Dryad API: 0 / 0. CLOSED.
- Pangaea (ws.pangaea.de ES): 0 / 0 (valid responses verified). CLOSED.
- Mendeley Data: own API ignores quotes (biomed noise); site:data.mendeley.com dork 0;
  10.17632 DataCite negative. CLOSED.
- IEEE DataPort: site: dork 0 (aero-engine RUL datasets only); 10.21227 DataCite
  negative. CLOSED.
- DataONE (cn.dataone.org solr, federates KNB/EDI/etc.): "sectional aeronautical chart"
  0; title:aeronautical = CA drinking-water assessments + Executive Agreements. CLOSED.
- NOAA NCEI (beyond closed historicalcharts.noaa.gov): OAS accession system is the OCEAN
  archive (cookie-dance needed: -c/-b jar, 302 self-redirect loop otherwise); the one
  "aeronautical" accession hit (0108176) = MTSAT SST ("aeronautical mission" of the
  satellite — false positive). NCEI geoportal ES (metadata/geoportal/elastic + opensearch):
  0. OneStop search API 404 (dead). site:ncei.noaa.gov "sectional aeronautical" = aviation-
  weather AC 00-45E PDFs only. CLOSED (low prior confirmed).
- data.gov catalog API: package_search returns Not Found (API retired/moved) — skipped,
  gov-source classes already closed in prior rounds.
- Academic supplementary-materials angle: 2 broad searches (georeferenced historical
  sectional/aeronautical charts + data-availability phrasing) surface only LibGuides
  pointing at FAA/LOC/NOAA (all closed) and land-use studies using topo maps, not
  sectionals. No paper found that deposited chart scans.

## Technique notes
- Zenodo unauth API caps size=25 (400 otherwise).
- Figshare search DOES honor quoted phrases (0 ≠ error); unquoted = OR-token noise.
- DataCite query= is Lucene-ish: unquoted multi-word = AND, quotes = phrase, no stemming
  (plural checked separately). One query genuinely sweeps every member repo's metadata.
- NCEI OAS: requires cookie jar (Set-Cookie OAS_prd_client + guest_connection, then the
  302 loop resolves). Search box on OAS pages is just search.usa.gov.
- share.osf.io v2 ES index still queryable (sharev2_elastic8 index) but full of Zenodo
  SEO spam; v3 index-card-search is the maintained endpoint.
