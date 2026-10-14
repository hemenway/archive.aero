# 18 — Apex cutover and production rebuild

## Current production — 2026-10-13 (evening, eighties release)

Twelve 1985–1992 eras and 21 full-sheet artifacts added (worklist 04 entry of the
same date). Pointers after this release:

- Production manifest: `next/manifest.229990827e88.json` (3,774 eras).
- Chart index: `next/charts.ae4a901ac334/`; airport records `next/airports.419050ae8e8d/`;
  pin shards `next/pins.83c597a524d8/`.
- Metadata bundle: `sectionals/metadata-9a94f060.bundle`; `dates.csv` 3,774 keys.
- Site Worker `archive-aero-site` version `4f821a9c-a7a0-4d2a-af12-601a9d704171`; tiles Worker
  version `c434866c-a051-4739-82eb-69fa1634f462` with `CHART_CACHE_VERSION="2026-10-13-eighties"`.
  Published 2026-10-13 21:43 (commit `11d662f`); live checks: manifest, bundle, timeline,
  a CG-18 tile and the CG-18 edition page all 200.
- Release inputs and evidence: `~/archive-next-build/2026-10-13-eighties/`
  (`work/upload-plan.json`, `publication/*.log`, `publication/verified-objects.jsonl`).
- `scripts/next_release_run.py` now does the plan/upload/permanent/stage phases of
  "Repeat the production data build" for a slicerd run; the overlay, card,
  airport, manifest, shell and bundle steps below are unchanged.

## Previous production — 2026-10-13 (georef release)

The georef and recovered-history airspace follow-up is published after explicit
approval. All 297 planned objects passed full-byte storage readback; public
served-byte checks passed for 23 archives (three sampled ranges each), four
cards, the bundle, manifest and timeline. **All ten live browser views and all 50 airport page/JSON-LD checks passed.** The October 12 cutover record below is historical; these are the
current pointers:

- Site Worker: `archive-aero-site`, version
  `4ef92237-747b-41e6-94e0-bb1afd3d94d8`.
- Tiles Worker: `tiles`, version `16d60293-72c5-4b7d-b755-cb982f5e77e2`, with
  `CHART_CACHE_VERSION="2026-10-13-georef"`. The revision now covers permanent
  era archives as well as chart keys/cards and the timeline, for block and
  whole-response caches. Hashed archives keep their existing cache entries.
- Production manifest: `next/manifest.3399e551dd77.json` (3,762 eras).
- Chart index: `next/charts.9a7e9403a5f5/` (150 records).
- Airport records: `next/airports.dae9aa908458/` (50 airports plus index).
- Pin shards: `next/pins.b195a9d65c01/` (66 shards).
- Airspace: `airspace/class-20261013.cce35551b42d.pmtiles`, with 153 regional
  cycles including 116 US cycles back to 2017-10-12.
- Airfields: the existing `next/airfields.3561417ba096.bin` and `.json`
  reference is preserved. Jev acceptance corrections remain open in worklist 16.
- Metadata bundle: `sectionals/metadata-61e8c828.bundle`; the legacy viewer
  points to it and to the new hashed airspace archive.
- Sitemaps: 7,121 unique edition URLs and 50 airport URLs. Four edition cards
  were generated for this release.

Six eras were rebuilt: Boston 1957 is newly present; SF 1971, Dallas 1981 and
Dutch Harbor 2004–2005 have changed hashes; the other two 2004 eras are
byte-identical rebuilds. All six permanent era addresses and five permanent
chart addresses were published alongside their immutable copies. Release inputs,
artifacts and evidence are retained at `~/archive-next-build/2026-10-13-georef/`,
including `release.json`, `publication-plan.json`,
`publication/verified-objects.jsonl`, `publication/served-byte-checks.json` and
deployment logs. The persistent `~/archive-next-build/prod/` build inputs are
refreshed from that verified release while preserving old assets. The promoted
stage rebuilds the reviewed manifest byte for byte; candidate symlinks remain
valid, and backups preserve the prior records and four retired local stubs.

## Historical cutover — 2026-10-12

The apex shell and airport/edition pages are deployed. `beta.archive.aero`
returns 410 for every path, including static assets. There is no `/classic/`
viewer. All 7,306 stale permanent chart keys were refreshed, including the two
2016 repairs below. The final report has **zero stale, ready or pending keys**;
7,417 keys have upload receipts. The 20 documented exceptions remain untouched
(Boston 1957 held, 10 retired and 9 renamed inset addresses).

- Site Worker: `archive-aero-site`, route `archive.aero/*`, version
  `32e27843-d1b2-4411-a8bf-567d2029c2d7`.
- Tiles Worker: `tiles`, version `acb3c192-aae2-4c87-91c9-1485b314ef08`,
  deployed after the completed refresh and timeline upload with
  `CHART_CACHE_VERSION="2026-10-12-reslice"`.
- The site Worker binds `DATA` to `tiles`. Server-rendered chart and airport
  records must use that service binding: public same-zone requests bypassed
  the tiles route and returned 404 from its R2 origin.
- Beta Worker: `archive-aero-beta`, version
  `8d64267d-e43d-4e18-ada4-2f4e5190cd90`; `RETIRED="1"` and
  `assets.run_worker_first: true`. The rebuild/retirement guide is `next/BETA.md`.
- Production manifest: `next/manifest.56cf6d2fc143.json`.
- Chart index: `next/charts.0e4fcec7abc2/` (150 records).
- Airport records: `next/airports.acf9bdccccc6/` (50 airports plus index).
- Pin shards: `next/pins.a4d88f6bc08b/`.
- Cards: **7,120 JPEGs, every eligible edition page**. The earlier prepared
  timeline omitted 52 newly published editions; their cards and the two
  repaired 2016 cards were generated during the cutover.
- Sitemaps: 7,120 unique edition URLs and 50 airport URLs. Alternate end-date
  records share a canonical edition address and are deduplicated.
- Metadata bundle: `sectionals/metadata-16bcc4d8.bundle` (3,761 eras), uploaded;
  the origin viewer's fingerprinted build points to it.

The reslice's hashed archives remain in `charts-beta`, production's immutable
bucket. `tiles` routes hashed keys and `next/*` to `BUCKET_HASHED`. `charts`
retains permanent chart keys, cards, timeline JSON, metadata bundles and the
older era archives. No bulk bucket copy was performed.

## The two 2016 repairs

Denver 2016-01-07 and San Francisco 2016-08-18 had malformed NARA GeoTIFF
transforms (~0.0000847 metres/pixel). Their original FAA ZIPs already on disk
have ~42.33 metres/pixel, matching raster dimensions and matching sampled
pixels. The two malformed duplicate catalog rows were retired after backing
up the catalog; surviving ZIP rows carry provenance notes. Raw sources were
not modified. Full-sheet warps were visually inspected before publication.

New hashed objects, also published to their existing permanent addresses:

- `sectionals/chart/denver/2016-01-07.82dba6cf01b8.pmtiles`
- `sectionals/chart/san_francisco/2016-08-18.3ccdcecdef23.pmtiles`

Both are z0–11 with normal geographic bounds. Corrected records supersede
the bad records in the reslice's append-only hashed plan. Rebuild evidence and
sources are under `~/archive-next-build/repair-2016/`. This repairs full-sheet
artifacts/cards; the older mosaic omissions remain in worklist 01.

## Verification and evidence

`worklists/data/apex_cutover_2026-10-12/` contains publication logs, pre-refresh
receipt backup, repair plan, browser screenshots, all-airport HTTP checks and
test logs. Canonical chart receipts remain in
`worklists/data/chart_pmtiles/uploads.jsonl`.

Passed: 93 stub browser tests, 8 real-module browser tests, 75 Worker tests,
21 page tests, 71 data-plane tests, and the Python publication/data/airport
builder suites. Production browser checks covered root, Boston airport and
edition pages, both repaired 2016 charts, and the Denver airport on mobile;
they rendered tiles without console errors or failed requests. All 50 airport
pages returned 200 with Airport JSON-LD. The metadata bundle passed 25 remote
era byte comparisons. The final timeline response matches the local file byte
for byte; sampled permanent archives return matching sizes and repaired bounds.

The publisher now supports `--workers`, forces replacement even when size and
mtime match, requires matching SHA-256 metadata, and flushes/fsyncs each
verified receipt. Interrupted batches resume from their last receipt.

`CHART_CACHE_VERSION` in `worker/wrangler.toml` versions the edge-cache entries
for permanent chart keys/cards, era archives and `timeline_data.json`, including both block
and whole-response caches. Bump it **after** a replacement batch and timeline
upload, then deploy `tiles`. Hashed archives keep their existing cache entries.
Browser caches of plain addresses retain their previously issued HTTP TTL;
the new viewer uses the immutable hashed addresses.

## Repeat the production data build

Persistent build inputs live under `~/archive-next-build/prod/`, replacing
the handoff's temporary Codex workspace path:

- `stage/`: sparse metadata stubs for the chosen era archives, basemap/airspace.
- `work/plan.json`: latest uploaded records from the reslice hashed plan.
- `work/timeline.json`: current timeline, with chart links restricted to the
  artifacts present in that plan.
- `next/`: generated immutable manifest, indexes, cards references and sitemaps.

When publication/catalog state changes, regenerate `timeline_data.json` and
`coverage.json`, refresh the staged plan/timeline, then build overlays:

```bash
~/venv/bin/python scripts/next_build_overlays.py --airfields airfields.json --timeline ~/archive-next-build/prod/work/timeline.json --plan ~/archive-next-build/prod/work/plan.json --out ~/archive-next-build/prod/next --og-manifest /Volumes/projects/archive-next/og/manifest.jsonl --sitemap https://archive.aero
```

Render any missing cards with `scripts/next_og_images.py --index <new charts
folder> --out /Volumes/projects/archive-next/og`, then repeat the overlays build
so it includes those cards. The renderer defaults to the production tiles host.
Build airports with `scripts/next_build_airports.py --charts <new charts folder>
--timeline timeline_data.json --out ~/archive-next-build/prod/next`.

```bash
~/venv/bin/python scripts/next_build_manifest.py --dir ~/archive-next-build/prod/stage --coverage coverage.json --overlays ~/archive-next-build/prod/next/overlays.json --airports ~/archive-next-build/prod/next/airports.json --out ~/archive-next-build/prod/next
```

Upload cards to `r2:charts/sectionals/`; upload the new pins, charts, airports
folders and manifest to `r2:charts-beta/next/`. Upload data **before** the shell.
Then rebuild with the emitted manifest name (the current example follows):

```bash
node next/build.mjs --manifest https://data.archive.aero/next/manifest.3399e551dd77.json --manifest-file ~/archive-next-build/prod/next/manifest.3399e551dd77.json --base / --outdir next/site/dist --sitemap ~/archive-next-build/prod/next/sitemap-sectionals.xml --sitemap ~/archive-next-build/prod/next/sitemap-airports.xml
cd next
npm ci --ignore-scripts
npm run deploy:site
```

The real build is 51,087 bytes of critical gzipped JavaScript against a 51,200
byte budget. New shell functionality belongs in the lazy cards chunk.

## Permanent chart refresh

```bash
~/venv/bin/python scripts/publish_chart_pmtiles_from_reslice.py --refresh --copy --workers 16
~/venv/bin/python scripts/build_timeline_data.py
~/venv/bin/python scripts/build_coverage.py
```

Upload `timeline_data.json` to `charts/sectionals/timeline_data.json`, bump the
chart cache revision and deploy `tiles`. A final report without `--copy` must
show zero ready/stale keys. Boston 1957 was published October 13; retire its
hold after the final live Boston check. The 10 retired and 9 renamed inset
addresses are preserved.

## Rollback and later retirement

Removing the `archive-aero-site` route restores the origin viewer. Keep its
assets and old era archives through the rollback window; metadata bundles also
remain necessary for historical analytics. Do not delete `charts-beta`: it is
production storage. Beta stays behind 410 for the retirement window; removing
its Worker/domain later is separate from deleting any data bucket.
