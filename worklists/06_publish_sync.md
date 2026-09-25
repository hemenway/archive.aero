# Worklist 06 — Publication state and reconciliation

## Current status — 2026-09-21

The **July 14 → July 21 sync is complete**. Subsequent publication batches have
superseded its 3,658-era count and repaired its partial late-2025/early-2026 ranges.
The current checked-in `dates.csv` has **3,751 entries** (unchanged by the
2026-09-25 Boston republish, which rewrote 26 existing keys in place); the viewer
config source `src/viewer.js` points to **`metadata-83ed9b48.bundle`**
(2026-09-25; the generated `assets/viewer.*.js` follows it).
The latest publication is the 2026-09-25 Boston 1953–70 bottom-latitude fix
(26 eras + 26 chart artifacts, same keys, see [04](04_georef_backlog.md)); the
latest `dates.csv`-changing commit is `d1034df` (2026-09-15).

Local `data/chart_pmtiles/uploads.jsonl` has **7,475 upload events / 7,342 unique
chart keys**, latest event `2026-09-15T22:07:25`. Events include overwrites;
unique keys are not a count of currently selected editions. These are repository
and ledger observations, **not a fresh R2/reachability audit**.

- [ ] Reconcile the **nine August 29 ACASIS additions**: recovery/build evidence
  exists, but the available upload ledger and September 15 timeline do not prove
  publication. See [11](11_acasis_img_recovery.md) and [12](12_sdcard_batch.md).
- [ ] Resolve the remaining candidate/georef issues before publishing their
  affected eras (04): Boston 1957 (`ca000440r`, the one 1953–70 Boston row not
  covered by the 2026-09-25 fix), SF 1971 half grouping, Juneau 2013 hold
  conflict, Dallas 1981 north and the Denver 1975 seam. Stored GCPs alone do
  not close them.
- [ ] Reconcile source availability from [02](02_disk_vs_dole.md) before rebuilds;
  142 resolver misses do not by themselves imply a live-site outage.
- [ ] For the next batch, identify the current output/mirror location. The July
  report's `/Volumes/drive/pmtiles` is **not present** on this machine at this audit.

## Publication procedure

1. Verify one affected chart end-to-end and inspect the rendered warp before a
   batch. Keep mosaic compression LZW/DEFLATE; the Go PMTiles reader cannot read ZSTD.
2. Convert/upload the verified era artifacts, then update their `dates.csv`
   references. Per-chart full-sheet artifacts use permanent extensionless keys;
   `scripts/publish_chart_pmtiles.py` publishes with copy-to, never sync.
3. Rebuild and publish the metadata bundle after PMTiles changes. Rebuild and
   upload `timeline_data.json` after inventory/upload-ledger changes, then rebuild
   tracked `coverage.json`. Use current scripts and `CLAUDE.md`; the viewer config
   moved from inline `index.html` to `src/viewer.js` in September; the hashed
   `assets/` bundle is generated output.
4. Verify that referenced artifacts and metadata are reachable and describe the
   same batch. Existing local caches and upload events are not fresh remote checks.
   Serve large PMTiles through the `tiles` Worker (>512 MB bypasses raw R2 cache).
5. Follow [URI-POLICY.md](../URI-POLICY.md): **retain published URIs**, including
   old era names no longer in `dates.csv`. The old suggestion to delete 209 stale
   R2 keys is retired. Keep historical metadata bundles for analytics offset decoding.

## Publication history relevant to the original gap

| Date | Recorded result |
|---|---|
| 2026-07-21 (`f08a5fb`) | Full July-run publish: 3,658 eras (3,656 new-run keys + 2 retained fallbacks), first metadata bundle; 7 empty sparse mosaics skipped; 23 UInt16 mosaics rescaled. |
| 2026-07-23 (`5719624`) | 14 broken ranges repaired and republished after sibling-georef and empty-mosaic fixes. |
| 2026-07-29 (`bfa31e5`) | July 26 full run published: 3,698 eras, including 2004–05 additions. |
| 2026-08-19 (`10c5328`, `ff91faf`) | September 3 FAA cycle, 10 Juneau eras and iFly-card 2013 editions published. |
| 2026-08-26/27 (`406260d`, `eca3d57`, `15e33ab`) | iFly build-machine/SD-card editions and Dallas WASP south/1982 halves published. |
| 2026-09-01 (`bb598f6`) | Sarangan donation published: Cincinnati 1991 and Detroit 1971. |
| 2026-09-15 (`d1034df`) | 11 georef-tool eras, 13 chart artifacts; current `dates.csv` count 3,751. |
| 2026-09-25 | Boston 1953-12 → 1970 bottom-latitude fix: 26 era keys and 26 `chart/boston_ma/<date>` artifacts republished in place (0.99 → 1.19 GB eras, 364 MB charts), bundle `metadata-83ed9b48`; `dates.csv` unchanged. Same-key republish, so the Worker's per-block edge cache can serve stale blocks for ≤24 h at colos other than the one refreshed by the post-upload no-cache pass. |

## Historical July 16 comparison (closed)

The original audit compared **3,666 mosaics** in
`/Volumes/projects/2026-07-14 slicer run/` with **3,390** then-published entries.
[data/publish_sync.csv](data/publish_sync.csv) preserves that snapshot:
**485 new unpublished keys**, **209 old-only keys** (189 bare dates, 20 changed
ranges). July 21 conversion used webp q80/native zooms; 23 UInt16 mosaics needed
`-rescale linear -rescale-range 0,65535 -alpha-band 4`. The late-2025/early-2026
partial-range warning belonged to that run and was followed by the July 23 repair.
Do not rerun the old 485-item queue or compare current publishing against that
single historical output directory.
