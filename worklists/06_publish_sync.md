# Worklist 06 — Publication state and reconciliation

## Current status — 2026-10-01

**[October 1 georef batch](georef_publish_2026-10-01.md) complete:** 17 era
artifacts and 30 full-sheet chart artifacts uploaded, R2 size-checked and
read back through every CDN block. All 23 donated sectional faces are published.
`dates.csv` now has **3,761 references**; viewer metadata is
**`metadata-1c04feec.bundle`**. Inventory and coverage were regenerated after merging
the verified chart upload events. Permanent old object keys are retained.

The separate September 29 modern-era reslice/republish job continues; its logs
under `/Volumes/projects/2026-09-29 g2p yshift reslice/` are the evidence for that
queue. It may supersede the bundle above while preserving this batch's keys.

- [ ] Reconcile the **nine August 29 ACASIS additions** against current remote
  objects and upload evidence (11/12); this batch does not close that audit.
- [ ] Resolve remaining georef/candidate issues in 04: Boston 1957, SF 1971,
  Juneau's hold, Dallas 1981 north and Denver 1975, plus the remaining GCP lanes.
- [ ] Reconcile the September 21 source audit from 02 separately. All sources
  required by this batch were present; that does not prove every catalog source.

Current batch output: `/Volumes/drive/georef_publish_2026-10-01/`; row-level,
upload and read-back evidence: `worklists/data/georef_publish_2026-10-01/`.
The former `/Volumes/drive/pmtiles` mirror remains absent; use actual run paths.

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
| 2026-10-01 | 17 georef-tool/donation eras + 30 chart artifacts verified and published; see the completed batch report. |
| 2026-07-21 (`f08a5fb`) | Full July-run publish: 3,658 eras (3,656 new-run keys + 2 retained fallbacks), first metadata bundle; 7 empty sparse mosaics skipped; 23 UInt16 mosaics rescaled. |
| 2026-07-23 (`5719624`) | 14 broken ranges repaired and republished after sibling-georef and empty-mosaic fixes. |
| 2026-07-29 (`bfa31e5`) | July 26 full run published: 3,698 eras, including 2004–05 additions. |
| 2026-08-19 (`10c5328`, `ff91faf`) | September 3 FAA cycle, 10 Juneau eras and iFly-card 2013 editions published. |
| 2026-08-26/27 (`406260d`, `eca3d57`, `15e33ab`) | iFly build-machine/SD-card editions and Dallas WASP south/1982 halves published. |
| 2026-09-01 (`bb598f6`) | Sarangan donation published: Cincinnati 1991 and Detroit 1971. |
| 2026-09-15 (`d1034df`) | 11 georef-tool eras, 13 chart artifacts; current `dates.csv` count 3,751. |
| 2026-09-25 | Boston 1953-12 → 1970 bottom-latitude fix: 26 era keys and 26 `chart/boston_ma/<date>` artifacts republished in place (0.99 → 1.19 GB eras, 364 MB charts), bundle `metadata-83ed9b48`; `dates.csv` unchanged. Same-key republish, so the Worker's per-block edge cache can serve stale blocks for ≤24 h at colos other than the one refreshed by the post-upload no-cache pass. |
| 2026-09-30 → 10-02 (`92dbb36` … `8238d0e`) | 18 modern eras `2022-05-19` → `2024-12-26` republished in place, one commit + bundle each (last `metadata-c0dd24c1`). See the y-shift section below. |

## 2026-09-30 modern-era y-shift republish (closed)

The 18 era objects still in R2 from the 2026-07-14 run (uploaded 07-18 → 07-21)
had been converted by a geotiff2pmtiles older than upstream `aa74aa0`, which
read rows with the X pixel size. The slicer's mosaic takes the finest x-res and
y-res independently, so modern mosaics were slightly non-square
(2022-05-19: y/x − 1 = 8.17e-5) and those archives drew the charts north by up
to ~17 px at z12 (~500 m in the southern US; ~1 px on 2024-09-05 → 12-26). The
07-28 republish had kept them because their *mosaics* matched — the converted
tiles were never compared. Proof: same mosaic strip, pre-fix g2p +16.88 px,
fixed g2p +0.14 px.

Re-sliced with the 2026-09-30 slicer (lanczos, square z12 tile-grid mosaic) and
converted with **b300c9e + a half-pixel sampling fix**: stock g2p puts source
pixel k's centre at k instead of k + 0.5 (0.5 px NW offset plus a half-pixel
blur; DFW z12 PSNR vs mosaic 21.6 → 29.2 dB). Patch, binary and sha256 live in
`/Volumes/projects/2026-09-29 g2p yshift reslice/bin/`; the binary runs from
`~/Library/Caches/archive.aero/geotiff2pmtiles/b300c9e-halfpx/` (the SMB share
is noexec). Not upstream yet. Every era: tiles vs mosaic ≤ 0.03 px over 12
probes; the live bundle byte-matches R2 for all 18; 2022-12-29 and 2023-02-23
went out once on stock b300c9e first and were republished on the patched build.

Each era was published as soon as it passed (bundle built from the local file
before upload → bundle → era → commit `index.html`/`src/viewer.js`/`assets` →
push), so the new-bytes/old-bundle window was the push plus the Pages deploy.
Run dir keeps `mosaics/` (592 GB), `pmtiles/` (60 GB, = live), and two
superseded sets (old-slicer, stock-converter) for rollback.

Still open: the other ~3,740 eras carry the stock half-pixel offset
(≈0.2–0.3 z12 px on 22.7 m mosaics) until reconverted; upstream the g2p fix.

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
