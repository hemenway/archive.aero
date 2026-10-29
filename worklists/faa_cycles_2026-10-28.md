# FAA VFR cycles — 2026-10-28

## Publication completed — 2026-10-29

Both requested cycles are archived and verified. September was already published;
the missing October 29 cycle and all 57 full-sheet editions are now live.
NUC slice, conversion, alignment and archive jobs all succeeded. Alignment passed
12 probes, worst shift 0.00 px at 0.3 px tolerance. The mosaic has 7,529,437
addressed tiles, z0–12, and is 3,810,139,589 bytes.

- Production manifest: `next/manifest.6e6525010d79.json` (3,775 eras).
- New mosaic: `sectionals/2026-10-29_to_2026-12-24.5e74978c9db9.pmtiles`.
- Chart index: `next/charts.785ce80026fe/`; pins: `next/pins.2be1b8a3451a/`;
  airport records: `next/airports.3c6f68e146c1/`.
- Legacy bundle: `sectionals/metadata-5ca3db1f.bundle` (20,703,745 bytes);
  SHA-256 `5ca3db1f5269a80087c7758fe3bf5919da540323c20236c12f746e65553335f5`.
- Site Worker version: `2638c1e2-8ed4-457a-b81d-b8e269990e40`.
- Tiles Worker version: `4d9b9c63-c916-47fb-8289-6d7f9fd7c033`,
  cache revision `2026-10-29-faa-cycle`.
- All 58 archives (5,343,110,736 bytes) passed full-byte SHA-256 storage readback.
  Hashed production and permanent keys are both uploaded, with receipts.
- All 336 metadata/card objects passed full-byte readback. All 57 new cards
  rendered without failed or empty tiles (2,495 tile requests).
- Public checks: 348 archive ranges, eight geographically diverse mosaic tiles,
  and exact manifest/bundle/timeline bytes passed.
- Live browser: October 29 Albuquerque mosaic and full-sheet edition render
  correctly; date/edition metadata correct and no console warnings or errors.
- All 3,774 prior era records and existing chart records are preserved; basemap,
  airspace and airfields references unchanged. Unrelated October 25 additions
  were excluded. Old assets and permanent URLs were preserved.
- Frontend checks, next build contract, three frontend tests and diff whitespace
  checks passed. Production shell built within all normal byte budgets.
- Persistent production inputs promoted to `~/archive-next-build/prod/`;
  previous inputs backed up under the candidate's `baseline/prod-work/`.

Release evidence: `~/archive-next-build/2026-10-28-faa-cycle/publication/`
(`archive-readback.json`, `metadata-readback.json`, `served-byte-checks.json`,
`verified-objects.jsonl`, deployment logs). Raw download evidence is below.
Legacy Pages deployment and regression CI passed; the follow-up is deleted.
No remaining publication work. Final evidence is recorded below.

## Authorization and status

User requested both FAA cycles downloaded and archived, with active sectionals in
rawtiffs and deferred chart families in rawtiffs_attic, then missing cycles processed
on the NUC and uploaded to the website. User explicitly approved an automatic
follow-up in this chat to finish publication and notify them when published.
No further publication approval is needed. Do not notify for unchanged progress.

Both currently offered VFR sets (2026-09-03, 2026-10-29) are preserved: 380 files,
27,746,579,364 bytes. ZIP CRCs and SHA-256 checked; PDF magic/EOF and SHA-256 checked.
106 sectional ZIPs live in /Volumes/projects/rawtiffs/<MM-DD-YYYY>/.
274 files (84 deferred-family ZIPs and 190 PDFs) live in
/Volumes/projects/rawtiffs_attic/faa_vfr_cycles/<MM-DD-YYYY>/<family>/.
The 84 ZIPs cover TAC/flyways, helicopter, Grand Canyon and Caribbean products.
FAA All_Files ZIPs duplicate the individual products and were intentionally skipped.
Four older US/Alaska wall-planning links returned 404; outside these two cycles.

Evidence: /Volumes/projects/faa-cycle-audit-2026-10-28/verified-manifest.json
and sectional-sheets.json. Local copies/scripts/FAA page snapshot:
worklists/data/faa_cycles_2026-10-28/ (gitignored).

September's 57 sheets are already published. October's 53 ZIPs yield 57 sheets;
all embedded GeoTIFF transforms/CRSs/color tables validated. Albuquerque was
warped with its cutline, rendered and visually checked. Original TIFF/sidecar
bytes are separated into 57 sheet containers under rawtiffs/10-29-2026/sheets/;
original distribution ZIPs retained. Includes Hawaiian insets and both Aleutian halves.
FAA metadata gives 2026-10-29 through 2026-12-23 inclusive; catalog end 2026-12-24.
57 catalog rows added (7861 total), with provenance and previous-cycle cutlines.
Backup: ~/archive.aero-attic/csv-backups/pre_faa_oct29_2026_2026-10-28.csv.
Catalog remains gitignored. Do not commit it.

## NUC run

Run: 2026-10-28_faa-cycle
Era: 2026-10-29_to_2026-12-24
Synced release: 20261028-220329-bb60336

- slice + 57 full-sheet artifacts: 1028-220355-fbcff877
- convert + alignment check: 1028-220355-5bf36186
- archive, moving mosaic to HDD: 1028-220355-09046e5d

Use archive_slicer status/get_job or slicerd/mac/slicerctl. No upload job queued:
the newer production release flow needs the local outputs, and a slicerd upload
would delete them. Slice started 2026-10-28 22:03:58 America/Chicago.
No Mac worker: user asked for the NUC. Raw source overlay auto-remounted.

## Publication procedure and historical preparation

Read CLAUDE.md and worklists/18_apex_cutover.md. Use the current hashed production
release flow, not only slicerctl publish (which updates the legacy viewer).
Verify all 57 chart artifacts, complete mosaic and alignment before publishing.
Use scripts/next_release_run.py plan/upload/permanent/stage with a dedicated
candidate ~/archive-next-build/2026-10-28-faa-cycle. Hashed objects go to
charts-beta (production despite the name), permanent keys/cards/timeline/bundle to charts.
Then overlays, missing cards, overlays again, airports, manifest, upload next/
objects and cards, build/deploy next/site, refresh legacy bundle/dates/frontend
and commit/push relevant generated frontend changes. Read Cloudflare/Wrangler
skills before deployment. User has authorized upload/deploy.

The documented ~/archive-next-build/prod directory was missing. Current production
manifest was downloaded from next/manifest.229990827e88.json (3774 eras). The
retained NUC hashed plans account for ALL 3774 exact live hashes. Saved merged
records: worklists/data/faa_cycles_2026-10-28/restored-plan.json (3774 eras + 7408
chart records), from 2026-10-03_reslice, 2026-10-13_georef and 2026-10-13_eighties.
Restoration of ~/archive-next-build/prod is COMPLETE. Its sparse metadata stage,
plan, OG manifest, coverage and published timeline are restored. Rebuilding the
manifest with the original generated timestamp matched the live manifest exactly
in every field (all 3774 eras and all overlay references). See prod/RESTORED.txt.
Current live refs are preserved in production-manifest.json. Do not regress the
basemap, airspace, airfields, or any existing era hashes. Preserve unrelated local
changes and avoid accidentally publishing other new catalog work from October 25.

Useful implementation details if restoring:
- NUC stubs at slicer-runs/<run>/hashed/stubs/<new-key>, or prefixes from retained
  pmtiles files for October 13 runs. Only 57 MB of metadata prefixes for 3774 eras.
- scripts/next_release_run.py stage expects prod/stage/{sectionals,basemap,airspace},
  prod/work/plan.json, and prod/work/og-manifest.jsonl.
- basemap: /Volumes/projects/protomaps_basemap/protomaps-20260826.pmtiles, hashed
  key basemap/protomaps-20260826.0938e8f9d55b.pmtiles.
- airspace: /Volumes/projects/airspace_pmtiles/class-20261013.pmtiles, hashed key
  airspace/class-20261013.cce35551b42d.pmtiles.
- current OG manifest: /Volumes/projects/archive-next/og/manifest.jsonl.
- metadata-9a94f060.bundle is the old legacy bundle; build_metadata_bundle.py
  parse_bundle/build_bundle allow preserving old verified records and adding a new
  era. Or build against complete local sparse metadata stubs with permanent names.
- current local shell dist next/site/dist/budgets.json points to the live manifest.

Finish with remote object verification, live manifest/era tile/full-sheet checks,
and a browser check for October 29. Update this worklist with release URLs and
receipts. Notify the user only on completion, actionable failure, or required input.
Disable the follow-up after verified completion; do not archive the chat.

Automatic follow-up: `finish-faa-cycle-publication`, hourly until verified completion.

Scope caution: both the restored published timeline and root timeline_data.json
still contain the October 13 inventory (165 locations, 7742 chart entries). The
catalog had 7804 rows before this task, including later unpublished work. When
building the release, compare generated timeline/coverage against the saved live
baseline and merge only the 57 October 29 sheets; do not silently publish unrelated
October 25 additions. Preserve existing groups/rings and other overlay references.
Root timeline and coverage have not yet been changed by this task.

Heartbeat 2026-10-28 23:04 America/Chicago: slice remains healthy; mosaic GeoTIFF
36% with about 38 minutes estimated for that stage. Convert and archive remain
queued behind it. No publication attempted while processing is incomplete.

Heartbeat 2026-10-29 00:06 America/Chicago: slice succeeded. Mosaic conversion
started at 23:33; z12 is progressing (13%, about 375 tiles/s). Archive waits on
conversion/alignment. No failure or required user action; publication still pending.

Heartbeat 2026-10-29 01:07 America/Chicago: conversion continues normally at
37% of z12, about 371 tiles/s. Archive and publication remain dependent on
conversion/alignment completion. No user action needed.

Heartbeat 2026-10-29 02:10 America/Chicago: conversion continues normally at
65% of z12, about 392 tiles/s. Archive and publication await conversion/alignment.
No failure or required user action.

Heartbeat 2026-10-29 03:10 America/Chicago: conversion continues normally at
94% of z12, about 405 tiles/s (highest-zoom ETA about 15 minutes; lower zooms,
packaging, alignment and archive still follow). No user action needed.

Heartbeat 2026-10-29 04:10 America/Chicago: all NUC stages succeeded. Era archive
3.81 GB, z0–12, 7,529,437 addressed tiles. Alignment passed 12 probes with worst
shift 0.00 px (0.3 px tolerance). Release candidate preparation/publication started.

## Final completion — 2026-10-29 04:37 America/Chicago

Release commit `a2b8d4d544f03c0f91c0c5824c1d79a222113f86` pushed to main.
GitHub Pages deployment [40850040623](https://github.com/hemenway/archive.aero/actions/runs/40850040623)
and full regression/browser suite [40850040559](https://github.com/hemenway/archive.aero/actions/runs/40850040559)
both passed. All 57 new edition pages return correct metadata; live `dates.csv`
exactly matches the release and both new legacy assets are served successfully.
The initial Python urllib page probe received a client-specific 403; curl and
browser verification passed. No website or content failure remained.

`finish-faa-cycle-publication` was deleted after publication verification.
Chat kept open. No previous published data was deleted. The unrelated Numbers
backup remains untouched. Completion and page evidence are saved in
`publication/completion.json` and `publication/edition-pages.json` under the candidate.
