# Worklists

**Updated 2026-10-13: six-era georef run and recovered-history airspace build complete; release candidate unpublished. Apex cutover completed 2026-10-12.** Start here for current status and next actions. The dated
results in each worklist are evidence from that run; they are not fresh production
checks. Original hunt records and completed cleanup reports are retained below.

## Next actions

0. ~~Run the apex cutover ([18](18_apex_cutover.md)).~~ Complete October 12:
   viewer and 50 airport pages live, all 7,120 edition cards published,
   permanent keys refreshed, beta retired behind 410. The runbook records
   production rebuilds and rollback.
1. ~~Reconcile source availability ([02](02_disk_vs_dole.md)).~~ Clean on
   October 4: all 7,744 rows resolve on disk and no file is unexplained.
2. ~~Publish the 71 missing chart artifacts ([06](06_publish_sync.md)).~~ Done
   October 12: 52 copied from the completed reslice and receipted, 10 retired,
   9 renamed (see 06, which also records the `Honolulu` / `Honolulu Inset`
   location split to resolve). **Review and publish the October 13 release candidate**:
   all six affected eras completed slice, conversion and archive in
   `2026-10-13_georef`, with five changed full-sheet chart artifacts (Boston 1957,
   SF 1971 north/south, Dutch Harbor 2004 south, Dallas 1981 north). SF's corners,
   printed standard parallels and fold cutlines are now refined; grid-alignment
   errors for all six mosaics are ≤0.01 px against a 0.3 px tolerance. Candidate
   `~/archive-next-build/2026-10-13-georef` is being assembled with the new airspace
   build. Publication, served-data verification and pointer updates remain pending;
   see [04](04_georef_backlog.md).
   [14](14_geotiff2pmtiles_audit_handoff.md) also tracks missing evidence for the
   historical PMTiles run-length repairs.
3. **Work the remaining chart repairs ([01](01_dole_slicer_failures.md),
   [04](04_georef_backlog.md)).** Use the current status sections, not the original
   July counts or completed repair checklists.
4. **Finish ATC stabilization ([08](08_atchistory_migration.md)).** Confirm the
   outstanding console actions and renewed 404 validation, reconcile the build
   tree before another sync, and review broader `/atc/` changes on October 30.
5. **Resolve Jev acceptance findings ([16](16_jev_typesafe.md)).** The
   [October 13 review](16_jev_validation_2026-10-13.md) covers 50 Freeman labels and
   69 ATC posts; acceptance failed. Fix the remaining date/status and entity/source
   errors before publishing replacements or new metadata consumers. The existing
   published airfields hash equals the local artifact's hash; the georef candidate
   preserves that published reference. Sample completion is not approval of the data.
6. **Publish the verified recovered-history airspace build ([15](15_airspace_overlay.md)).**
   Thirty older NASR cycles are verified and included in the new local build:
   153 held regional cycles total, including 116 US cycles. It is part of the
   unpublished October 13 candidate; the added history is not yet live in the viewer.

Cold-backup destination selection ([10](10_cold_archive.md)), acquisition outreach
([03](03_web_sources_searched.md)), and growth decisions ([09](09_growth_and_revenue.md))
remain separate open work. No scheduled helper or deployment is implied by this index.

## Chart inventory and publishing

| Worklist | Current status / next step |
|---|---|
| [01 — Slicer failures](01_dole_slicer_failures.md) | Historical July failure report with later repairs; follow the reconciled remaining items before another affected-range run. |
| [02 — Disk vs. catalog](02_disk_vs_dole.md) | **Clean October 4:** 0 unresolved rows, 0 unexplained files. Duplicate candidates still need hash proof before any cleanup. |
| [03 — Sources searched](03_web_sources_searched.md) | Preserved venue index and search history; use the open acquisition lanes and avoid repeating closed searches. |
| [04 — Georeferencing](04_georef_backlog.md) | October 13: six-era run and five changed chart artifacts built; publication pending. Source-aware audit: 107 incomplete of 7,742 rows, a triage queue rather than a manual-georef count. B3 status fix implemented. |
| [05 — Date quality](05_date_quality.md) | 263 END-ESTIMATED, 236 GAP, 45 DATE-APPROX flags on 301 distinct rows; flags overlap. Review the dated snapshot and provenance before changes. |
| [06 — Publication](06_publish_sync.md) | October 12 publication cleanup and cutover complete; October 13 six-era/five-chart candidate awaits publication and served verification. Retain permanent keys and update supporting metadata together. |
| [07 — Acquisition/import queue](07_download_queue.md) | Expanded 1,517-finding queue attempted in July; six HUNT26 rows retained. Revalidate the deferred 258-file / 229-row import plan against current holdings. |

## Project work

| Worklist | Current status / next step |
|---|---|
| [08 — ATC migration](08_atchistory_migration.md) | Cutover executed August 31; stabilization continues. September facility-browser and directory-link fixes are recorded. |
| [08a — Backlink repair](08a_backlink_repair.md) | Remaining backlink decisions and optional external edits; preserve redirects and use current canonical targets. |
| [09 — Growth and revenue](09_growth_and_revenue.md) | Airport/edition-page pilot live October 12; audience qualification, opt-in and Jev metadata integration remain open. |
| [10 — Cold archive](10_cold_archive.md) | Planned; choose destination after a fresh source manifest. No verified off-site upload is recorded. |
| [11 — ACASIS recovery](11_acasis_img_recovery.md) | Nine sectionals imported; reconcile their publication. Locate the recovery image/checkpoints before any targeted resume; deferred chart families remain in the attic. |
| [12 — First SD-card batch](12_sdcard_batch.md) | 18 only-copy 2014–15 rows published; the nine later ACASIS imports have a separate publication follow-up. |
| [13 — Second SD-card batch](13_sdcard_batch2.md) | 17 TAC editions plus Grand Canyon retained; no new sectionals. Enroute preservation complete; partial-card recovery, visual QA and collection support remain open. |
| [14 — geotiff2pmtiles audit](14_geotiff2pmtiles_audit_handoff.md) | PR #49 recorded merged; managed dependency in use. Reconcile production-repair evidence and remaining audit findings against the current upstream revision. |
| [15 — Airspace](15_airspace_overlay.md) | October 13 local build includes 30 verified older NASR cycles; 153 regional cycles, 116 US. Candidate unpublished. SUA, labels and other extensions remain open. |
| [16 — Jev / TypeSafe](16_jev_typesafe.md) | Samples complete: 50 Freeman labels and 69 ATC posts. Acceptance failed; source-review corrections and new metadata consumers remain open. Preserve the existing published airfields reference until a replacement is verified. |
| [18 — Apex cutover](18_apex_cutover.md) | Completed October 12: apex viewer, 50 airport pages, 7,120 edition cards, permanent-key refresh and beta 410. Production rebuild recipes, verification and rollback retained. |
| [17 — Repo split](17_repo_split.md) | Plan drafted 2026-10-02, nothing moved: site stays in this repo, pipeline (slicerd, scripts, shapefiles, worklists) extracted with history. Six decisions open; decouple-in-place steps can start before the redesign branches land. |

## Completed runs and reference records

| Record | Purpose |
|---|---|
| [08b — Freeze/cutover runbook](08b_freeze_cutover_runbook.md) | Historical execution record for the completed August 31 cutover; use 08 for current follow-up. |
| [14 — Upstream drafts](14_geotiff2pmtiles_upstream_drafts.md) | Preserved issue/PR drafts and references; not a new submission queue. |
| [Deep archive cleanup — September 19](deep_archive_cleanup_2026-09-19.md) | Completed consolidation: 823 originals preserved and 2,680 exact duplicates removed, with manifests. |
| [Duplicate extraction cleanup — September 19](dup_extractions_cleanup_2026-09-19.md) | Completed removal of 15 hash-verified duplicates, with retained-copy evidence. |
| [Georef publication — October 1](georef_publish_2026-10-01.md) | Completed 17-era / 30-chart donation and georef batch with byte-verified remote evidence. |
| [Jev acceptance review — October 13](16_jev_validation_2026-10-13.md) | Expanded samples, deterministic fixes and remaining publication blockers; source evidence and cached replays retained locally. |
| [Helper brief](helper_prompt.md) | Read-only monitoring instructions; read `CLAUDE.md` first. A brief is not evidence of a configured schedule. |

## Evidence and maintenance

- Read `CLAUDE.md` before working a queue. Keep catalog-derived data out of git;
  load `master_dole_v2.csv` through `scripts/dole_v2.py`.
- Row-level audits, manifests, upload ledgers, and verification reports live in
  [`data/`](data/) (gitignored). Recompute against the correct mounted sources
  before treating a dated CSV as the current queue.
- `09`, `10`, `13`, both `14` documents, and `helper_prompt.md` are intentionally
  gitignored local plans. This cleanup preserves that setting; their links work
  in this checkout but the files are absent from a fresh clone.
- Update a worklist's current status when recording a result. Distinguish
  cataloged, generated, uploaded, and verified; keep the evidence date. Do not
  infer publication from a local artifact or delete a permanent URL to tidy a list.
- The original 01–07 baselines were generated July 16 from the July 14 slicer run,
  the catalog, source audit, and hunt findings. Their regeneration recipes and
  later results live in the respective worklists.

## Preserved search archive

[`search_archive/`](search_archive/) is the verbatim hunt record moved from
`missing_from_dole/` on July 16. Do not rewrite its historical findings as a
current download or execution queue; use 03 and 07 for current disposition.

| File | What it preserves |
|---|---|
| `ACTION_PLAN.md` | Original acquisition tiers and their July status. |
| `dole_search_log.txt` | Parts 1–8: sources, dead ends, and exact queries. |
| `dole_search_handoff_round8.md` | Round-8 hunt state, rules and acquisition techniques. |
| `dole_search_handoff_1970-2010.md` | Superseded round-5 handoff. |
| `dole_search_lane_reports/` | Per-lane findings, inventories, IDs and outreach drafts. |
| `missing_from_dole_online.csv` | 1,490 verified-at-the-time online findings, the source for 07. |
| `missing_from_dole.csv` | Closed 136-row local-disk audit from July 7. |
| `catalog_additions_report.csv` | Per-file ADD/SKIP/FIX disposition of the July 14 cataloging. |
| `end_date_fill_report.csv` | All 294 date/end-date changes from the July 14 fill, with method. |

The older `unlogged_in_rawtiffs.csv` audit is retired under
`~/archive.aero-attic/legacy-data/`; 02 is its successor.
