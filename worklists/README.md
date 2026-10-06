# Worklists

**Reviewed 2026-09-21; georef/publication updated 2026-10-01; 02/04/06 updated 2026-10-04; 06 updated 2026-10-05.** Start here for current status and next actions. The dated
results in each worklist are evidence from that run; they are not fresh production
checks. Original hunt records and completed cleanup reports are retained below.

## Next actions

1. ~~Reconcile source availability ([02](02_disk_vs_dole.md)).~~ Clean on
   October 4: all 7,744 rows resolve on disk and no file is unexplained.
2. **Publish the 71 missing chart artifacts ([06](06_publish_sync.md)).** The
   August 29 run's per-chart files (nine ACASIS sectionals, 62 inset and
   W. Aleutian sheets) never reached R2 and no local copy remains; take them
   from the October 3 reslice with `scripts/publish_chart_pmtiles_from_reslice.py`
   (October 5: 5 ready, 66 waiting for the reslice to reach their eras). Dallas 1981 north is catalog-ready and needs a
   slice and publish ([04](04_georef_backlog.md)).
   [14](14_geotiff2pmtiles_audit_handoff.md) also tracks missing evidence for the
   historical PMTiles run-length repairs.
3. **Work the remaining chart repairs ([01](01_dole_slicer_failures.md),
   [04](04_georef_backlog.md)).** Use the current status sections, not the original
   July counts or completed repair checklists.
4. **Finish ATC stabilization ([08](08_atchistory_migration.md)).** Confirm the
   outstanding console actions and renewed 404 validation, reconcile the build
   tree before another sync, and review broader `/atc/` changes on October 30.
5. **Finish Jev validation and publication ([16](16_jev_typesafe.md)).** W1/W2 ran
   locally. Complete the remaining samples/review, publish the airfield dates,
   then connect the ATC metadata to the airport-page pilot in [09](09_growth_and_revenue.md).
6. **Rebuild airspace with the recovered history ([15](15_airspace_overlay.md)).**
   Thirty older NASR cycles are held locally; verify and include them in a new
   immutable build before treating them as available in the viewer.

Cold-backup destination selection ([10](10_cold_archive.md)), acquisition outreach
([03](03_web_sources_searched.md)), and growth decisions ([09](09_growth_and_revenue.md))
remain separate open work. No scheduled helper or deployment is implied by this index.

## Chart inventory and publishing

| Worklist | Current status / next step |
|---|---|
| [01 — Slicer failures](01_dole_slicer_failures.md) | Historical July failure report with later repairs; follow the reconciled remaining items before another affected-range run. |
| [02 — Disk vs. catalog](02_disk_vs_dole.md) | **Clean October 4:** 0 unresolved rows, 0 unexplained files. Duplicate candidates still need hash proof before any cleanup. |
| [03 — Sources searched](03_web_sources_searched.md) | Preserved venue index and search history; use the open acquisition lanes and avoid repeating closed searches. |
| [04 — Georeferencing](04_georef_backlog.md) | 115 pre-2011 rows lack complete GCPs; this is a triage filter, not a hand-georef count. Current table separates source, GCP, half-sheet, and publication issues. |
| [05 — Date quality](05_date_quality.md) | 263 END-ESTIMATED, 236 GAP, 45 DATE-APPROX flags on 301 distinct rows; flags overlap. Review the dated snapshot and provenance before changes. |
| [06 — Publication](06_publish_sync.md) | October 1 georef batch complete; 3,761 active `dates.csv` entries. ACASIS audit and modern-era reslice remain separate; retain permanent keys. |
| [07 — Acquisition/import queue](07_download_queue.md) | Expanded 1,517-finding queue attempted in July; six HUNT26 rows retained. Revalidate the deferred 258-file / 229-row import plan against current holdings. |

## Project work

| Worklist | Current status / next step |
|---|---|
| [08 — ATC migration](08_atchistory_migration.md) | Cutover executed August 31; stabilization continues. September facility-browser and directory-link fixes are recorded. |
| [08a — Backlink repair](08a_backlink_repair.md) | Remaining backlink decisions and optional external edits; preserve redirects and use current canonical targets. |
| [09 — Growth and revenue](09_growth_and_revenue.md) | Audience qualification, opt-in, and a small airport/edition-page pilot remain open; Jev metadata is now available locally as an input. |
| [10 — Cold archive](10_cold_archive.md) | Planned; choose destination after a fresh source manifest. No verified off-site upload is recorded. |
| [11 — ACASIS recovery](11_acasis_img_recovery.md) | Nine sectionals imported; reconcile their publication. Locate the recovery image/checkpoints before any targeted resume; deferred chart families remain in the attic. |
| [12 — First SD-card batch](12_sdcard_batch.md) | 18 only-copy 2014–15 rows published; the nine later ACASIS imports have a separate publication follow-up. |
| [13 — Second SD-card batch](13_sdcard_batch2.md) | 17 TAC editions plus Grand Canyon retained; no new sectionals. Enroute preservation complete; partial-card recovery, visual QA and collection support remain open. |
| [14 — geotiff2pmtiles audit](14_geotiff2pmtiles_audit_handoff.md) | PR #49 recorded merged; managed dependency in use. Reconcile production-repair evidence and remaining audit findings against the current upstream revision. |
| [15 — Airspace](15_airspace_overlay.md) | US/France/Brazil class overlay shipped; recovered older NASR cycles await a verified build. SUA, labels and other extensions remain open. |
| [16 — Jev / TypeSafe](16_jev_typesafe.md) | Local W1/W2 batches complete: 2,822 airfields, 1,333 ATC posts. Validation follow-ups, airfields publication, and all metadata consumers remain open. |
| [17 — Repo split](17_repo_split.md) | Plan drafted 2026-10-02, nothing moved: site stays in this repo, pipeline (slicerd, scripts, shapefiles, worklists) extracted with history. Six decisions open; decouple-in-place steps can start before the redesign branches land. |

## Completed runs and reference records

| Record | Purpose |
|---|---|
| [08b — Freeze/cutover runbook](08b_freeze_cutover_runbook.md) | Historical execution record for the completed August 31 cutover; use 08 for current follow-up. |
| [14 — Upstream drafts](14_geotiff2pmtiles_upstream_drafts.md) | Preserved issue/PR drafts and references; not a new submission queue. |
| [Deep archive cleanup — September 19](deep_archive_cleanup_2026-09-19.md) | Completed consolidation: 823 originals preserved and 2,680 exact duplicates removed, with manifests. |
| [Duplicate extraction cleanup — September 19](dup_extractions_cleanup_2026-09-19.md) | Completed removal of 15 hash-verified duplicates, with retained-copy evidence. |
| [Georef publication — October 1](georef_publish_2026-10-01.md) | Completed 17-era / 30-chart donation and georef batch with byte-verified remote evidence. |
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
