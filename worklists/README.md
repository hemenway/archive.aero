# worklists/ — error categories to work through

Generated 2026-07-16 from the 2026-07-14 slicer run, `master_dole_v2.csv`, a fresh
rawtiffs↔dole audit, and the chart hunt's search record. Each doc is a self-contained
worklist; row-level data lives in [`data/`](data/); the complete verbatim hunt record
(search log, handoffs, lane reports, findings CSVs) lives in
[`search_archive/`](search_archive/) — it is the preserved copy of the former repo-root
`missing_from_dole/` folder (minus its symlink dirs), moved here 2026-07-16.

| Doc | Category | Headline |
|---|---|---|
| [01_dole_slicer_failures.md](01_dole_slicer_failures.md) | In the dole but failed in the slicer | 222 files never warped; 90 date ranges skipped; **2026-01-22 mosaic has only Wichita**; 2 pipeline bugs (zip-PDF resolver, zip-bytes-saved-as-.pdf); 6 silent single-chart holes |
| [02_disk_vs_dole.md](02_disk_vs_dole.md) | On disk but not in the dole (and vice versa) | **CLEAN as of 2026-08-20** (0 rows unresolvable — the 11 Pack-12 JPGs were staged; the 2 mystery G4332 files identified as Grand Canyon VFR charts, out of scope); 623 deletable NARA duplicates = 38 GB; re-runnable via `scripts/audit_disk_vs_dole.py` |
| [03_web_sources_searched.md](03_web_sources_searched.md) | Places on the web already searched | Master venue index with verdicts; do-not-re-search list; ranked open frontiers (ASU contact, NLA scan request, AVSIM logins) |
| [04_georef_backlog.md](04_georef_backlog.md) | Dole rows without GCPs | 226 pre-2011 rows need georeferencing (mostly the 189 newly cataloged downloads); 8 rows lack even a cutline |
| [05_date_quality.md](05_date_quality.md) | Date uncertainty flags | 259 END-ESTIMATED, 236 GAP (ranked as search targets), 40 DATE-APPROX |
| [06_publish_sync.md](06_publish_sync.md) | Live site out of sync with the new run | 485 mosaics unpublished; 209 stale pmtiles keys; regenerate coverage/timeline after upload |
| [07_download_queue.md](07_download_queue.md) | Verified online findings not yet downloaded/cataloged | 1,309 of 1,490 pending (760 wayback PDFs, 279 wayback zips, 72 LOC gct00498, 43 raremaps DZI…); batching plan; the "31 on disk but uncataloged" item is closed (2026-08-20) |

Docs 08+ are **project plans rather than error categories** — same self-contained format,
but they track a piece of work start-to-finish instead of a backlog of rows:

| Doc | Category | Headline |
|---|---|---|
| [08_atchistory_migration.md](08_atchistory_migration.md) | atchistory.org → archive.aero/atc/ | Full static flatten of a 3.6 GB WordPress+FrontPage site into R2 + a redirect worker; designed canonical URI scheme; **cutover executed 2026-08-31** (freeze ran a day late) — [08a](08a_backlink_repair.md) backlink repair, [08b](08b_freeze_cutover_runbook.md) runbook |
| [09_growth_and_revenue.md](09_growth_and_revenue.md) | Growth & revenue after the merge | 4 indexable pages vs. atchistory's ~1,400 + Wikipedia backlinks; programmatic airport/edition pages (~8,700 URLs); prints as the primary revenue bet; 6 open decisions |
| [10_cold_archive.md](10_cold_archive.md) | Private cold archive of the originals | Off-site copy of the source scans and the catalog |
| [11_acasis_img_recovery.md](11_acasis_img_recovery.md) | ACASIS `.imgcv2` recovery | 891 charts not in the catalog carved off the iFly build machine's disk image; 868 extracted to the attic, 9 promoted 2026-08-29 |
| [12_sdcard_batch.md](12_sdcard_batch.md) | 2026-08 SD-card batch | Five iFly EFB cards mined; 18 only-copy 2014-15 rows published |
| [13_sdcard_batch2.md](13_sdcard_batch2.md) | 2026-08/09 second SD-card batch + ACASIS pre2015 | 17 TAC + Grand Canyon ed 3 to the attic, zero new sectionals; edition numbers are per chart type |

`helper_prompt.md` is the standing brief for a helper session (read `CLAUDE.md` first). Docs 09, 10, 13 and `helper_prompt.md` are not yet tracked in git — add them when their contents settle.

## Suggested working order

1. **Mechanical unblocks** (hours): ~~stage the 11 Pack-12 JPGs (02A)~~ (done 2026-08-20),
   unzip the 13 fake-PDFs (01B), fix the Denver `Service_Unavailable` row (01D), re-pull
   Seward_93 (01B), fix the zip-PDF resolver in slicer.py (01B).
2. **Re-warp the two crippled newest mosaics** with an SRS-capable GDAL (01A) — biggest
   user-visible payoff.
3. **Georeferencing sprints** (04 + 01C): Juneau's 18 editions light up 1994–2012;
   Hawaiian 15 light up 1947–69; the 78 N/S half-sheets light up 2003–05; WASP's 6 are
   rare desert-era content.
4. **Re-run slicer** for affected ranges, then **publish sync** (06).
5. **Ongoing**: outreach + logins from 03's open list; date-quality gaps (05) feed the hunt.

## Provenance / how to regenerate

- 01: mined from `slicer_run.log` + `processing_review_*.csv` in
  `/Volumes/projects/2026-07-14 slicer run/` (line numbers cited in the doc).
- 02: walk of `/Volumes/projects/rawtiffs/` mirroring `slicer.py:_build_source_index`
  resolution rules, diffed against `master_dole_v2.csv`.
- 03: condensed from `search_archive/dole_search_log.txt` Parts 1–8 + round-8 handoff +
  lane reports (those stay authoritative).
- 04/05: filters over `master_dole_v2.csv` (no `gcp1_px` & pre-2011; note-column flags).
- 06: diff of run-folder tif basenames vs `dates.csv` `date_iso`.
- 07: `search_archive/missing_from_dole_online.csv` URLs matched against dole download_links.

## search_archive/ contents (verbatim hunt record — keep!)

| File | What it is |
|---|---|
| `ACTION_PLAN.md` | tiered action plan (Tier 0/1 closed 2026-07-14; Tiers 2–6 = outreach/scans/logins/physical/frontiers, still open) |
| `dole_search_log.txt` | Parts 1–8: every source ever searched incl. dead ends and exact queries |
| `dole_search_handoff_round8.md` | authoritative hunt state, rules, and the TECHNIQUE cheat-sheet (raremaps GCS bypass, Trove keyless API, NARA WAF quirks, bot-wall list) |
| `dole_search_handoff_1970-2010.md` | superseded round-5 handoff (historical) |
| `dole_search_lane_reports/` | per-lane detail incl. the contact-email payloads: ASU barcodes/FILE_NAMEs (`lane_j`), GlidePlan inventory (`glideplan_macmaps_inventory.txt`), Welch/Fox file lists (`lane_g`), AVSIM download IDs (`lane_d`) |
| `missing_from_dole_online.csv` | 1,490 verified online findings (worklist 07's source) |
| `missing_from_dole.csv` | the CLOSED 136-row local-disk audit (2026-07-07) |
| `catalog_additions_report.csv` | per-file ADD/SKIP/FIX disposition of the 2026-07-14 cataloging (210 rows added) |
| `end_date_fill_report.csv` | all 294 date/end_date changes from the 2026-07-14 end-date fill, with method |

Also related: `unlogged_in_rawtiffs.csv` (2025-12, superseded by 02; archived to ~/archive.aero-attic/legacy-data/).
