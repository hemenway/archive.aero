# Worklist 07 — Online findings download queue

## Current status — 2026-09-21

The **1,517-row findings queue was attempted and analyzed in July**, not left as
1,309 untouched downloads. `rawtiffcandidates/_analysis_report.md` remains the
July 24 evidence: 295 include, 1,021 include-after-georef, 158 already cataloged,
24 KML indexes, 13 reference-only, 4 failed and 2 suspect-truncated. These are
analysis verdicts, not 1,517 successful usable downloads.

Edition-level dedupe on July 25 reduced the proposed additions to 264 candidates.
Only **six HUNT26 rows remain in the current 7,672-row catalog**, confirmed via
`scripts/dole_v2.py`; the other **258 files / 229 proposed rows** were deferred
back to `/Volumes/projects/rawtiffcandidates/`. That is the last reviewed import
plan, not a fresh claim that all 258 are still unique or ready.

- [ ] Revalidate `data/hunt26_plan.csv` against today's **location + edition +
  side** holdings and the files before resuming any import; resolve its 104
  historical location/year collisions and explicit exclusions.
- [ ] Prepare the retained candidates that need work (GPO fiche stitching,
  georef, source/edition checks), then verify one chart end-to-end before batching.
  Do not replay the old whole-queue download or import instructions blindly.
- [ ] Keep the July failure/truncation dispositions; Seward 93 requires an
  alternative source, not another pull of its known broken capture.
- [ ] Pursue new contact/login/scan-request leads through [03](03_web_sources_searched.md).
  Outreach/access completion is unrecorded here.

The July 25 **31 already-on-disk-but-uncataloged** items were closed by the
August 20 audit. That old queue stays closed; the **new** September 21 source
availability/disposition findings belong in [02](02_disk_vs_dole.md).
No source files, catalog rows or generated artifacts were changed for this doc update.

## Historical July 16 queue baseline

The original URL-only comparison was **181 cataloged / 1,309 pending** out of
1,490 findings; the pending export [data/download_queue_pending.csv](data/download_queue_pending.csv)
is a historical snapshot. Its largest lanes were 760 visual-layout FAA PDFs,
279 old-layout ZIPs, 72 LOC scans, 55 NARA scans, 43 raremaps, 19 cartweb,
17 NOAA, 11 FAA samples, 10 NLA and 43 one-offs. These counts were superseded by
round 9 and the full download/edition-dedupe passes below. URL mismatch alone
must not be treated as a new edition or an uncompleted download.

## 2026-07-24 — full queue attempted and analyzed

All 1,517 rows attempted into `/Volumes/projects/rawtiffcandidates/` (183 GB, per-source
subdirs). Row-level results: `_download_manifest.csv` / `_manifest.jsonl` (url, sha1,
bytes, status) and `_analysis.csv` / `_analysis_report.md` (dims, dpi, georef, verdict).
No byte-identical duplicates in the set.

**Verdicts:** 295 include (55 GB) · 1,021 FAA print-PDFs include-after-georef (130 GB) ·
158 already cataloged (+24 usahas KML indexes) · 13 reference-only
(low-res) · 4 failed and 2 suspect-truncated. The 24 KML indexes are separate from the
158 already-cataloged verdicts; these are file/URL verdicts, not catalog-row counts.

Findings that update the assumptions above:

- **The FAA PDFs are not GeoPDFs.** Sampled every capture year 2014–2026: all are
  Photoshop image-PDFs, ~300 dpi (16–18k px wide), zero SRS/neatline. The 01A GeoPDF
  SRS fix is moot for them — they need `scripts/georef_infer_from_sibling.py` corner-GCP
  transfer, same as the 2026-07-22 repair batch. old-layout `*_P.pdf` and visual-layout
  PDFs are the same animal.
- **Wayback truncation is not only the 1 MB pattern** — captures also die at arbitrary
  offsets (256 KB…97 MB) while CDX advertises full length; every retry breaks at the
  identical byte. Remedy that worked: CDX capture-hop (try other timestamps, largest
  first) — recovered 9 of 14. Unrecoverable from any capture: `houston_105_p.pdf`,
  `washington_108_p.pdf`, `seward_93.zip`, LA visual 09-2022 + 10-2023 (all 1 MB).
- OSU CONTENTdm (Tulsa/OKC 1945) refuses python-requests but serves curl — both
  recovered at 9000 px. raremaps `img_121271` max pyramid level 404s; stitched at
  level−1 (13k px). 2004-set `Albuquerque North.jpg`, UNT high_res, Newberry Commons,
  Ohio Memory, El Paso CONTENTdm returned <5k px derivatives → reference_only.
- NOAA Data Sampler CD-ROM was classified `failed` / `unreadable raster` in
  `_analysis_report.md`; the earlier "downloaded but unexplored" note does not
  establish a usable chart source.

FAA print PDFs require 300-dpi rasterization and sibling-based georef; the
July 25 edition dedupe below determines which are worth preparing at all.

### Dedupe vs holdings (2026-07-25)

Edition-level compare of the 1,316 include-verdict rows against rawtiffs + dole
(`_dedupe_vs_holdings.csv` on the volume) — the hunt deduped by URL only, and most
FAA-era content was already held from other sources (NARA rg-237 cycle pulls, FAA bulk):

- **264 genuinely new** — nearly all pre-1971: NARA rg-370 (55), LOC ca-scans (52),
  raremaps (46), GPO microfiche (23), NOAA (23), cartweb (14), FAA sample zips (11),
  NLA (11), 19 misc one-offs; FAA-era only 8: `anchorage_88 atlanta_87 brownsville_87
  great_falls_81 juneau_51 los_angeles_89` zips, `western_aleutian_islands_51_p.pdf`,
  and the sole new visual chart `Western_Aleutian_Islands.pdf`.
  **104 of the 264 carry a (location, year) collision with an existing dole row** —
  verify edition/side before import.
- **1,021 duplicates** of already-cataloged content (by cycle/edition, not URL) —
  includes 756/757 of the visual-PDF lane and 245/277 of the old-layout lane. Nothing
  to do.
- **31 on disk but uncataloged** — file already in rawtiffs, no dole row: 25 old-layout
  SEC editions + 6 LOC ca-scans. This was the historical import-only remainder.
  **Closed 2026-08-20:** that two-way audit (`scripts/audit_disk_vs_dole.py`)
  found **0 catalogable files left**; everything real got rows during the July–August
  sessions, and the 13 residual unclaimed files are all explained (see worklist 02's
  2026-08-20 note + `data/uncataloged_on_disk_2026-08-20.csv`).

## 2026-07-25 — HUNT26: staged, then reverted to the 6 ready zips

The 264 dedupe-new rows were fully identified/dated and a reviewed import plan
built (`worklists/data/hunt26_plan.csv` — one row per file with parsed
location/date/edition/cutline/LCC, exclusion verdicts, and VERIFY-EDITION
flags). The full import was then **backed out the same day**: only the 6
tfw-georeferenced wayback old-layout zips (anchorage_88, atlanta_87,
brownsville_87, great_falls_81, juneau_51, los_angeles_89) stayed — files in
`/Volumes/projects/rawtiffs/hunt_2026-07/wayback_old_layout/`, 6 dole rows
with note tag `HUNT26`, sliceable as-is. Everything else was moved back to
`/Volumes/projects/rawtiffcandidates/` under its original paths.

Historical tools for a future revalidated batch: `worklists/data/hunt26_move.py` (moves per the plan,
handles the jp2→LZW-tif conversion that strips the bogus Greenwich georef and
the .php→.jpg renames) then `scripts/import_hunt26.py` (skips the six already-present HUNT26 zips;
that check does not replace a current edition-level review). Georef-tool sidebar items G1–G8 for the 229-row
hand-GCP backlog are preserved in `worklists/data/hunt26_worklist_items.py`.

Durable findings from the identification pass (all encoded in the plan CSV):

- The 49-file LOC gct00498 "Base & Duplicates" set was identified visually —
  titles, LOC accession stamps, printed changes-after dates. `ca*v` files are
  chart-back TEXT pages, not map halves (same for the 3 gct00089 `*v` probes,
  Corpus Christi 1961/63 backs, tulsa2, Boston/Dallas 1961 versos).
- NARA rg-370 hunt-name years are wrong for Dallas/El Paso — the NARA URL
  filenames (`Dallas_03051946.JPG`, MMDDYYYY) are authoritative.
- The two rg-351 "Seattle 1945" files are SECRET 1:50k **Crete** topo sheets
  (hunt misidentification); the sampleVFR tifs are 1-bit print separation
  plates; AOD_VFRCharting_LA.zip duplicates the held wayback LA_98 zip.
- The archive.org vintage jp2s carry a bogus Greenwich geotransform.
- GPO fiche zips hold jp2 camera tiles + a hugin .pto — stitch before GCP.

Other July 25 candidate dispositions: the 1,021 edition-dups, 158
already cataloged, and 13 reference-only. (The 31 disk-but-uncataloged rows are
closed as of 2026-08-20 — see the dedupe section above.)
