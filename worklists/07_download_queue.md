# Worklist 07 — Online findings download queue

*Generated 2026-07-16 by matching `search_archive/missing_from_dole_online.csv` (1,490
verified findings) URLs against `master_dole_v2.csv` download_links.
**181 cataloged · 1,309 pending.** Pending rows extracted to
[`data/download_queue_pending.csv`](data/download_queue_pending.csv).*

Every URL here was already verified live (200 + plausible size) during the hunt — this is
a download/catalog queue, not a search list. Verification and dedupe rules:
`search_archive/dole_search_handoff_round8.md`.

## Already cataloged (181) — the 2026-07 Tier-1 batch

archive.org 2004 set (78), simviation realcharts (42), NARA Juneau 305976 (23), usahas
ChartGeek mosaics (23), plus 15 one-offs (aviationtoolbox, NOAA, glideplan, ESRI, FAA
sample…). These are staged in `/Volumes/projects/rawtiffs/dole_gap_2026-07/` and have dole
rows — their remaining work is georeferencing ([04](04_georef_backlog.md)) and slicer
fixes ([01](01_dole_slicer_failures.md)).

## Pending (1,309), by source

| Source | Rows | Notes |
|---|---|---|
| Wayback FAA `visual/<date>/PDFs/` print PDFs | 760 | 2011+ cycles; 13 whole + 11 partial editions. ⚠ 407 of the CDX captures are 1MB-truncated (editions 08-2021, 5×2022, 3×2023 unrecoverable as PDFs) — the queue rows are the verified-good ones, but re-check size on download |
| Wayback FAA old-layout `sectional_files/*` zips | 279 | 2011–2019 edition zips — the big post-2010 backfill |
| LOC gct00498 "Base & Duplicates" | 72 | pre-1971 tiffs, direct tile.loc.gov pulls |
| NARA rg-370/305976 (non-Juneau remainder) | 55 | 1940s TX, Tulsa 1950, Seattle 1945, etc. |
| raremaps.com GCS | 43 | DZI pyramids 20–33k px — need tile-assembly like usahas (incl. **LA 1976 r+v, desert era**) |
| U. Alabama cartweb | 19 | MrSID `getimage` wid=5000 |
| NOAA historical charts | 17 | 1932–35 CAA + 1950 batch remainder |
| FAA `chart_sample_files` | 11 | incl. the 317 MB AOD LA separation-plates zip |
| NLA Sheila Scott collection | 10 | annotated 1965–71 US sectionals, `/image` 5000px |
| Institutional one-offs (LOC probes, UWM, Leventhal, OK State, Newberry, Birmingham, El Paso, Ohio, UNT/UTA, curtiswright, NLA Whiting, archive.org items) | 43 | mixed eras, mostly pre-1971 |

Era shape of the pending set: **773 post-2010 · 136 pre-1971 · 5 in-era 1971–2010 ·
395 undated-in-CSV** (mostly wayback PDFs whose `date_or_edition` is an edition string —
the URL carries the cycle date).

## Suggested batching

1. **LOC gct00498 (72)** — same pipeline as the existing ca-scans; biggest pre-1971 payload.
2. **Wayback old-layout zips (279)** — same handling as existing wayback zip rows; watch
   for truncated captures (take largest CDX capture; ~1 MB on a big file = reject).
3. **Wayback visual PDFs (760)** — needs the GeoPDF SRS fix from
   [01A](01_dole_slicer_failures.md) first, or they'll all fail the same way.
4. **raremaps DZI (43)** — write a small tile-stitcher (usahas mosaic code is the template).
5. The rest are small manual batches.

## Still-open acquisition lanes beyond this queue

Contacts / scan requests / logins / physical media — see
[03_web_sources_searched.md §2](03_web_sources_searched.md) and
`search_archive/ACTION_PLAN.md` Tiers 2–6. The email payloads (ASU barcodes, GlidePlan
inventory, Welch/Fox file lists, AVSIM download IDs) are in
`search_archive/dole_search_lane_reports/`.

## How to regenerate

```bash
~/venv/bin/python - <<'EOF'
import csv
dole = {(r['download_link'] or '').strip() for r in csv.DictReader(open('master_dole_v2.csv'))}
online = list(csv.DictReader(open('worklists/search_archive/missing_from_dole_online.csv')))
print(sum((r['url'] or '').strip() not in dole for r in online), 'pending')
EOF
```

## 2026-07-24 — full queue downloaded and analyzed

All 1,517 rows pulled to `/Volumes/projects/rawtiffcandidates/` (183 GB, per-source
subdirs). Row-level results: `_download_manifest.csv` / `_manifest.jsonl` (url, sha1,
bytes, status) and `_analysis.csv` / `_analysis_report.md` (dims, dpi, georef, verdict).
No byte-identical duplicates in the set.

**Verdicts:** 295 include (55 GB) · 1,021 FAA print-PDFs include-after-georef (130 GB) ·
158 already cataloged (+24 usahas KML indexes = the Tier-1 181) · 13 reference-only
(low-res) · 6 dead.

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
- NOAA Data Sampler CD-ROM iso downloaded but unexplored (vector samples, likely
  reference-only).

Suggested inclusion order stays as §Suggested batching, with 3 replaced by:
sibling-georef the FAA print PDFs (both lanes), then 300 dpi convert per the
established PDF-source convention.

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
  SEC editions + 6 LOC ca-scans. Need dole rows only, no download.

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

To redo the batch later: `worklists/data/hunt26_move.py` (moves per the plan,
handles the jp2→LZW-tif conversion that strips the bogus Greenwich georef and
the .php→.jpg renames) then `scripts/import_hunt26.py` (idempotent — skips the
6 already-present zips). Georef-tool sidebar items G1–G8 for the 229-row
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

Remaining on the candidates volume besides these: the 1,021 edition-dups, 158
already cataloged, 13 reference-only, and the 31 disk-but-uncataloged rows
(still need dole rows only).
