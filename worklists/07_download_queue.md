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
