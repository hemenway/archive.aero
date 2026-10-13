# Worklist 05 — Date-quality flags in the dole

**Updated 2026-09-21** from the current **7,672-row** catalog, loaded through
`scripts/dole_v2.py`. No rows have blank `date` or `end_date` values. The three
primary note flags total **544 flag occurrences on 301 distinct rows**; flags
can overlap. These count retained note text, not a new validation of every date.

The linked [data/date_quality.csv](data/date_quality.csv) was **regenerated
2026-10-12** from the live catalog (311 rows; END-ESTIMATED 263, GAP 236,
DATE-APPROX 45, END-FROM-NEXT-EDITION 14, BLANK_MAP 6).

## 2026-10-12 reconciliation against the current catalog

Every `GAP … before next (DATE)` note was recomputed against the editions the
catalog holds today (same location and half, next later date), and every
`END-ESTIMATED` row was tested with the July rule recovered from
`end_date_fill_report.csv` (snap to the next edition when its start is within
~1.1 × the location's typical period, else keep date + period):

- **230 of 236 GAP notes unchanged** — no edition found since July closes them;
  they remain search targets, not data errors.
- **6 notes stale, reconciled** (backup `pre_date_quality_reconcile_2026-10-12.csv`):
  Portland OR 1947-04-20 snapped to `ca003021` 1947-10-23 (was estimated
  1947-10-16); Beaumont 1960, Butte 1939 and Savannah 1941 already had
  `end_date` = the next edition (rechained when it was catalogued) but kept the
  old GAP text, now `END-FROM-NEXT-EDITION`; Cleveland 1940 (2,497 → 1,350 d,
  WASP 1943 scan) and San Francisco 1944 (556 → 391 d) keep END-ESTIMATED with
  the gap re-measured.
- **DATE-APPROX (45):** no dated duplicate has surfaced for any of them. The
  2009 ChartGeek mosaics sit a year before the simviation 2010 rows (different
  editions); the 2004 archive.org/AVSIM Hawaiian pair and the 2004 Alaska
  east/west JPGs are known same-set alternatives. Nothing to change.
- 14 END-ESTIMATED rows have no later edition at all (ends of discontinued
  series); their estimate is the best available.

Every row has a `date` and `end_date` (the 2026-07-14 end-date fill closed all blanks —
see `search_archive/end_date_fill_report.csv` for how each was derived), but three
classes of rows carry uncertainty flags that affect timeline accuracy:

| Flag | Rows | Meaning | Fix path |
|---|---|---|---|
| `END-ESTIMATED` | 263 | `end_date` guessed from typical edition cadence (`typNNNd` in note) — no next edition was in the dole to anchor it | When a next edition is found, review/recompute the end date and reconcile the note; cataloging alone does not prove the flag was cleared. Otherwise verify against edition tables (LOC/NOAA cartobibliographies) |
| `GAP` | 236 | Long hole before the next known edition (`GAP NNNd before next`) — editions almost certainly existed in between and are missing from the dole | These are **search targets**, not data errors — feed the biggest ones into the hunt (see [03_web_sources_searched.md](03_web_sources_searched.md)) |
| `DATE-APPROX` | 45 | Date read from context, not printed on the chart (e.g. "ca. 2004-05 per AVSIM", usahas "mid-2009") | Firm up only if a dated duplicate surfaces; low priority |

Minor flags: `END-FROM-NEXT-EDITION` (12) and `BLANK_MAP` (6) are informational, not defects.

## Largest retained GAP notes (unchanged in the current catalog)

| Gap | Chart | After edition dated | Row |
|---|---|---|---|
| 2,497 d | Cleveland, OH | 1940-03-01 | ca000928.tif |
| 1,961 d | San Antonio, TX | 1961-02-08 | ca003342r.tif |
| 1,884 d | Boise, ID | 1960-03-30 | ca000393r.tif |
| 1,856 d | New Orleans, LA | 1961-03-01 | ca002634r.tif |
| 1,842 d | Lake Huron, MI | 1960-05-11 | ca001961r.tif |
| 1,827 d | Aroostook, ME | 1935-11-01 | ca000186.tif |
| 1,823 d | Aroostook, ME | 1960-05-02 | ca000161r.tif |
| 1,822 d | Burlington, VT | 1960-05-03 | ca000493r.tif |
| 1,821 d | Sioux City, IA | 1960-06-01 | ca003645r.tif |
| 1,807 d | Lewiston, ME | 1960-05-18 | ca002078r.tif |

Note the cluster of ~1,800-day gaps starting 1960-61: that is the LOC collection thinning
out at its end, not lost editions of individual charts — the early-1960s era needs a
*collection-level* source (NLA Australia Bib 1030946 is the known lever, ACTION_PLAN Tier 4).

Next action: rank current `GAP` notes from `dole_v2.load_rows()` and reconcile
newly found editions before changing estimates. The table is a search lead, not
a recomputed edition-coverage proof; use [03](03_web_sources_searched.md) and the
current holdings to avoid repeating already-closed searches.
