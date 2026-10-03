# Worklist 05 — Date-quality flags in the dole

**Updated 2026-09-21** from the current **7,672-row** catalog, loaded through
`scripts/dole_v2.py`. No rows have blank `date` or `end_date` values. The three
primary note flags total **544 flag occurrences on 301 distinct rows**; flags
can overlap. These count retained note text, not a new validation of every date.

The linked [data/date_quality.csv](data/date_quality.csv) is the **2026-07-16
snapshot** (535 flag occurrences), not a regenerated current export.

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
