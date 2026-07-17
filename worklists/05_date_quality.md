# Worklist 05 — Date-quality flags in the dole

*Generated 2026-07-16 from `master_dole_v2.csv` `note` column. Row-level detail:
[`data/date_quality.csv`](data/date_quality.csv) (535 flagged rows, one line per flag).*

Every row has a `date` and `end_date` (the 2026-07-14 end-date fill closed all blanks —
see `search_archive/end_date_fill_report.csv` for how each was derived), but three
classes of rows carry uncertainty flags that affect timeline accuracy:

| Flag | Rows | Meaning | Fix path |
|---|---|---|---|
| `END-ESTIMATED` | 259 | `end_date` guessed from typical edition cadence (`typNNNd` in note) — no next edition was in the dole to anchor it | Each one resolves automatically when the *next* edition of that chart is found and cataloged; otherwise verify against edition tables (LOC/NOAA cartobibliographies) |
| `GAP` | 236 | Long hole before the next known edition (`GAP NNNd before next`) — editions almost certainly existed in between and are missing from the dole | These are **search targets**, not data errors — feed the biggest ones into the hunt (see [03_web_sources_searched.md](03_web_sources_searched.md)) |
| `DATE-APPROX` | 40 | Date read from context, not printed on the chart (e.g. "ca. 2004-05 per AVSIM", usahas "mid-2009") | Firm up only if a dated duplicate surfaces; low priority |

Minor flags: `END-FROM-NEXT-EDITION` (10) and `BLANK_MAP` (6) are informational, not defects.

## Biggest GAPs (highest-value search targets)

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

Full sorted list: `grep GAP data/date_quality.csv` (the `note` column carries the day count).
