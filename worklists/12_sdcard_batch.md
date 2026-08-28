# Worklist 12 — 2026-08 SD-card batch (five iFly EFB cards)

*Imaged 2026-08-20 + 2026-08-27 with `scripts/sd_slurp.py`; mined 2026-08-27.
Images (the permanent as-found record): `/Volumes/ACASIS/All SD cards/2026-08-20/`.*

## The cards

| seq | vol UUID | size | era (newest chart start) | image |
|---|---|---|---|---|
| 001 | df16946b | 7.9 GB | Dec 2017 card (eds → 2018-02) | complete |
| 003 | (none) | 7.9 GB | **2014-15 cycle** (eds → 2015-04) | complete |
| 004 | 44251c7e | 15.9 GB | fall 2017 (eds → 2017-11) | complete |
| 006 | 68d74d30 | 7.9 GB | late 2018 (eds → 2018-12) | complete |
| 007 | 82d820c0 | 15.9 GB | spring 2018, **stale 2014-15 AK set** | **partial 44.62%** — NAND failure above 7.109 GB, controller EOF-fences the rest; see the batch README on ACASIS |

## Method

- Read-only FAT32 walk of each image (`scripts/fatwalk.py`, new; verified
  byte-identical against an OS mount). Live **and deleted** directory entries
  enumerated for VfrSec/VfrTac/VfrWac on every card.
- Harvested all 483 `.gtj` georef headers first, built the
  (type, location, edition, start) matrix, and deduped **before** extracting
  anything, per the mining order in CLAUDE.md.
- Dedupe universe: `master_dole_v2.csv` (7,587 rows pre-import), every attic
  TAC/WAC set, `acasis_imgcv2_FAA_originals` (full-res, supersedes iFly
  copies), NARA rg-237 (2016+), and all 1,517 verified findings of the
  2026-07 hunt (`search_archive/missing_from_dole_online.csv`, incl. its
  wayback CDX sweeps).
- **Format discovery:** the 2014-15 sectionals use a fil2 variant whose
  offsets index a pre-compression layout (~2.5x the `.dat`, sentinel ≠ file
  size) — the `.dat` is the same tiles as back-to-back JPEGs in record order,
  257 px with 1 px overlap. `scripts/ifly_extract.py` now auto-detects this
  and maps tiles sequentially (`format: fil2-sequential` in manifests). Same
  card's WACs and all 2017/2018 bundles use real offsets (the proven path).
- Georef: **EPSG:3395 ellipsoidal Mercator** bbox (the 2026-08-24 iFly-grid
  finding). Initially extracted as 3857 — the KSEA landmark test only rules
  out linear-latitude (81 px) and cannot see the 3857/3395 difference — then
  the height test was replicated on this batch's own `.gtj` headers
  (3395 fits nominal height within ±0.6 px on every chart; 3857 misses by
  +3..+62 px) and **all 35 GeoTIFFs were regenerated as 3395 on 2026-08-27**
  before slicing. Dole notes updated; 5 neighbour rows' stale end_dates
  (chained to next-then-known editions) corrected per the 08-26 lesson.

## Yield

**Catalog + rawtiffs (18 rows, 18 tifs; only-copy):**
- `rawtiffs/ifly_efb_card_2015/` — 18 editions of the 2014-15 gap era from
  card 003: Albuquerque 94, Anchorage 95, Chicago 89, Cincinnati 93,
  Fairbanks 95, Great Falls 88, Green Bay 89, Hawaiian Islands 91,
  Kansas City 93, Los Angeles 96, New Orleans 95, New York 90, Phoenix 92,
  San Antonio 94, Seattle 88, Seward 95, St Louis 91, Twin Cities 89.
  Every row's derived end_date = the card expiry + 1 day, i.e. each fills an
  exact one-edition hole between existing catalog rows.
- Import: `scripts/import_ifly_sdcards.py` (CSV backup
  `pre_ifly_sdcards_2026-08-27_202544.bak`); dole 7,587 → 7,605 rows net
  (19 imported, Western Aleutian retracted same day — see below).

**Attic (out of catalog scope):**
- `ifly_sdcard_wac_2014-15/` + `geotiff_wac_2014-15/` — 13 intermediate WAC
  editions from card 003 (its VfrTac dir held WACs), between the 2013 set
  and the finals.
- `ifly_sdcard_tac_honolulu_inset/` + `geotiff_tac_honolulu_inset/` —
  Honolulu Inset TAC eds 97 (card 001) + 99 (card 006).
- `ifly_sdcard_sec_2014-15_2017/` — the source bundle trios for the 20
  rawtiffs sectionals; extraction manifests sit in each rawtiffs parent. No
  side staging dir was kept (rule: extracted/derived files live in
  rawtiffs / rawtiffs_attic directly).

**Everything else was a dupe:** 171 of 202 sectional tuples already
catalogued; all 2017-18 city TACs covered by `acasis_imgcv2_FAA_originals`
(full-res) or the 2019/2020 attic sets; the 22-chart WAC final set identical
on cards 001/004/006 to `geotiff_vfrwac_final`; Grand Canyon TAC 3 already in
the 2020 set.

## Deliberately not staged

- **Western Aleutian ed 50 (RETRACTED) and ed 51**: both editions turn out
  to already be catalogued as NARA rg-237 full-res rows under the per-half
  location names "Western Aleutian Islands East"/"...West". The batch dedupe
  normalized card locations to the combined "Western Aleutian Islands" name
  (which the 2013-era rows use) and never saw them, so ed 50 was imported as
  `ifly_efb_card_2017_…` — caught pre-slice while planning era membership,
  row retracted + rawtiffs folder deleted the same day (backup
  `pre_wai_retract_2026-08-27_210951.bak`). The E/W bundle trios remain in
  the attic as salvage record. **Lesson: (location, edition) dedupe must
  also match variant location spellings — the catalog names AK halves
  inconsistently across eras (combined pre-2016, per-half NARA-era).**
- **Atlanta 94, Brownsville 94, Charlotte 97, Dutch Harbor 50, Halifax 92,
  Jacksonville 95, Miami 96, Washington 97, Wichita 94**: the attic's
  `acasis_imgcv2_FAA_originals/SEC` holds full-res FAA originals of all nine
  — they should enter the catalog from there (their own import), not from
  0.9x iFly copies. ← open item
- Card 007's Salt Lake City 99 / San Antonio 101 (2018): in the image's dead
  zone, and both editions already catalogued from FAA sources.

## Deleted-file check

Card 003's VfrTac also holds **77 deleted entries — a 2015-03 city TAC set**
(26 charts, Colorado Springs → Tampa). All clusters reallocated before
imaging (`fatwalk carve`: 0/77 clean); names recorded, data gone. No other
deleted chart entries on any card.

## Coverage statement

Examined: full live+deleted VfrSec/VfrTac/VfrWac trees on all five images
(007: VfrSec only — its VfrTac/VfrWac directories sit in the dead zone and
are unreadable). NOT examined: Charts/ (plates), EnrHigh/EnrLow, AptDiagrams,
AptImages, Data, Navi, StaticData, WxData, iFlyStreets, User, custom —
consistent with the 64 GB-card precedent (no enroute/plate collections);
and no unallocated-space signature carving beyond the directory-entry pass.

## Published (2026-08-27, same day)

Era keys are (date, end_date) pairs, so the batch resolved to **11 era
mosaics**: 7 NEW keys (`2014-08-13_to_2014-10-16` from the Hawaiian
boundary fix, `2014-10-16_to_2015-04-30`, `2014-11-13_to_2015-04-30`,
`2014-11-13_to_2015-05-28`, `2014-12-11_to_2015-05-28`,
`2014-12-11_to_2015-06-25`, `2015-01-08_to_2015-06-25`) and 4 same-key
rebuilds from the neighbour end_date fixes (`2014-05-29_to_2014-11-13`,
`2014-05-29_to_2014-12-11`, `2014-06-26_to_2014-12-11`,
`2014-06-26_to_2015-01-08` — worker-cache staleness ≤24 h applies to
those four). `2014-08-13_to_2015-02-25` emptied out and dropped from
dates.csv (its R2 object stays, per URI permanence). The four mover-era
temp dirs were wiped pre-run so removed charts could not leak from the
resume cache; 11 unaffected in-window keys were rebuilt but deliberately
not uploaded (content unchanged).

- Slicer window `2014-05-29..2015-01-08`, run dir
  `/Volumes/projects/2026-08-27 sdcard slicer run/` (22 mosaics built).
  Mosaic verification: 3 eras eyeballed (4-chart composite, 7-chart mover
  with New Orleans 94 moved in, Hawaiian 90 solo). One benign fallback:
  the georef-less `..._234930_` Hawaiian wayback duplicate failed as
  always; the slicer used the alternate row.
- Conversion: standing recipe (geotiff2pmtiles webp q80, auto zooms,
  8-bit). 11 archives, ~873 MB, uploaded via `rclone copyto` and
  size-verified.
- **18 chart artifacts** published (uploads.jsonl 7,289 → 7,307).
- dates.csv 3,733 → **3,739 eras**; live bundle
  `metadata-74c0969c.bundle` (built `--remote`, all 3,739 archives
  readable, 0 failures); timeline_data.json 7,609 charts republished
  (pm stamps verified for the new keys); coverage.json rebuilt.
- Live checks: bundle, a new era archive, and a new chart artifact all
  serve 206 from data.archive.aero.

## Still open

1. Import the nine FAA-original sectionals from the attic (separate import,
   full-res beats the cards).
2. Worklist 07 unchanged.
