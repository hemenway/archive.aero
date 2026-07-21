# Worklist 06 — Published site out of sync with the 2026-07-14 slicer run

**✅ EXECUTED 2026-07-21.** The full run was converted (geotiff2pmtiles, webp q80, auto
zooms to native max) and uploaded to `r2:charts/sectionals`, replacing same-key objects.
`dates.csv` now has 3,658 rows: 3,656 new-run keys + 2 retained old archives for the
damaged ranges (`2026-03-19_to_2026-05-14`, `2025-11-27_to_2026-01-22` — see worklist 01;
`2026-01-22_to_2026-03-19` stays unpublished, Wichita-only). 7 empty sparse mosaics
(2,253-byte 1941/1946–1950 ranges) were also skipped. 23 16-bit mosaics (1947–1961)
needed `-rescale linear -rescale-range 0,65535 -alpha-band 4`. The 209 stale-key objects
remain in R2 unreferenced (optional cleanup). coverage.json/bounds_cache rebuilt, 0
unreachable; first metadata bundle published (`bundleUrl` in index.html). Local mirror:
`/Volumes/drive/pmtiles`.

*Generated 2026-07-16 by diffing mosaic names in `/Volumes/projects/2026-07-14 slicer run/`
(3,666 tifs) against `dates.csv` (3,390 published pmtiles). Row-level detail:
[`data/publish_sync.csv`](data/publish_sync.csv).*

The 2026-07-14 run regenerated the full mosaic set **after** the catalog additions and the
end-date fill, so many date-range keys changed. The live site (`dates.csv` →
`data.archive.aero/sectionals/<range>.pmtiles`) still reflects the old set.

## The delta

| Status | Count | What it is |
|---|---|---|
| `NEW_UNPUBLISHED` | 485 | Mosaics in the new run with no published pmtiles under that key. Heaviest decades: 1950s (226), 1960s (96), 1940s (82), 1930s (70) — i.e. the end-date fill renamed a large slice of the historical set, plus genuinely new content from the 189 cataloged downloads. |
| `PUBLISHED_STALE_KEY` | 209 | Published pmtiles whose key no longer exists in the run output. 189 are old **bare-date names** (`1930-12-01` vs new `1930-12-01_to_1932-08-01`) — the old naming convention; 20 are ranged keys whose end date changed. |

## Work to do

1. **Convert + upload the 485** through `geotiff2pmtiles` → R2.
   ⚠ Mosaic compression must stay LZW/DEFLATE — the Go reader can't decode ZSTD.
2. **Update `dates.csv`** to the new key set (drop the 209 stale keys, add the 485).
3. **Delete or leave stale R2 objects** — decide: leaving them wastes storage but breaks
   nothing once dates.csv stops referencing them; deleting is cleaner. (Check nothing else
   hard-codes the bare-date URLs.)
4. **Regenerate `coverage.json`** (`scripts/build_coverage.py`, last generated 2026-07-07 — predates
   the run) and `timeline_data.json` / `bounds_cache.json` (cache currently holds 3,201 keys,
   also pre-run).
5. **Verify reachability after upload** — `build_coverage.py` prints a WARNING listing any
   range in dates.csv whose pmtiles is unreachable; the current bounds_cache has 0 unreachable
   entries, but re-check after the sync (there was historically a set of ~17 missing-from-R2
   1950s pmtiles; confirm it stays closed).
6. Remember the serving constraints when uploading big files: >512 MB pmtiles bypass the CF
   cache on the raw R2 domain — everything should route through the `tiles` Worker
   (see memory notes / commit 373fbda hardening).

## Partial mosaics warning

The newest 3 cycles' mosaics from this run are **partial** (GeoPDF SRS failures during the
run — see [01_dole_slicer_failures.md](01_dole_slicer_failures.md)). Fix those before
publishing the affected ranges, or you'll ship holes in the freshest charts.

## How to regenerate the delta

```bash
~/venv/bin/python - <<'EOF'
import csv, os
run = {f[:-4] for f in os.listdir('/Volumes/projects/2026-07-14 slicer run') if f.endswith('.tif')}
pub = {r['date_iso'] for r in csv.DictReader(open('dates.csv'))}
print(len(run-pub), 'unpublished;', len(pub-run), 'stale')
EOF
```
