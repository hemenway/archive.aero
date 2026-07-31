# slicer-go — native Go port of scripts/slicer.py

Feature-parity port of the FAA chart slicer (download → warp/cut → mosaic →
one GeoTIFF per date). The Python original stays authoritative at
`scripts/slicer.py`; this port exists for speed, robustness and single-binary
deployment.

## Build & run

```
cd slicer-go
go build -o slicer .
./slicer --help
./slicer --start-date 2017-02-02 --end-date 2017-02-02 -y
```

Flags mirror the Python CLI (`-s/-o/-c/-b/-t`, `--warp-output`,
`--compression`, `--parallel-warp`, `--download-delay`, `--all`,
`--start-date/--end-date`, `-y`). One syntax difference:
`--clip-projwin "ULX ULY LRX LRY"` takes a single quoted string instead of
four separate arguments.

## Design

All raster/vector work is delegated to the Homebrew GDAL CLI tools — one
subprocess per operation — instead of in-process bindings:

| Python (osgeo)                     | Go                              |
|------------------------------------|---------------------------------|
| `gdal.Open` metadata probes        | `gdalinfo -json`                |
| `gdal.Warp`                        | `gdalwarp`                      |
| `gdal.Translate`                   | `gdal_translate`                |
| `gdal.BuildVRT`                    | `gdalbuildvrt`                  |
| `osr.CoordinateTransformation`     | `gdaltransform` (stdin points)  |
| shapefile SRS → proj4              | `gdalsrsinfo -o proj4`          |
| ogr ring reads                     | `ogrinfo -json -features`       |

Why subprocesses: GDAL does the heavy pixel work either way; per-process
execution removes the GIL, isolates block caches per worker, and a crashing
warp can no longer take down the whole run. `gdal.SetConfigOption` calls
become per-invocation environment variables (`GDAL_CACHEMAX`,
`GDAL_NUM_THREADS`, …).

Requires `brew install gdal` (tested against GDAL 3.12.2). The tool checks
for all seven binaries at startup and fails fast if any is missing.

## Verified parity

Tested 2026-07-23 against the Python original on the three date ranges
starting 2017-02-02 (5 charts: GCP warp, shapefile-cutline warp, paletted
RGBA-expand, single- and multi-chart mosaics):

- `2017-02-02_to_2018-02-01.tif` — byte-for-byte identical
- `2017-02-02_to_2017-07-20.tif`, `2017-02-02_to_2017-08-17.tif` — identical
  size and identical per-band pixel checksums (`gdalinfo -checksum`); only
  TIFF header bytes differ (CLI vs API writer ordering)

CSV intake matches exactly: 7,458/7,458 rows → 3,756 date ranges.

## Known divergences (intentional)

- `/vsimem/` intermediates (RGBA source VRTs, GCP VRTs) become small on-disk
  temp files next to the warp output, deleted after use — subprocesses cannot
  share vsimem.
- The warp phase gives each worker subprocess `available/workers²` MB of GDAL
  cache so the phase-wide total matches the Python original's single shared
  cache of `available/workers`.
- The mosaic progress meter parses `gdalwarp`'s stdout ticker instead of a
  progress callback; ETA log lines are equivalent.
- `requests`-style per-chunk timeouts are approximated with header/dial
  timeouts (no overall body deadline).

The full flow (identical in both implementations) is diagrammed in the
"slicer.py — Anatomy of the Chart Pipeline" artifact.
