#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT="/Volumes/drive/sync"
REMOTE="r2:charts/sectionals"

# Parallel file workers (start with 2–4; tune upward if RAM/disk can handle it)
JOBS="${JOBS:-3}"

# rclone tuning (your flags + sensible companions)
RCLONE_FLAGS=(
  -P
  --s3-upload-concurrency 16
  --s3-chunk-size 128M
  --buffer-size 128M
  --s3-disable-checksum
  --stats 1s
)

for cmd in gdal_translate gdaladdo pmtiles rclone xargs; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing: $cmd" >&2; exit 127; }
done

process_one() {
  local in="$1"
  local dir base stem mb pm
  dir="$(dirname "$in")"
  base="$(basename "$in")"
  stem="${base%.*}"
  mb="$dir/${stem}.mbtiles"
  pm="$dir/${stem}.pmtiles"

  echo "==> [$BASHPID] Processing: $in"

  # TIFF -> MBTiles
  gdal_translate -of MBTILES "$in" "$mb"

  # Overviews
  gdaladdo -r bilinear "$mb" 2 4 8 16 32 64 128 256

  # MBTiles -> PMTiles
  pmtiles convert "$mb" "$pm"

  # Remove MBTiles
  rm -f "$mb"

  # Upload PMTiles (flat into REMOTE)
  rclone copyto "$pm" "$REMOTE/$(basename "$pm")" "${RCLONE_FLAGS[@]}"

  echo "✔ [$BASHPID] Done: $in"
}

export -f process_one
export ROOT REMOTE
export RCLONE_FLAGS

# Find TIFFs, excluding any path containing '/.temp/'
find "$ROOT" -type f \( -iname '*.tif' -o -iname '*.tiff' \) -not -path '*/.temp/*' -print0 \
| xargs -0 -n 1 -P "$JOBS" bash -lc 'process_one "$0"'


