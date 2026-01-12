#!/usr/bin/env bash
set -euo pipefail

IN_ROOT="/Volumes/projects/trial"
OUT_DIR="/Users/ryanhemenway/Desktop/upload"
R2_DEST="r2:sectionals"

# Threads
export GDAL_NUM_THREADS=8

mkdir -p "$OUT_DIR"

echo "Searching for mosaic TIFFs under: $IN_ROOT"

found_any=false

find "$IN_ROOT" -type f \( -iname '*mosaic*.tif' -o -iname '*mosaic*.tiff' \) -print0 |
while IFS= read -r -d '' IN; do
  found_any=true

  base="$(basename "$IN")"
  stem="${base%.*}"

  # Extract ISO date (YYYY-MM-DD) from filename or path
  iso_date="$(printf '%s' "$base" | grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -n 1 || true)"
  if [[ -z "$iso_date" ]]; then
    iso_date="$(printf '%s' "$IN" | grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -n 1 || true)"
  fi

  if [[ -n "$iso_date" ]]; then
    OUT_NAME="${iso_date}.tif"
  else
    OUT_NAME="${stem}.tif"
  fi

  OUT="$OUT_DIR/$OUT_NAME"

  # Skip if already up-to-date
  if [[ -f "$OUT" && "$OUT" -nt "$IN" ]]; then
    echo "Skipping (up to date): $OUT_NAME"
    continue
  fi

  echo
  echo "=== Converting ==="
  echo "IN : $IN"
  echo "OUT: $OUT"

  gdal_translate \
    -of COG \
    -co COMPRESS=DEFLATE \
    -co PREDICTOR=2 \
    -co BIGTIFF=IF_SAFER \
    -co NUM_THREADS=8 \
    -co RESAMPLING=LANCZOS \
    "$IN" "$OUT"

done

echo
echo "=== Syncing upload folder to R2 ==="

rclone sync "$OUT_DIR" "$R2_DEST" \
  -P \
  --s3-upload-concurrency 16 \
  --s3-chunk-size 128M \
  --buffer-size 128M \
  --s3-disable-checksum \
  --stats 1s

echo
echo "All done."
