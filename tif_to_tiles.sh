#!/usr/bin/env bash
set -euo pipefail

# Usage: script.sh input.tif [output_folder]
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 input.tif [output_folder]"
  exit 1
fi

INPUT="$1"
if [[ ! -f "$INPUT" ]]; then
  echo "Input not found: $INPUT"
  exit 1
fi

# If user didn't pass output folder as arg 2, ask via Finder dialog
OUTDIR="${2:-}"
if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$(osascript -e 'POSIX path of (choose folder with prompt "Choose an output folder for MBTiles/PMTiles:")')"
fi

# Normalize: remove trailing slash
OUTDIR="${OUTDIR%/}"
mkdir -p "$OUTDIR"

BASENAME="$(basename "$INPUT")"
BASENAME="${BASENAME%.*}"

MBTILES="$OUTDIR/${BASENAME}.mbtiles"
PMTILES="$OUTDIR/${BASENAME}.pmtiles"
LOGFILE="$OUTDIR/${BASENAME}_convert.log"

# Capture ALL output (including tool progress on stderr) to a log file
# so you can Quick Look/open it while the shortcut runs.
exec > >(tee -a "$LOGFILE") 2>&1

echo "Input:   $INPUT"
echo "Outdir:  $OUTDIR"
echo "Log:     $LOGFILE"
echo

echo "1/3 Converting -> MBTiles: $MBTILES"
# NOTE: don't use --quiet (it suppresses output) :contentReference[oaicite:0]{index=0}
gdal_translate -of MBTILES "$INPUT" "$MBTILES"

echo
echo "2/3 Building overviews on: $MBTILES"
# NOTE: don't use --quiet (it suppresses output) :contentReference[oaicite:1]{index=1}
gdaladdo -r bilinear "$MBTILES" 2 4 8 16 32 64 128 256 512 1024

echo
echo "3/3 Converting -> PMTiles: $PMTILES"
# pmtiles convert supports flags like --tmpdir / --no-deduplication :contentReference[oaicite:2]{index=2}
pmtiles convert "$MBTILES" "$PMTILES"

echo
echo "Done!"
echo "Outputs:"
echo "  $MBTILES"
echo "  $PMTILES"
echo "Log:"
echo "  $LOGFILE"
