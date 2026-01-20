#!/bin/bash
# unzip_all.sh - Unzip all chart files in parallel
#
# Usage: ./unzip_all.sh [output_directory] [max_parallel_jobs]
#
# Examples:
#   # Unzip into source directory (same location as ZIPs)
#   ./unzip_all.sh /Volumes/drive/newrawtiffs 4
#
#   # Unzip into separate directory
#   ./unzip_all.sh /Volumes/drive/newrawtiffs_unzipped 4

set -e

SOURCE_DIR="/Volumes/drive/newrawtiffs"
OUTPUT_DIR="${1:-$SOURCE_DIR}"
PARALLEL_JOBS="${2:-4}"

# Ensure we can find parallel command
if ! command -v parallel &> /dev/null; then
    echo "Installing GNU parallel..."
    brew install parallel 2>/dev/null || {
        echo "ERROR: GNU parallel not found and could not install"
        echo "Install with: brew install parallel"
        exit 1
    }
fi

echo "=== Archive.aero Chart Unzipper ==="
echo "Source directory: $SOURCE_DIR"
echo "Extract to: $OUTPUT_DIR"
echo "Parallel jobs: $PARALLEL_JOBS"

if [ "$SOURCE_DIR" = "$OUTPUT_DIR" ]; then
    echo "Mode: Inline (directories created in source directory)"
else
    echo "Mode: Separate (directories created in output directory)"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count ZIP files
ZIP_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.zip" | wc -l)
echo "Found $ZIP_COUNT ZIP files to extract"
echo ""

# Function to unzip a single file
unzip_file() {
    zipfile="$1"
    output_dir="$2"
    zip_basename=$(basename "$zipfile" .zip)

    # Extract to named subdirectory
    # If directory already exists, skip (allows resuming interrupted extractions)
    if [ -d "$output_dir/$zip_basename" ]; then
        SIZE=$(du -sh "$output_dir/$zip_basename" | cut -f1)
        echo "⊘ $zip_basename (already extracted - $SIZE)"
        return 0
    fi

    unzip -q "$zipfile" -d "$output_dir/$zip_basename" 2>/dev/null || {
        echo "ERROR: Failed to extract $zipfile"
        return 1
    }

    echo "✓ $zip_basename ($(du -sh "$output_dir/$zip_basename" | cut -f1))"
}

export -f unzip_file

# Unzip in parallel
echo "Starting parallel extraction..."
echo ""

find "$SOURCE_DIR" -maxdepth 1 -name "*.zip" | \
    parallel --no-notice -j "$PARALLEL_JOBS" unzip_file {} "$OUTPUT_DIR"

echo ""
echo "=== Extraction Complete ==="

# Calculate sizes
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
FILE_COUNT=$(find "$OUTPUT_DIR" -name "*.tif" | wc -l)

echo "Total extracted size: $TOTAL_SIZE"
echo "Total TIF files: $FILE_COUNT"
echo ""

if [ "$SOURCE_DIR" = "$OUTPUT_DIR" ]; then
    echo "Next steps (inline extraction):"
    echo "  1. ZIP and unzipped directories now coexist in: $OUTPUT_DIR"
    echo "  2. Run: newslicer.py -s '$OUTPUT_DIR'"
    echo "  3. (Optional) Delete ZIPs to free 100 GB: rm -rf '$SOURCE_DIR'/*.zip"
else
    echo "Next steps (separate extraction):"
    echo "  1. Verify extraction: ls -la '$OUTPUT_DIR' | head"
    echo "  2. Run: newslicer.py -s '$OUTPUT_DIR'"
fi
echo ""
