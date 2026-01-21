#!/bin/bash
#
# Parallel Date Processing for newslicer.py
# This script processes multiple dates simultaneously to maximize CPU usage
#
# Usage:
#   ./parallel_process.sh                    # Process 3 dates at a time (default)
#   ./parallel_process.sh 2                  # Process 2 dates at a time
#   DATES="2024-01-15 2024-02-20" ./parallel_process.sh  # Process specific dates
#

set -e

# Configuration
WORKERS=${1:-3}  # Number of parallel processes (default: 3)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWSLICER="$SCRIPT_DIR/newslicer.py"

# Get dates to process (either from env var or from CSV)
if [ -n "$DATES" ]; then
    # Use user-provided dates
    DATE_LIST=($DATES)
else
    # Extract all dates from master_dole.csv (skip header, get unique dates)
    CSV_FILE="${CSV_FILE:-$SCRIPT_DIR/master_dole.csv}"
    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: CSV file not found: $CSV_FILE"
        echo "Set CSV_FILE environment variable or use DATES to specify dates manually"
        exit 1
    fi

    echo "Extracting dates from $CSV_FILE..."
    mapfile -t DATE_LIST < <(tail -n +2 "$CSV_FILE" | cut -d',' -f1 | sort -u)
fi

TOTAL_DATES=${#DATE_LIST[@]}

if [ $TOTAL_DATES -eq 0 ]; then
    echo "ERROR: No dates found to process"
    exit 1
fi

echo "=========================================="
echo "Parallel newslicer.py Processing"
echo "=========================================="
echo "Total dates: $TOTAL_DATES"
echo "Parallel workers: $WORKERS"
echo "Compression: DEFLATE (use --compression to override)"
echo ""
echo "Expected speedup: ${WORKERS}x (compared to sequential)"
echo "Expected CPU usage: 90-100% (vs 30% for single process)"
echo ""
echo "Dates to process:"
printf '%s\n' "${DATE_LIST[@]}" | head -10
if [ $TOTAL_DATES -gt 10 ]; then
    echo "... and $((TOTAL_DATES - 10)) more"
fi
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Create a function to process a single date
process_date() {
    local date=$1
    local log_dir="$SCRIPT_DIR/logs"
    mkdir -p "$log_dir"

    local log_file="$log_dir/newslicer_${date}.log"

    echo "[$(date +'%H:%M:%S')] Starting: $date (log: $log_file)"

    # Run newslicer for this specific date
    # Pass through any additional arguments from environment
    python3 "$NEWSLICER" \
        --date-filter "$date" \
        --compression "${COMPRESSION:-DEFLATE}" \
        ${EXTRA_ARGS} \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date +'%H:%M:%S')] ✓ Completed: $date"
    else
        echo "[$(date +'%H:%M:%S')] ✗ Failed: $date (check $log_file)"
    fi

    return $exit_code
}

export -f process_date
export SCRIPT_DIR
export NEWSLICER
export COMPRESSION
export EXTRA_ARGS

# Process dates in parallel using GNU parallel (or xargs if parallel not available)
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for processing..."
    printf '%s\n' "${DATE_LIST[@]}" | parallel -j "$WORKERS" --bar process_date {}
    exit_code=$?
else
    echo "GNU parallel not found, using xargs (no progress bar)..."
    echo "Install with: sudo apt-get install parallel (or brew install parallel)"
    printf '%s\n' "${DATE_LIST[@]}" | xargs -P "$WORKERS" -I {} bash -c 'process_date "$@"' _ {}
    exit_code=$?
fi

echo ""
echo "=========================================="
echo "Processing Complete"
echo "=========================================="
echo "Check individual logs in: $SCRIPT_DIR/logs/"
echo ""

# Count successes and failures
success_count=0
fail_count=0

for date in "${DATE_LIST[@]}"; do
    log_file="$SCRIPT_DIR/logs/newslicer_${date}.log"
    if [ -f "$log_file" ]; then
        if grep -q "COMPLETE" "$log_file"; then
            ((success_count++))
        else
            ((fail_count++))
            echo "Failed: $date (see $log_file)"
        fi
    fi
done

echo "Summary: $success_count succeeded, $fail_count failed"

exit $exit_code
