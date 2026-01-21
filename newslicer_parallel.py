#!/usr/bin/env python3
"""
Parallel wrapper for newslicer.py - Process multiple dates simultaneously
Usage: ./newslicer_parallel.py --workers 3
"""
import subprocess
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

def process_single_date(date: str, args: argparse.Namespace) -> tuple:
    """Process a single date using newslicer.py"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "newslicer.py"),
        "-s", str(args.source),
        "-o", str(args.output),
        "-c", str(args.csv),
        "-b", str(args.shapefiles),
        "-f", args.format,
        "-z", args.zoom,
        "-r", args.resample,
    ]

    # Filter to only process this specific date
    # We'd need to modify newslicer.py to accept a --date-filter argument
    # For now, this is a template

    print(f"[Worker] Starting date: {date}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            print(f"[Worker] ✓ Completed: {date}")
            return (date, True, None)
        else:
            print(f"[Worker] ✗ Failed: {date}")
            return (date, False, result.stderr[:500])
    except subprocess.TimeoutExpired:
        print(f"[Worker] ✗ Timeout: {date}")
        return (date, False, "Timeout after 2 hours")
    except Exception as e:
        print(f"[Worker] ✗ Error: {date} - {e}")
        return (date, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Process multiple dates in parallel",
        epilog="""
Example:
  %(prog)s --workers 3

  This will process 3 dates simultaneously, fully utilizing your CPU.
        """
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dates to process in parallel (default: 2, recommend: 2-3 for 30%% CPU usage per task)"
    )

    parser.add_argument("-s", "--source", type=Path, default=Path("/Volumes/drive/newrawtiffs"))
    parser.add_argument("-o", "--output", type=Path, default=Path("/Volumes/drive/sync"))
    parser.add_argument("-c", "--csv", type=Path, default=Path("/Users/ryanhemenway/archive.aero/master_dole.csv"))
    parser.add_argument("-b", "--shapefiles", type=Path, default=Path("/Users/ryanhemenway/archive.aero/shapefiles"))
    parser.add_argument("-f", "--format", type=str, default='geotiff', choices=['geotiff', 'tiles', 'both'])
    parser.add_argument("-z", "--zoom", type=str, default='0-11')
    parser.add_argument("-r", "--resample", type=str, default='bilinear', choices=['nearest', 'bilinear', 'cubic', 'cubicspline'])

    args = parser.parse_args()

    # TODO: Get list of dates from CSV
    dates_to_process: List[str] = []  # Load from CSV

    print(f"Processing {len(dates_to_process)} dates with {args.workers} parallel workers")
    print(f"Expected speedup: {args.workers}x (if I/O allows)\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_date, date, args): date for date in dates_to_process}

        completed = 0
        failed = 0
        for future in as_completed(futures):
            date, success, error = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                print(f"Error for {date}: {error}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {completed} completed, {failed} failed")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
