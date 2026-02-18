#!/usr/bin/env python3
"""
Download missing TIFFs listed in early_master_dole.csv into a target directory.

Rules:
- Skip any row with missing filename/link.
- Skip download if filename already exists anywhere under target root.
- Download missing files to the target root directory.
- Optional: skip files ending in v.tif.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync missing early_master_dole TIFF files")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("early_master_dole.csv"),
        help="Path to early_master_dole.csv",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/Volumes/projects/rawtiffs"),
        help="Destination directory",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on downloads (0 = all missing)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay between successful downloads (seconds)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Socket timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per file",
    )
    parser.add_argument(
        "--skip-v",
        action="store_true",
        help="Skip files whose name ends with v.tif",
    )
    return parser.parse_args()


def load_expected(csv_path: Path) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("tif_filename") or "").strip()
            link = (row.get("download_link") or "").strip()
            if not name or not link:
                continue
            # Keep first link if duplicates ever appear.
            expected.setdefault(name, link)
    return expected


def existing_names(root: Path) -> set[str]:
    return {p.name for p in root.rglob("*") if p.is_file()}


def download_file(url: str, target: Path, timeout: int, retries: int) -> Tuple[bool, str]:
    headers = {"User-Agent": USER_AGENT}
    temp_target = target.with_name(target.name + ".part")
    backoff = 3

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp, temp_target.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            temp_target.replace(target)
            return True, ""
        except urllib.error.HTTPError as e:
            if temp_target.exists():
                temp_target.unlink()
            if e.code in (403, 404):
                return False, f"HTTP {e.code}"
            if attempt == retries:
                return False, f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            if temp_target.exists():
                temp_target.unlink()
            if attempt == retries:
                return False, str(e)

        time.sleep(backoff)
        backoff *= 2

    return False, "unknown error"


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    dest = args.dest.resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    if not dest.exists() or not dest.is_dir():
        print(f"ERROR: Destination missing or not a directory: {dest}")
        return 1

    expected = load_expected(csv_path)
    existing = existing_names(dest)
    missing: List[Tuple[str, str]] = [(name, url) for name, url in expected.items() if name not in existing]

    skipped_v = 0
    if args.skip_v:
        before = len(missing)
        missing = [(name, url) for name, url in missing if not name.lower().endswith("v.tif")]
        skipped_v = before - len(missing)

    print(f"CSV: {csv_path}")
    print(f"Destination: {dest}")
    print(f"Expected unique filenames: {len(expected)}")
    print(f"Existing unique names under destination: {len(existing)}")
    print(f"Missing by filename: {len(missing)}")
    if args.skip_v:
        print(f"Skipped v-side files (--skip-v): {skipped_v}")

    if not missing:
        print("Nothing to download.")
        return 0

    if args.max_files > 0:
        missing = missing[: args.max_files]
        print(f"Capped to first {len(missing)} missing files due to --max-files")

    failed: List[Tuple[str, str]] = []
    start = time.time()
    total = len(missing)

    for i, (name, url) in enumerate(missing, start=1):
        target = dest / name
        if target.exists():
            print(f"[{i}/{total}] skip {name} (now exists)")
            continue

        print(f"[{i}/{total}] download {name}")
        ok, err = download_file(url, target, timeout=args.timeout, retries=args.retries)
        if ok:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"          ok {size_mb:.1f} MB")
            if args.sleep > 0:
                time.sleep(args.sleep)
        else:
            print(f"          fail {err}")
            failed.append((name, err))

    elapsed = time.time() - start
    print(f"Done in {elapsed / 60:.1f} min")
    print(f"Failed: {len(failed)}")

    if failed:
        fail_path = Path.cwd() / f"early_missing_failures_{int(time.time())}.csv"
        with fail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tif_filename", "error"])
            writer.writerows(failed)
        print(f"Failure list: {fail_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
