#!/usr/bin/env python3
"""Regenerate dates.csv from `rclone lsf -R` output for the R2 charts bucket."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date


DATE_KEY_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:_to_\d{4}-\d{2}-\d{2})?$")


@dataclass(frozen=True)
class Row:
    key: str
    url: str
    path: str
    start: date
    end: date


def parse_key(key: str) -> tuple[date, date]:
    if "_to_" in key:
        start_s, end_s = key.split("_to_", 1)
        return date.fromisoformat(start_s), date.fromisoformat(end_s)
    d = date.fromisoformat(key)
    return d, d


def read_rclone_lsf(remote: str) -> list[str]:
    cmd = ["rclone", "lsf", "-R", remote]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("error: `rclone` not found in PATH", file=sys.stderr)
        raise SystemExit(1)

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        print(f"error: rclone failed ({proc.returncode})", file=sys.stderr)
        if stderr:
            print(stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    return proc.stdout.splitlines()


def choose_better(existing: Row, candidate: Row) -> Row:
    # Prefer the shorter object path if duplicate keys exist in multiple prefixes.
    if len(candidate.path) < len(existing.path):
        return candidate
    if len(candidate.path) > len(existing.path):
        return existing
    return min(existing, candidate, key=lambda r: r.path)


def build_rows(lines: list[str], base_url: str, sectionals_prefix: str) -> list[Row]:
    base = base_url.rstrip("/")
    prefix = sectionals_prefix.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"

    rows_by_key: dict[str, Row] = {}

    for raw in lines:
        path = raw.strip()
        if not path or path.endswith("/"):
            continue
        if not path.startswith(prefix):
            continue
        if not path.lower().endswith(".pmtiles"):
            continue

        filename = path.rsplit("/", 1)[-1]
        key = filename[: -len(".pmtiles")]
        if not DATE_KEY_RE.fullmatch(key):
            continue

        try:
            start, end = parse_key(key)
        except ValueError:
            continue

        row = Row(
            key=key,
            url=f"{base}/{path.lstrip('/')}",
            path=path,
            start=start,
            end=end,
        )
        if key in rows_by_key:
            rows_by_key[key] = choose_better(rows_by_key[key], row)
        else:
            rows_by_key[key] = row

    return sorted(
        rows_by_key.values(),
        key=lambda r: (r.start, r.end, r.key),
        reverse=True,
    )


def write_csv(rows: list[Row], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date_iso", "url"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"date_iso": row.key, "url": row.url})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update dates.csv from `rclone lsf -R` output for R2."
    )
    parser.add_argument(
        "--remote",
        default="r2:charts",
        help="rclone remote/bucket to list (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="dates.csv",
        help="CSV file to write (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default="https://data.archive.aero",
        help="Public base URL prepended to object paths (default: %(default)s)",
    )
    parser.add_argument(
        "--sectionals-prefix",
        default="sectionals",
        help="Only include PMTiles under this bucket prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read `rclone lsf -R` output from stdin instead of running rclone",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.stdin:
        lines = sys.stdin.read().splitlines()
    else:
        # Allow piping from rclone without needing --stdin.
        if not sys.stdin.isatty():
            piped = sys.stdin.read()
            lines = piped.splitlines() if piped.strip() else read_rclone_lsf(args.remote)
        else:
            lines = read_rclone_lsf(args.remote)

    rows = build_rows(lines, args.base_url, args.sectionals_prefix)
    if not rows:
        print("error: no sectionals .pmtiles entries found in rclone listing", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Would write {len(rows)} rows to {args.output}")
        print(f"Newest: {rows[0].key} -> {rows[0].url}")
        print(f"Oldest: {rows[-1].key} -> {rows[-1].url}")
        return 0

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Newest: {rows[0].key}")
    print(f"Oldest: {rows[-1].key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
