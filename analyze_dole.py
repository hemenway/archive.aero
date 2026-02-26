#!/usr/bin/env python3
"""
Compatibility helpers for timeline/CSV tools.

This reintroduces the small subset of the old analyze_dole module used by
timeline_preview_gui.py:
  - analyze_csv
  - build_file_index
  - resolve_tif_file
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_date(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def analyze_csv(csv_path: str) -> Tuple[List[str], Dict[str, List[dict]]]:
    """
    Parse master_dole-style CSV into a location->records mapping.

    Records include parsed datetime values in `date` and `end_date` and preserve
    the original CSV fields. Invalid rows are reported in `errors`; rows with an
    invalid/missing location are skipped.
    """

    errors: List[str] = []
    locations_data: Dict[str, List[dict]] = defaultdict(list)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_end_date_column = any(col in fieldnames for col in ("end_date", "End Date"))
        is_early_master_format = not has_end_date_column and "Date" in fieldnames and "Location" in fieldnames
        required_aliases = {
            "tif_filename": ("tif_filename", "filename"),
            "date": ("date", "Date"),
            "location": ("location", "Location"),
        }
        missing = [
            canonical
            for canonical, aliases in required_aliases.items()
            if not any(alias in fieldnames for alias in aliases)
        ]
        if missing:
            errors.append(f"Missing expected CSV column(s): {', '.join(missing)}")

        for line_num, row in enumerate(reader, start=2):
            rec = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            location = (rec.get("location") or rec.get("Location") or "").strip()
            if not location:
                errors.append(f"Line {line_num}: missing location")
                continue

            raw_date = rec.get("date") or rec.get("Date") or ""
            raw_end_date = rec.get("end_date") or rec.get("End Date") or ""
            start_dt = _parse_date(raw_date)
            end_dt = _parse_date(raw_end_date)

            if raw_date and not start_dt:
                errors.append(f"Line {line_num}: invalid date '{raw_date}'")
            if raw_end_date and not end_dt:
                errors.append(f"Line {line_num}: invalid end_date '{raw_end_date}'")
            if start_dt and end_dt and end_dt < start_dt:
                errors.append(
                    f"Line {line_num}: end_date {end_dt.date().isoformat()} before date {start_dt.date().isoformat()}"
                )

            rec["date"] = start_dt
            rec["end_date"] = end_dt
            rec["line"] = line_num
            rec["location"] = location
            if is_early_master_format:
                rec["edition"] = (rec.get("edition") or "").strip() or "Unknown"
            else:
                rec["edition"] = ((rec.get("edition") or rec.get("Edition") or "").strip() or "Unknown")
            rec["tif_filename"] = (rec.get("tif_filename") or rec.get("filename") or "").strip()

            locations_data[location].append(rec)

    for records in locations_data.values():
        records.sort(key=lambda r: (r.get("date") is None, r.get("date") or datetime.max))
        for i, rec in enumerate(records):
            if rec.get("date") and rec.get("end_date"):
                continue
            if rec.get("date") and is_early_master_format:
                rec["end_date"] = rec["date"] + timedelta(days=56)
            elif rec.get("date") and i + 1 < len(records) and records[i + 1].get("date"):
                rec["end_date"] = records[i + 1]["date"]
            elif rec.get("date"):
                rec["end_date"] = rec["date"] + timedelta(days=1)

    return errors, dict(locations_data)


def _norm(value: str) -> str:
    return (value or "").strip().replace("\\", "/").lower()


def _stem(value: str) -> str:
    return os.path.splitext(value)[0]


def _add_index(index: Dict[str, List[str]], key: str, path: str) -> None:
    key = _norm(key)
    if not key:
        return
    bucket = index.setdefault(key, [])
    if not bucket or bucket[-1] != path:
        if path not in bucket:
            bucket.append(path)


def _wayback_tail_keys(name: str) -> List[str]:
    name_l = _norm(name)
    keys: List[str] = []
    markers = (
        "_content_aeronav_sectional_files_",
        "_sectional_files_",
        "/sectional_files/",
    )
    for marker in markers:
        if marker in name_l:
            tail = name_l.split(marker, 1)[1]
            if tail:
                keys.append(tail)
                keys.append(_stem(tail))
    return keys


def _keys_for_filename(name: str) -> List[str]:
    name_l = _norm(name)
    if not name_l:
        return []

    base = os.path.basename(name_l)
    rel_no_ext = _stem(name_l)
    base_no_ext = _stem(base)

    keys = [
        name_l,
        rel_no_ext,
        base,
        base_no_ext,
    ]
    keys.extend(_wayback_tail_keys(name_l))
    if base != name_l:
        keys.extend(_wayback_tail_keys(base))

    seen = set()
    out = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def build_file_index(root_dir: str) -> Dict[str, List[str]]:
    """
    Recursively index files under `root_dir` by several lookup keys.

    The index maps normalized keys to a list of candidate file paths.
    """

    index: Dict[str, List[str]] = {}
    if not root_dir or not os.path.isdir(root_dir):
        return index

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir)
            rel_l = _norm(rel_path)
            base_l = _norm(filename)
            stem_l = _stem(base_l)

            for key in (rel_l, _stem(rel_l), base_l, stem_l):
                _add_index(index, key, full_path)

            for extra_key in _wayback_tail_keys(rel_l):
                _add_index(index, extra_key, full_path)

            # Add immediate parent directory keys so zip stem -> extracted tif can match.
            parent = os.path.basename(os.path.dirname(rel_l))
            if parent:
                _add_index(index, parent, full_path)
                _add_index(index, _stem(parent), full_path)

    return index


def _iter_candidates(file_index: Dict[str, List[str]], keys: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for key in keys:
        for path in file_index.get(_norm(key), []):
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _ext_rank(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tif", ".tiff"}:
        return 0
    if ext == ".pdf":
        return 1
    if ext in {".jpg", ".jpeg", ".png"}:
        return 2
    if ext == ".zip":
        return 3
    return 4


def resolve_tif_file(tif_filename: str, file_index: Dict[str, List[str]]) -> Optional[str]:
    """
    Resolve a CSV `tif_filename` to a local file path using the file index.
    """

    if not tif_filename or not file_index:
        return None

    req = _norm(tif_filename)
    req_base = os.path.basename(req)
    req_stem = _stem(req_base)

    keys = _keys_for_filename(tif_filename)
    candidates = _iter_candidates(file_index, keys)

    if not candidates and req_stem:
        # Last-resort fuzzy scan across indexed paths.
        seen = set()
        for paths in file_index.values():
            for path in paths:
                pnorm = _norm(path)
                if req_stem in pnorm and path not in seen:
                    seen.add(path)
                    candidates.append(path)

    if not candidates:
        return None

    def rank(path: str):
        pnorm = _norm(path)
        pbase = os.path.basename(pnorm)
        pstem = _stem(pbase)
        exact_base = 0 if pbase == req_base else 1
        exact_stem = 0 if pstem == req_stem else 1
        contains_stem = 0 if req_stem and req_stem in pnorm else 1
        return (exact_base, exact_stem, contains_stem, _ext_rank(path), len(path))

    return min(candidates, key=rank)
