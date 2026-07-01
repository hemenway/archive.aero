#!/usr/bin/env python3
"""
Helpers for normalizing and merging Dole CSV variants.

This module is intentionally free of GDAL/runtime dependencies so that CSV
behavior can be validated independently from raster processing.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CANONICAL_FIELDS = [
    "download_link",
    "tif_filename",
    "filename",
    "location",
    "date",
    "end_date",
    "edition",
]

METADATA_FIELDS = [
    "source_dataset",
    "source_row_num",
    "location_norm",
    "date_quality",
    "candidate_group",
    "candidate_rank",
    "gcp_xy_complete",
    "row_status",
]

ABBREVIATIONS = {
    "hawaiian_is": "hawaiian_islands",
    "dallas_ft_worth": "dallas_ft_worth",
    "mariana_islands_inset": "mariana_islands",
}


def _pick(row: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return ""


def _parse_iso_date(value: str) -> Optional[dt.date]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _normalize_iso_date(value: str) -> str:
    parsed = _parse_iso_date(value)
    return parsed.isoformat() if parsed else ""


def normalize_location(name: str) -> str:
    norm = (
        str(name or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(",", "")
    )
    while "__" in norm:
        norm = norm.replace("__", "_")
    return ABBREVIATIONS.get(norm, norm)


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _is_truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _has_complete_numeric_gcp_xy(row: Dict[str, str]) -> bool:
    keys = ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y")
    for key in keys:
        value = _pick(row, (key, key.upper()))
        if not value:
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def compute_early_end_dates(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], str]:
    """
    Compute early-dataset end dates per normalized location:
    next start date when available, else start + 56 days.
    """
    by_location: Dict[str, set] = {}
    for row in rows:
        location = _pick(row, ("location", "Location"))
        location_norm = normalize_location(location)
        date_iso = _normalize_iso_date(_pick(row, ("date", "Date", "start_date", "Start Date")))
        if not (location_norm and date_iso):
            continue
        by_location.setdefault(location_norm, set()).add(date_iso)

    output: Dict[Tuple[str, str], str] = {}
    for location_norm, iso_dates in by_location.items():
        ordered = sorted(dt.date.fromisoformat(d) for d in iso_dates)
        for idx, start in enumerate(ordered):
            if idx + 1 < len(ordered):
                end = ordered[idx + 1]
            else:
                end = start + dt.timedelta(days=56)
            output[(location_norm, start.isoformat())] = end.isoformat()
    return output


def build_candidate_group(row: Dict[str, str], source_row_num: int) -> str:
    date_iso = (row.get("date") or "").strip()
    location_norm = (row.get("location_norm") or normalize_location(row.get("location", ""))).strip()
    if date_iso and location_norm:
        return f"{date_iso}|{location_norm}"

    filename = (row.get("filename") or row.get("tif_filename") or "").strip()
    if filename:
        token = _normalize_token(Path(filename).stem)
    else:
        token = f"row{int(source_row_num)}"
    return f"undated|{token}"


def rank_candidate(row: Dict[str, str]) -> int:
    if _is_truthy(row.get("gcp_xy_complete", "")):
        return 1
    if str(row.get("source_dataset") or "").strip().lower() == "master":
        return 2
    return 3


def normalize_row(
    raw_row: Dict[str, str],
    source_dataset: str,
    source_row_num: int,
    early_end_dates: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, str]:
    row = dict(raw_row)
    dataset = str(source_dataset or "").strip().lower() or "master"
    status_parts: List[str] = []

    download_link = _pick(row, ("download_link", "Download Link", "Download link"))
    tif_filename = _pick(row, ("tif_filename", "filename", "TIF Filename", "Filename"))
    filename = _pick(row, ("filename", "tif_filename", "Filename", "TIF Filename"))
    location = _pick(row, ("location", "Location", "geo_where"))
    edition = _pick(row, ("edition", "Edition", "Web Edition Number")) or "Unknown"
    location_norm = normalize_location(location)

    if not filename and tif_filename:
        filename = tif_filename
    if not tif_filename and filename:
        tif_filename = filename
    if not filename:
        status_parts.append("missing_filename")

    raw_date = _pick(row, ("date", "Date", "start_date", "Start Date"))
    date_iso = _normalize_iso_date(raw_date)
    if date_iso:
        date_quality = "valid_iso"
    else:
        date_quality = "invalid_or_blank"
        if raw_date:
            status_parts.append(f"invalid_date:{raw_date}")
        else:
            status_parts.append("missing_date")

    raw_end_date = _pick(row, ("end_date", "End Date"))
    if dataset == "early":
        if date_iso and location_norm and early_end_dates is not None:
            end_date = early_end_dates.get((location_norm, date_iso), "")
        else:
            end_date = ""
    else:
        end_date = _normalize_iso_date(raw_end_date)
        if not end_date and date_iso:
            end_date = date_iso
        if raw_end_date and not end_date:
            status_parts.append(f"invalid_end_date:{raw_end_date}")

    gcp_xy_complete = _has_complete_numeric_gcp_xy(row)

    row["download_link"] = download_link
    row["tif_filename"] = tif_filename
    row["filename"] = filename
    row["location"] = location
    row["date"] = date_iso
    row["end_date"] = end_date
    row["edition"] = edition
    row["source_dataset"] = dataset
    row["source_row_num"] = str(int(source_row_num))
    row["location_norm"] = location_norm
    row["date_quality"] = date_quality
    row["gcp_xy_complete"] = "1" if gcp_xy_complete else "0"
    row["candidate_group"] = build_candidate_group(row, source_row_num)
    row["candidate_rank"] = str(rank_candidate(row))
    row["row_status"] = ";".join(status_parts) if status_parts else "ok"
    return row


def load_combined_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load and normalize rows from combined or legacy CSV formats.
    """
    path = Path(csv_path)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    def infer_source(row: Dict[str, str]) -> str:
        source = str(row.get("source_dataset") or "").strip().lower()
        if source in {"early", "master"}:
            return source
        if "Date" in fieldnames and "Location" in fieldnames and "end_date" not in fieldnames:
            return "early"
        if path.name.lower().startswith("early_"):
            return "early"
        return "master"

    source_by_row: List[str] = [infer_source(r) for r in raw_rows]
    early_rows = [row for row, source in zip(raw_rows, source_by_row) if source == "early"]
    early_end_dates = compute_early_end_dates(early_rows)

    normalized_rows: List[Dict[str, str]] = []
    for idx, (row, source) in enumerate(zip(raw_rows, source_by_row), start=2):
        normalized_rows.append(
            normalize_row(
                row,
                source_dataset=source,
                source_row_num=idx,
                early_end_dates=early_end_dates,
            )
        )
    return normalized_rows
