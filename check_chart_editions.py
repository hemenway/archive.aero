#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input": 0.40 / 1_000_000,
        "cached_input": 0.10 / 1_000_000,
        "output": 1.60 / 1_000_000,
    },
    "gpt-5": {
        "input": 1.25 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-5.1": {
        "input": 1.25 / 1_000_000,
        "cached_input": 0.125 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-5-mini": {
        "input": 0.25 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 2.00 / 1_000_000,
    },
    "gpt-5-pro": {
        "input": 15.00 / 1_000_000,
        "cached_input": 0.0,
        "output": 120.00 / 1_000_000,
    },
}


def clean(value):
    return (value or "").strip()


def parse_int(value):
    match = re.search(r"\d+", clean(value))
    return int(match.group()) if match else None


def parse_explicit_edition(value):
    text = clean(value).upper()
    if not text:
        return None

    if any(marker in text for marker in ["REVISED", "BASE:", "MAP DIVISION"]):
        return None

    if re.search(r"\b\d{1,4}\s*-\s*\d{1,4}\b", text):
        return None

    if re.fullmatch(r"\(?[A-Z]+-\d+\)?", text):
        return None

    match = re.search(r"\b(\d{1,3})(?:\s*(?:ST|ND|RD|TH))?\s+EDITION\b", text)
    if match:
        return int(match.group(1))

    match = re.search(r"\bEDITION\s+(\d{1,3})(?:\s*(?:ST|ND|RD|TH))?\b", text)
    if match:
        return int(match.group(1))

    return None


def parse_iso(value):
    value = clean(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def obj_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def usage_counts(response):
    usage = obj_get(response, "usage", None)
    input_tokens = int(obj_get(usage, "input_tokens", 0) or 0)
    output_tokens = int(obj_get(usage, "output_tokens", 0) or 0)
    input_details = obj_get(usage, "input_tokens_details", None)
    cached_tokens = int(obj_get(input_details, "cached_tokens", 0) or 0)
    return input_tokens, output_tokens, cached_tokens


def normalize_basis(value):
    return clean(value) or "none_or_unclear"


def validated_start_date(metadata):
    iso = parse_iso(metadata.get("start_date_iso"))
    if not iso:
        return ""

    basis = normalize_basis(metadata.get("start_date_basis"))
    text = clean(metadata.get("start_date_text")).upper()

    if basis not in {"chart_effective_or_cutoff", "none_or_unclear"}:
        return ""

    if any(marker in text for marker in ["BASE:", "EDITION OF ", "REVISED "]):
        return ""

    if any(marker in text for marker in ["LIBRARY OF CONGRESS", "MAP DIVISION"]):
        return ""

    return iso


def validated_end_date(metadata):
    iso = parse_iso(metadata.get("end_date_iso"))
    if not iso:
        return ""

    basis = normalize_basis(metadata.get("end_date_basis"))
    text = clean(metadata.get("end_date_text")).upper()
    evidence = clean(metadata.get("evidence_text")).upper()

    explicit_scheduled_end = "TERMINATION OF ITS SCHEDULED EFFECTIVE PERIOD" in (
        f"{text}\n{evidence}"
    )

    if basis not in {"explicit_chart_end", "none_or_unclear"} and not explicit_scheduled_end:
        return ""

    if any(
        marker in text
        for marker in [
            "NEXT SCHEDULED EDITION",
            "NEXT EDITION",
            "REVISED ",
            "BASE:",
            "EDITION OF ",
            "DATA RECEIVED THROUGH",
        ]
    ):
        return ""

    if "REVISED TO" in evidence:
        return ""

    return iso


def estimate_cost_usd(model, input_tokens, output_tokens, cached_tokens):
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        raise ValueError(f"No pricing configured for model: {model}")
    non_cached_input = max(0, input_tokens - cached_tokens)
    return (
        non_cached_input * pricing["input"]
        + cached_tokens * pricing["cached_input"]
        + output_tokens * pricing["output"]
    )


def row_start_date(row):
    return clean(row.get("date") or row.get("Date"))


def resolve_source_tifs(tiff_root, filename):
    path = tiff_root / filename
    if filename.endswith(".zip"):
        extracted_dir = tiff_root / Path(filename).stem
        if not extracted_dir.is_dir():
            raise FileNotFoundError(f"missing extracted ZIP directory: {extracted_dir}")
        tif_paths = sorted(p for p in extracted_dir.rglob("*.tif") if p.is_file())
        if not tif_paths:
            raise FileNotFoundError(f"no TIFFs found in extracted ZIP directory: {extracted_dir}")
        return tif_paths

    if not path.exists():
        raise FileNotFoundError(f"missing TIFF: {path}")
    return [path]


def find_bottom_margin_top(src_tif):
    _width, height = map(
        int,
        subprocess.check_output(
            ["magick", "identify", "-format", "%w %h", str(src_tif)],
            text=True,
        ).split(),
    )

    sample_w = 256
    sample_h = 256
    lower_fraction = 0.22
    raw = subprocess.check_output(
        [
            "magick",
            str(src_tif),
            "-gravity",
            "south",
            "-crop",
            f"100%x{int(lower_fraction * 100)}%+0+0",
            "+repage",
            "-colorspace",
            "Gray",
            "-resize",
            f"{sample_w}x{sample_h}!",
            "-depth",
            "8",
            "gray:-",
        ]
    )

    start_y = int(height * (1.0 - lower_fraction))
    dark_threshold = 48
    min_dark_ratio = 0.35
    side_margin = 8

    for y in range(sample_h):
        row = raw[y * sample_w : (y + 1) * sample_w]
        inner = row[side_margin : sample_w - side_margin]
        dark_ratio = sum(1 for px in inner if px <= dark_threshold) / len(inner)
        if dark_ratio >= min_dark_ratio:
            border_y = start_y + int((y / sample_h) * (height * lower_fraction))
            return max(0, border_y - 12)

    return int(height * 0.88)


def prepare_crop(src_tif, tmp_dir):
    width, height = map(
        int,
        subprocess.check_output(
            ["magick", "identify", "-format", "%w %h", str(src_tif)],
            text=True,
        ).split(),
    )

    out = tmp_dir / f"{src_tif.stem}_edition_crop.png"
    strip = tmp_dir / f"{src_tif.stem}_bottom_strip.png"
    full = tmp_dir / f"{src_tif.stem}_bottom_full.png"
    left = tmp_dir / f"{src_tif.stem}_bottom_left.png"
    center = tmp_dir / f"{src_tif.stem}_bottom_center.png"
    right = tmp_dir / f"{src_tif.stem}_bottom_right.png"
    margin_top = find_bottom_margin_top(src_tif)
    crop_height = max(1, height - margin_top)

    subprocess.run(
        [
            "magick",
            str(src_tif),
            "-crop",
            f"{width}x{crop_height}+0+{margin_top}",
            "+repage",
            str(strip),
        ],
        check=True,
    )

    subprocess.run(
        ["magick", str(strip), "-resize", "2200x", str(full)],
        check=True,
    )
    subprocess.run(
        [
            "magick",
            str(strip),
            "-gravity",
            "west",
            "-crop",
            "42%x100%+0+0",
            "+repage",
            "-resize",
            "2200x",
            str(left),
        ],
        check=True,
    )
    subprocess.run(
        [
            "magick",
            str(strip),
            "-gravity",
            "center",
            "-crop",
            "42%x100%+0+0",
            "+repage",
            "-resize",
            "2200x",
            str(center),
        ],
        check=True,
    )
    subprocess.run(
        [
            "magick",
            str(strip),
            "-gravity",
            "east",
            "-crop",
            "42%x100%+0+0",
            "+repage",
            "-resize",
            "2200x",
            str(right),
        ],
        check=True,
    )
    subprocess.run(
        ["magick", str(full), str(left), str(center), str(right), "-append", str(out)],
        check=True,
    )
    return out


def prepare_source_crop(source_tifs, tmp_dir, output_stem):
    if len(source_tifs) == 1:
        return prepare_crop(source_tifs[0], tmp_dir)

    crop_paths = [prepare_crop(src_tif, tmp_dir) for src_tif in source_tifs]
    out = tmp_dir / f"{output_stem}_edition_crop.png"
    subprocess.run(
        ["magick", *(str(path) for path in crop_paths), "-append", str(out)],
        check=True,
    )
    for crop_path in crop_paths:
        if crop_path.exists():
            crop_path.unlink()
    return out


def extract_metadata(client, model, crop_path, service_tier):
    b64 = base64.b64encode(crop_path.read_bytes()).decode("utf-8")
    request = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract chart edition metadata from this image. "
                            "Return only JSON. "
                            "The image may be a stitched composite of the full bottom margin plus zoomed left, center, and right panels. "
                            "Dates can appear on the left side, and edition/title information often appears on the right. "
                            "Maps may omit edition number and/or next-edition/end date. "
                            "If a field is unreadable or absent, return an empty string. "
                            "Use exact visible text for *_text fields. "
                            "Use YYYY-MM-DD for *_iso fields only when the image shows a full date. "
                            "Do not infer from sequence or neighboring charts. "
                            "Only treat an edition number as valid when the chart explicitly says something like "
                            "'25TH EDITION' or 'EDITION 25'. "
                            "Do not use chart identifiers like '(W-5)', hyphen codes like '45-12', bare numbers like '500', "
                            "or revision/publication dates as edition numbers. "
                            "For start_date, use only the current chart's operative date, such as a date in phrases like "
                            "'after MAY 29, 1953', 'includes data received through JUNE 3, 1958', "
                            "'effective September 22, 1960', or a standalone chart issue date. "
                            "Do not use 'BASE: Edition of ...', 'Revised ...', library stamps, or publication stamps as start_date. "
                            "For end_date, use only an explicit end/superseded/until date for the current chart. "
                            "Do not use revision dates or 'next scheduled edition' dates as end_date. "
                            "Classify each extracted field with a basis enum so downstream code can reject ambiguous cases."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "chart_metadata",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "edition_number_text": {"type": "string"},
                        "edition_basis": {
                            "type": "string",
                            "enum": [
                                "explicit_edition",
                                "chart_code_or_series",
                                "base_map_reference",
                                "publication_or_stamp",
                                "none_or_unclear",
                            ],
                        },
                        "start_date_text": {"type": "string"},
                        "start_date_iso": {"type": "string"},
                        "start_date_basis": {
                            "type": "string",
                            "enum": [
                                "chart_effective_or_cutoff",
                                "base_map_reference",
                                "publication_or_stamp",
                                "next_edition_schedule",
                                "none_or_unclear",
                            ],
                        },
                        "end_date_text": {"type": "string"},
                        "end_date_iso": {"type": "string"},
                        "end_date_basis": {
                            "type": "string",
                            "enum": [
                                "explicit_chart_end",
                                "base_map_reference",
                                "publication_or_stamp",
                                "next_edition_schedule",
                                "none_or_unclear",
                            ],
                        },
                        "publication_date_text": {"type": "string"},
                        "publication_date_iso": {"type": "string"},
                        "evidence_text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "edition_number_text",
                        "edition_basis",
                        "start_date_text",
                        "start_date_iso",
                        "start_date_basis",
                        "end_date_text",
                        "end_date_iso",
                        "end_date_basis",
                        "publication_date_text",
                        "publication_date_iso",
                        "evidence_text",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    }
    request["service_tier"] = service_tier
    response = client.responses.create(**request)
    return json.loads(response.output_text), response


def same_location_neighbor(rows, by_location, position_in_location, idx, direction):
    loc = clean(rows[idx].get("location"))
    indices = by_location[loc]
    pos = position_in_location[idx] + direction
    if 0 <= pos < len(indices):
        return indices[pos]
    return None


def nearest_edition_neighbor(rows, by_location, position_in_location, idx, direction):
    loc = clean(rows[idx].get("location"))
    indices = by_location[loc]
    pos = position_in_location[idx] + direction
    while 0 <= pos < len(indices):
        j = indices[pos]
        edition = parse_int(rows[j].get("edition") or rows[j].get("Edition"))
        if edition is not None:
            return j, edition, pos
        pos += direction
    return None, None, None


def infer_expected_edition(rows, by_location, position_in_location, idx):
    pos = position_in_location[idx]
    _prev_idx, prev_edition, prev_pos = nearest_edition_neighbor(
        rows, by_location, position_in_location, idx, -1
    )
    _next_idx, next_edition, next_pos = nearest_edition_neighbor(
        rows, by_location, position_in_location, idx, 1
    )

    if prev_edition is not None and next_edition is not None:
        gap = next_pos - prev_pos
        if gap > 0 and (next_edition - prev_edition) == gap:
            return prev_edition + (pos - prev_pos)

    if prev_edition is not None and prev_pos == pos - 1:
        return prev_edition + 1

    if next_edition is not None and next_pos == pos + 1:
        return next_edition - 1

    return None


def date_check(found_iso, expected_iso):
    found_iso = parse_iso(found_iso)
    expected_iso = parse_iso(expected_iso)
    if not expected_iso:
        return "n/a"
    if not found_iso:
        return "missing"
    if found_iso == expected_iso:
        return "ok"
    return "mismatch"


def edition_check(found_edition, expected_edition):
    if expected_edition is None:
        return "n/a"
    if found_edition is None:
        return "missing"
    if found_edition == expected_edition:
        return "ok"
    return "mismatch"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check chart edition/date OCR against neighboring CSV rows."
    )
    parser.add_argument(
        "--csv",
        default="/Users/ryanhemenway/archive.aero/master_dole_combined.csv",
        help="Master CSV path.",
    )
    parser.add_argument(
        "--tiff-root",
        default="/Volumes/projects/rawtiffs",
        help="Directory containing TIFFs named by the CSV filename column.",
    )
    parser.add_argument(
        "--output",
        default="/Users/ryanhemenway/archive.aero/edition_check_results.jsonl",
        help="Append-only JSONL output path.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--service-tier",
        default="flex",
        choices=["auto", "default", "flex", "priority"],
        help="Responses API service tier. Defaults to flex.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of charts to process this run.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    tiff_root = Path(args.tiff_root)
    out_path = Path(args.output)
    tmp_dir = Path(tempfile.mkdtemp(prefix="edition_check_"))

    client = OpenAI()

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    by_location = {}
    position_in_location = {}
    for idx, row in enumerate(rows):
        loc = clean(row.get("location"))
        by_location.setdefault(loc, []).append(idx)
    for loc, indices in by_location.items():
        indices.sort(key=lambda i: (parse_iso(row_start_date(rows[i])) or "9999-99-99", clean(rows[i].get("filename"))))
        for pos, idx in enumerate(indices):
            position_in_location[idx] = pos

    processed = set()
    cumulative_cost = 0.0
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                processed.add(record.get("filename"))
                cumulative_cost += float(record.get("estimated_cost_usd", 0.0) or 0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("a")
    processed_this_run = 0

    try:
        for idx, row in enumerate(rows):
            if args.limit is not None and processed_this_run >= args.limit:
                break

            filename = clean(row.get("filename"))
            if not filename or filename in processed:
                continue

            crop_path = None

            prev_idx = same_location_neighbor(
                rows, by_location, position_in_location, idx, -1
            )
            next_idx = same_location_neighbor(
                rows, by_location, position_in_location, idx, 1
            )

            csv_start = row_start_date(row)
            csv_end = clean(row.get("end_date"))
            csv_edition = clean(row.get("edition") or row.get("Edition"))
            expected_start = clean(rows[prev_idx].get("end_date")) if prev_idx is not None else ""
            expected_end = row_start_date(rows[next_idx]) if next_idx is not None else ""
            expected_edition = infer_expected_edition(
                rows, by_location, position_in_location, idx
            )

            record = {
                "row_number": idx + 2,
                "filename": filename,
                "location": clean(row.get("location")),
                "csv_start_date": csv_start,
                "csv_end_date": csv_end,
                "csv_edition": csv_edition,
                "prev_filename": clean(rows[prev_idx].get("filename")) if prev_idx is not None else "",
                "prev_end_date": expected_start,
                "next_filename": clean(rows[next_idx].get("filename")) if next_idx is not None else "",
                "next_start_date": expected_end,
                "expected_edition_from_neighbors": expected_edition if expected_edition is not None else "",
            }

            try:
                source_tifs = resolve_source_tifs(tiff_root, filename)
                crop_path = prepare_source_crop(source_tifs, tmp_dir, Path(filename).stem)
                metadata, response = extract_metadata(
                    client, args.model, crop_path, args.service_tier
                )
                input_tokens, output_tokens, cached_tokens = usage_counts(response)
                image_cost = estimate_cost_usd(
                    args.model, input_tokens, output_tokens, cached_tokens
                )
                cumulative_cost += image_cost

                raw_found_edition = parse_int(metadata["edition_number_text"])
                edition_basis = normalize_basis(metadata.get("edition_basis"))
                found_edition = (
                    parse_explicit_edition(metadata["edition_number_text"])
                    if edition_basis in {"explicit_edition", "none_or_unclear"}
                    else None
                )
                found_start_iso = validated_start_date(metadata)
                found_end_iso = validated_end_date(metadata)

                start_csv_status = date_check(found_start_iso, csv_start)
                start_prev_status = date_check(found_start_iso, expected_start)
                end_csv_status = date_check(found_end_iso, csv_end)
                end_next_status = date_check(found_end_iso, expected_end)
                edition_csv_status = edition_check(found_edition, parse_int(csv_edition))
                edition_seq_status = edition_check(found_edition, expected_edition)

                record.update(metadata)
                record.update(
                    {
                        "raw_found_edition_int": raw_found_edition if raw_found_edition is not None else "",
                        "found_edition_int": found_edition if found_edition is not None else "",
                        "validated_start_date_iso": found_start_iso,
                        "validated_end_date_iso": found_end_iso,
                        "start_match_csv_start": start_csv_status,
                        "start_match_prev_end": start_prev_status,
                        "end_match_csv_end": end_csv_status,
                        "end_match_next_start": end_next_status,
                        "edition_match_csv_edition": edition_csv_status,
                        "edition_sequence_check": edition_seq_status,
                        "usage_input_tokens": input_tokens,
                        "usage_cached_input_tokens": cached_tokens,
                        "usage_output_tokens": output_tokens,
                        "requested_service_tier": args.service_tier,
                        "actual_service_tier": obj_get(response, "service_tier", "") or "",
                        "estimated_cost_usd": round(image_cost, 6),
                        "cumulative_cost_usd": round(cumulative_cost, 6),
                    }
                )

                print(
                    f"[{idx + 1}/{len(rows)}] {filename} "
                    f"cost=${image_cost:.4f} total=${cumulative_cost:.2f} "
                    f"start_csv={start_csv_status} start_prev={start_prev_status} "
                    f"end_csv={end_csv_status} end_next={end_next_status} "
                    f"edition_csv={edition_csv_status} edition_seq={edition_seq_status}"
                )

            except Exception as exc:
                record["error"] = str(exc)
                print(f"[{idx + 1}/{len(rows)}] {filename} error: {exc}")

            out.write(json.dumps(record) + "\n")
            out.flush()
            processed.add(filename)
            processed_this_run += 1

            if crop_path and crop_path.exists():
                crop_path.unlink()

    finally:
        out.close()

    print(f"Wrote {out_path}")
    print(f"Cumulative estimated spend in this output file: ${cumulative_cost:.2f}")


if __name__ == "__main__":
    main()
