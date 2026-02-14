#!/usr/bin/env python3
"""
OCR Effective Dates - Reads effective from/to dates from FAA sectional chart images.
Uses the same file search logic as slicer.py with master_dole.csv.

Crops the left side of each chart image and uses Tesseract OCR to extract
the "EFFECTIVE <date> TO <date>" text.

Usage:
    python3 ocr_dates.py                          # Process all records
    python3 ocr_dates.py --location Albuquerque    # Filter by location
    python3 ocr_dates.py --before-year 2021        # Only process dates before 2021
    python3 ocr_dates.py --missing-only            # Only process Unknown editions
    python3 ocr_dates.py --output results.csv      # Save results to CSV
    python3 ocr_dates.py --update                  # Update master_dole.csv with OCR dates
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
except ImportError:
    print("ERROR: pytesseract and Pillow are required.")
    print("Install with: pip3 install pytesseract Pillow --break-system-packages")
    sys.exit(1)

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    gdal = None

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_FILE = SCRIPT_DIR / 'master_dole.csv'
SOURCE_DIR = Path('/Volumes/drive/newrawtiffs')

# Month name -> number mapping for date parsing
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}

# Date pattern: DD MON YYYY (e.g., "12 OCT 2017")
DATE_PATTERN = re.compile(
    r'\b(\d{1,2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})\b',
    re.IGNORECASE
)

# Edition pattern: e.g., "100TH EDITION" or just a number before EDITION
EDITION_PATTERN = re.compile(r'(\d+)\s*(?:TH|ST|ND|RD)?\s*EDITION', re.IGNORECASE)

# OCR settings tuned for date-panel text
OCR_TARGET_WIDTH_MIN = 2200
OCR_TARGET_WIDTH_MAX = 3200
OCR_CONFIGS = (
    '--oem 1 --psm 6 -c preserve_interword_spaces=1',
    '--oem 1 --psm 4 -c preserve_interword_spaces=1',
    '--oem 1 --psm 11 -c preserve_interword_spaces=1',
    '--oem 1 --psm 12 -c preserve_interword_spaces=1',
)
DEFAULT_OCR_TIMEOUT_SEC = 12
OCR_TIMEOUT_SEC = DEFAULT_OCR_TIMEOUT_SEC
OCR_RECORD_TIMEOUT_SEC = None
OCR_ATTEMPT_TIMEOUT_CAP_SEC = 25.0
HIGH_CONFIDENCE_PARSE_SCORE = 245
LEFT_THIRD_FRACTION = 1.0 / 3.0


def load_csv(csv_file):
    """Load CSV records into memory."""
    records = []
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def _variant_score(path):
    """
    Score TIFF filename preference.
    Positive: likely north variant. Negative: likely south variant.
    """
    name = path.name.lower()
    score = 0
    if re.search(r'(^|[_\-.])north([_\-.]|$)', name):
        score += 10
    if re.search(r'(^|[_\-.])n([_\-.]|$)', name):
        score += 4
    if re.search(r'(^|[_\-.])south([_\-.]|$)', name):
        score -= 10
    if re.search(r'(^|[_\-.])s([_\-.]|$)', name):
        score -= 4
    return score


def _prefer_north_variants(paths):
    """Return paths sorted with north variants first."""
    return sorted(paths, key=lambda p: (-_variant_score(p), p.name.lower(), str(p).lower()))


def _variant_key(path):
    """Normalize filename stem by removing north/south tokens."""
    stem = path.stem.lower()
    stem = re.sub(r'(^|[_\-.])(north|south|n|s)(?=([_\-.]|$))', r'\1', stem)
    stem = re.sub(r'[_\-.]+', '_', stem).strip('_')
    return stem


def _collect_tif_candidates(directory):
    """Collect .tif/.tiff files from a directory, preferring north variants."""
    tifs = list(directory.glob('**/*.tif')) + list(directory.glob('**/*.tiff'))
    return _prefer_north_variants(tifs)


def _filename_has_ns_token(path):
    """True if file name appears to encode north/south variants."""
    return bool(re.search(r'(^|[_\-.])(north|south|n|s)([_\-.]|$)', path.stem.lower()))


def _north_sibling_if_available(path):
    """When path looks like a north/south variant, try selecting north sibling."""
    if not _filename_has_ns_token(path):
        return path
    sibling_candidates = list(path.parent.glob('*.tif')) + list(path.parent.glob('*.tiff'))
    if not sibling_candidates:
        return path
    base_key = _variant_key(path)
    family = [p for p in sibling_candidates if _variant_key(p) == base_key]
    if not family:
        return path
    ranked = _prefer_north_variants(family)
    return ranked[0] if ranked else path


def resolve_source_file(record, source_dir=None):
    """
    Find the actual source file on disk for a record.
    Mirrors the file search logic from slicer.py's resolve_filename.
    Returns (path, filetype) or (None, None).
    """
    if source_dir is None:
        source_dir = SOURCE_DIR

    tif_filename = record.get('tif_filename', '')
    download_link = record.get('download_link', '')

    if not tif_filename:
        return None, None

    from urllib.parse import urlparse
    url_ext = ''
    if download_link:
        parsed = urlparse(download_link)
        url_ext = Path(parsed.path).suffix.lower()

    # Case 1: ZIP files - check for extracted directory first (like slicer.py)
    if tif_filename.endswith('.zip') or url_ext == '.zip':
        zip_stem = tif_filename
        if zip_stem.endswith('.zip'):
            zip_stem = zip_stem[:-4]

        extract_dir = source_dir / zip_stem
        if extract_dir.is_dir():
            tifs = _collect_tif_candidates(extract_dir)
            if tifs:
                return tifs[0], '.tif'

        # Try stem of filename as directory name
        dir_name = Path(tif_filename).stem
        extract_dir2 = source_dir / dir_name
        if extract_dir2.is_dir() and extract_dir2 != extract_dir:
            tifs = _collect_tif_candidates(extract_dir2)
            if tifs:
                return tifs[0], '.tif'

    # Case 2: Direct .tif file match
    direct = source_dir / tif_filename
    if direct.exists() and direct.is_file() and direct.suffix.lower() in ('.tif', '.tiff'):
        preferred = _north_sibling_if_available(direct)
        return preferred, preferred.suffix.lower()

    # Case 3: Try with the download URL's actual extension
    if url_ext and url_ext != direct.suffix.lower():
        alt = direct.with_suffix(url_ext)
        if alt.exists() and alt.is_file():
            return alt, url_ext

    # Case 4: For PDFs
    if url_ext == '.pdf':
        pdf_name = tif_filename
        if pdf_name.endswith('.tif'):
            pdf_name = pdf_name[:-4] + '.pdf'
        pdf_path = source_dir / pdf_name
        if pdf_path.exists():
            return pdf_path, '.pdf'

    # Case 5: Check if tif_filename is a directory name (extracted zip without .zip)
    dir_path = source_dir / tif_filename
    if dir_path.is_dir():
        tifs = _collect_tif_candidates(dir_path)
        if tifs:
            return tifs[0], '.tif'

    # Case 6: Strip extensions and look for directory
    for ext in ['.zip', '.tif']:
        if tif_filename.endswith(ext):
            stem = tif_filename[:-len(ext)]
            dir_path = source_dir / stem
            if dir_path.is_dir():
                tifs = _collect_tif_candidates(dir_path)
                if tifs:
                    return tifs[0], '.tif'

    # Case 7: Direct file exists (any type, e.g., ZIP that wasn't extracted)
    if direct.exists() and direct.is_file():
        return direct, direct.suffix.lower()

    return None, None


def _candidate_src_windows(width, height):
    """Return candidate (x, y, w, h) windows likely to include effective dates."""
    if height <= 0 or width <= 0:
        return []

    aspect = width / height
    x_right = LEFT_THIRD_FRACTION
    if aspect > 2.0:
        # Very wide charts (north/south halves) usually place dates lower on the left.
        y_bands = [
            (0.56, 0.96),
            (0.48, 0.88),
            (0.40, 0.80),
            (0.32, 0.72),
            (0.24, 0.64),
            (0.16, 0.56),
        ]
    elif aspect > 1.2:
        y_bands = [
            (0.50, 0.92),
            (0.42, 0.84),
            (0.34, 0.76),
            (0.26, 0.68),
            (0.18, 0.60),
            (0.10, 0.52),
        ]
    else:
        y_bands = [
            (0.58, 0.98),
            (0.50, 0.90),
            (0.42, 0.82),
            (0.34, 0.74),
            (0.26, 0.66),
            (0.18, 0.58),
            (0.10, 0.50),
        ]

    fractions = []
    for y0f, y1f in y_bands:
        # Full left-third coverage.
        fractions.append((0.00, y0f, x_right, y1f))
        # Narrower strip inside the left third to reduce unrelated text.
        fractions.append((0.00, max(0.0, y0f - 0.02), min(0.26, x_right), min(0.995, y1f + 0.02)))
        # Slightly inset variant catches panels offset from the left edge.
        fractions.append((0.02, y0f, min(0.30, x_right), y1f))

    # Coarse fallbacks still restricted to left third.
    fractions.extend([
        (0.00, 0.08, x_right, 0.55),
        (0.00, 0.28, x_right, 0.78),
        (0.00, 0.48, x_right, 0.98),
    ])

    windows = []
    seen = set()
    for x0f, y0f, x1f, y1f in fractions:
        x0 = max(0, min(width - 1, int(width * x0f)))
        y0 = max(0, min(height - 1, int(height * y0f)))
        x1 = min(width, max(x0 + 96, int(width * x1f)))
        y1 = min(height, max(y0 + 96, int(height * y1f)))
        if x1 > x0 and y1 > y0:
            win = (x0, y0, x1 - x0, y1 - y0)
            if win not in seen:
                seen.add(win)
                windows.append(win)
    return windows


def _normalize_crop_for_ocr(img):
    """Normalize one crop before OCR."""
    if img.mode != 'L':
        img = img.convert('L')
    if img.width > OCR_TARGET_WIDTH_MAX:
        scale = OCR_TARGET_WIDTH_MAX / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


def crop_date_regions(image_path, filetype='.tif'):
    """
    Crop candidate left-side date regions from a chart image.
    Returns a list of PIL Images ordered from most likely to fallback.
    """
    try:
        if filetype in ('.tif', '.tiff') and gdal:
            return _crop_with_gdal(image_path)
        if filetype == '.pdf':
            return _crop_pdf(image_path)
        # Fallback: try PIL directly
        return _crop_with_pil(image_path)
    except Exception as e:
        print(f"    Error cropping {image_path}: {e}")
        return []


def crop_date_region(image_path, filetype='.tif'):
    """Backward-compatible helper that returns only the first candidate crop."""
    regions = crop_date_regions(image_path, filetype=filetype)
    return regions[0] if regions else None


def _crop_with_gdal(image_path):
    """Use GDAL to crop candidate date regions from a GeoTIFF."""
    ds = gdal.Open(str(image_path))
    if not ds:
        return []

    w, h = ds.RasterXSize, ds.RasterYSize
    band1 = ds.GetRasterBand(1) if ds.RasterCount >= 1 else None
    has_palette = bool(band1 and band1.GetRasterColorTable() is not None)
    windows = _candidate_src_windows(w, h)
    images = []

    for x0, y0, crop_w, crop_h in windows:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Keep extraction reasonably sized; OCR can scale up if needed.
            if crop_w > OCR_TARGET_WIDTH_MAX:
                out_w = OCR_TARGET_WIDTH_MAX
                out_h = max(1, int(crop_h * (OCR_TARGET_WIDTH_MAX / crop_w)))
            else:
                out_w = crop_w
                out_h = crop_h

            translate_kwargs = {
                'format': 'JPEG',
                'srcWin': [x0, y0, crop_w, crop_h],
                'width': out_w,
                'height': out_h,
                'creationOptions': ['QUALITY=95'],
            }
            if has_palette:
                # Expand indexed color tables so OCR sees clean text instead of noisy palette values.
                translate_kwargs['rgbExpand'] = 'rgb'

            opts = gdal.TranslateOptions(**translate_kwargs)
            gdal.Translate(tmp_path, ds, options=opts)

            img = Image.open(tmp_path)
            img.load()
            images.append(_normalize_crop_for_ocr(img))
        except Exception:
            continue
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    ds = None
    return images


def _crop_with_pil(image_path):
    """Use PIL to crop candidate date regions."""
    base = Image.open(str(image_path))
    w, h = base.size
    windows = _candidate_src_windows(w, h)
    images = []

    for x0, y0, crop_w, crop_h in windows:
        crop = base.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        images.append(_normalize_crop_for_ocr(crop))

    base.close()
    return images


def _crop_pdf(image_path):
    """Convert PDF to image and return candidate date region crops."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Use sips on macOS to convert PDF to JPEG
        subprocess.run(
            ['sips', '-s', 'format', 'jpeg', '-Z', '4096',
             str(image_path), '--out', tmp_path],
            capture_output=True, timeout=60
        )
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            return _crop_with_pil(tmp_path)
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return []


def _normalize_ocr_text_for_dates(text):
    """Normalize common OCR month confusions before date parsing."""
    if not text:
        return ''
    normalized = text.upper()
    replacements = {
        '0CT': 'OCT',
        'M4Y': 'MAY',
    }
    for wrong, right in replacements.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def _line_has_primary_time_marker(line_upper):
    """Detect the common 0901Z time marker with OCR-tolerant variants."""
    return bool(re.search(r'\b0901[Z2I]?\b', line_upper))


def _is_amendment_context(line_upper):
    """True when line refers to amendments/data-received metadata, not edition dates."""
    blocked_terms = (
        'AMEND',
        'DATA RECEIVED',
        'INCLUDES AIRSPACE',
    )
    return any(term in line_upper for term in blocked_terms)


def _effective_line_score(line_upper):
    """Score likely primary EFFECTIVE date lines higher than auxiliary lines."""
    score = 0
    if 'EFFECTIVE' in line_upper:
        score += 1
    if 'EDITION' in line_upper:
        score += 4
    if _line_has_primary_time_marker(line_upper):
        score += 3
    if re.search(r'\bTO\b', line_upper):
        score += 1
    if _is_amendment_context(line_upper):
        score -= 6
    return score


def parse_dates_from_text(text):
    """
    Parse effective from/to dates from OCR text.
    Returns (effective_date, to_date, edition) as strings, or (None, None, None).

    Expected format:
        NNNth EDITION EFFECTIVE 0901Z DD MON YYYY
                            TO 0901Z DD MON YYYY
    """
    if not text:
        return None, None, None

    normalized = _normalize_ocr_text_for_dates(text)
    dates = DATE_PATTERN.findall(normalized)
    edition_match = EDITION_PATTERN.search(normalized)

    effective_date = None
    to_date = None
    edition = None

    if edition_match:
        edition = edition_match.group(1)

    # Look for EFFECTIVE ... date and TO ... date in the text
    lines = normalized.split('\n')

    effective_candidates = []
    to_candidates = []

    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        line_dates = DATE_PATTERN.findall(line)
        line_score = _effective_line_score(line_upper)

        if 'EFFECTIVE' in line_upper and line_dates:
            effective_candidates.append((line_score, i, line_dates[0]))
            to_inline = re.search(r'\bTO\b', line_upper)
            if to_inline:
                to_inline_dates = DATE_PATTERN.findall(line_upper[to_inline.start():])
                if to_inline_dates:
                    to_candidates.append((line_score + 1, i, to_inline_dates[0]))
        elif re.match(r'^\W*TO\b', line_upper) and line_dates:
            to_score = 2
            if _line_has_primary_time_marker(line_upper):
                to_score += 2
            if _is_amendment_context(line_upper):
                to_score -= 4
            to_candidates.append((to_score, i, line_dates[0]))

    # If we found dates on EFFECTIVE/TO lines, use those
    effective_idx = None
    if effective_candidates:
        # Highest score first, then earliest line.
        effective_candidates.sort(key=lambda c: (-c[0], c[1]))
        top_score, effective_idx, (d, m, y) = effective_candidates[0]
        if top_score >= 1:
            effective_date = format_date(d, m, y)
        else:
            effective_idx = None

    if to_candidates:
        if effective_idx is not None:
            # Prefer TO date nearest at/after chosen EFFECTIVE line.
            forward = [c for c in to_candidates if c[1] >= effective_idx]
            if forward:
                forward.sort(key=lambda c: (c[1] - effective_idx, -c[0]))
                _, _, (d, m, y) = forward[0]
                to_date = format_date(d, m, y)
            else:
                to_candidates.sort(key=lambda c: (-c[0], c[1]))
                _, _, (d, m, y) = to_candidates[0]
                to_date = format_date(d, m, y)
        else:
            to_candidates.sort(key=lambda c: (-c[0], c[1]))
            _, _, (d, m, y) = to_candidates[0]
            to_date = format_date(d, m, y)

    # Fallback: search nearby text windows around EFFECTIVE/TO markers.
    if not effective_date:
        effective_match = None
        for m in re.finditer(r'\bEFFECTIVE\b', normalized):
            context = normalized[max(0, m.start() - 80):m.start() + 120]
            if _is_amendment_context(context):
                continue
            effective_match = m
            if ('EDITION' in context) or _line_has_primary_time_marker(context):
                break
        if effective_match:
            eff_chunk = normalized[effective_match.start():effective_match.start() + 200]
            eff_dates = DATE_PATTERN.findall(eff_chunk)
            if eff_dates:
                d, m, y = eff_dates[0]
                effective_date = format_date(d, m, y)
    else:
        effective_match = re.search(r'\bEFFECTIVE\b', normalized)

    if not to_date:
        to_search_start = effective_match.start() if effective_match else 0
        to_match = re.search(r'\bTO\b', normalized[to_search_start:])
        if to_match:
            absolute_to = to_search_start + to_match.start()
            to_chunk = normalized[absolute_to:absolute_to + 200]
            to_dates = DATE_PATTERN.findall(to_chunk)
            if to_dates:
                d, m, y = to_dates[0]
                to_date = format_date(d, m, y)

    # Fallback: if OCR produced exactly two dates total, use them.
    # Avoid looser fallbacks because many charts include additional "data received" dates.
    if (not effective_date) and len(dates) == 2 and re.search(r'\bTO\b', normalized):
        d, m, y = dates[0]
        effective_date = format_date(d, m, y)
    if (not to_date) and len(dates) == 2 and re.search(r'\bTO\b', normalized):
        d, m, y = dates[1]
        to_date = format_date(d, m, y)

    return effective_date, to_date, edition


def format_date(day, month_str, year):
    """Convert day/month_name/year to YYYY-MM-DD format."""
    month = MONTH_MAP.get(month_str.upper())
    if not month:
        return None
    try:
        dt = datetime(int(year), month, int(day))
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        return None


def _parse_iso_date(value):
    """Parse YYYY-MM-DD to datetime or return None."""
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError:
        return None


def _candidate_parse_score(text, effective, to_date, edition, expected_effective=None, expected_to=None):
    """
    Score OCR parse quality.
    Higher score = more likely to be the primary chart effective date block.
    """
    normalized = _normalize_ocr_text_for_dates(text or '')
    score = 0
    if effective:
        score += 120
    if to_date:
        score += 60
    if edition:
        score += 15

    if re.search(r'\bEFFECTIVE\b', normalized):
        score += 15
    if re.search(r'\bTO\b', normalized):
        score += 10
    if re.search(r'\bEDITION\b', normalized):
        score += 10
    if _line_has_primary_time_marker(normalized):
        score += 8
    if _is_amendment_context(normalized):
        score -= 40

    date_count = len(DATE_PATTERN.findall(normalized))
    if date_count == 2:
        score += 10
    elif date_count > 4:
        score -= min(20, (date_count - 4) * 3)

    eff_dt = _parse_iso_date(effective)
    to_dt = _parse_iso_date(to_date)
    expected_eff_dt = _parse_iso_date(expected_effective)
    expected_to_dt = _parse_iso_date(expected_to)

    if eff_dt and expected_eff_dt:
        delta = abs((eff_dt - expected_eff_dt).days)
        if delta == 0:
            score += 140
        elif delta <= 3:
            score += 95
        elif delta <= 14:
            score += 55
        elif delta <= 45:
            score += 25
        elif delta >= 365:
            score -= 70
        elif delta >= 180:
            score -= 30

    if to_dt and expected_to_dt:
        delta = abs((to_dt - expected_to_dt).days)
        if delta == 0:
            score += 110
        elif delta <= 3:
            score += 70
        elif delta <= 14:
            score += 40
        elif delta <= 45:
            score += 18
        elif delta >= 365:
            score -= 60
        elif delta >= 180:
            score -= 25

    if eff_dt and to_dt:
        span_days = (to_dt - eff_dt).days
        if 1 <= span_days <= 120:
            score += 30
        elif 121 <= span_days <= 200:
            score += 12
        else:
            score -= 30

    score += min(12, len(normalized) // 120)
    return score


def _ocr_variants(base_img):
    """Generate OCR preprocessing variants for higher extraction accuracy."""
    base = ImageOps.autocontrast(base_img)
    variants = [base]
    variants.append(ImageOps.equalize(base))
    variants.append(ImageOps.autocontrast(base.filter(ImageFilter.MedianFilter(size=3))))

    for threshold in (145, 165):
        binary = base.point(lambda px, t=threshold: 255 if px >= t else 0, mode='1').convert('L')
        variants.append(binary)
    return variants


def extract_year(value):
    """Extract a 4-digit year from a date-like string."""
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Fast path for ISO-like dates
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])

    m = re.search(r'\b(19|20)\d{2}\b', s)
    if m:
        return int(m.group(0))
    return None


def record_year(record):
    """
    Return the primary year for filtering:
    use 'date' when present, otherwise fall back to 'end_date'.
    """
    for field in ('date', 'end_date'):
        year = extract_year(record.get(field, ''))
        if year is not None:
            return year
    return None


def ocr_image_candidates(img, deadline=None):
    """Run OCR with multiple preprocessing variants/configs and return candidate texts."""
    if img is None:
        return []

    # Avoid very large OCR inputs: keep width in a bounded range.
    # Cropping already scales up to a useful size.
    if img.mode != 'L':
        img = img.convert('L')
    img = ImageOps.autocontrast(img)

    if img.width < OCR_TARGET_WIDTH_MIN:
        scale = OCR_TARGET_WIDTH_MIN / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    elif img.width > OCR_TARGET_WIDTH_MAX:
        scale = OCR_TARGET_WIDTH_MAX / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    texts = []
    seen = set()
    for variant in _ocr_variants(img):
        if deadline is not None and monotonic() >= deadline:
            break
        for config in OCR_CONFIGS:
            if deadline is not None and monotonic() >= deadline:
                break
            try:
                timeout = OCR_TIMEOUT_SEC
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    timeout = max(0.5, min(OCR_TIMEOUT_SEC, OCR_ATTEMPT_TIMEOUT_CAP_SEC, remaining))
                text = pytesseract.image_to_string(
                    variant,
                    config=config,
                    timeout=timeout
                )
            except RuntimeError as e:
                # Timeout on one variant/config should not discard other variants.
                if 'timed out' in str(e).lower():
                    continue
                text = ''

            if not text:
                continue

            key = re.sub(r'\s+', ' ', text.upper()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            texts.append(text)

    return texts


def process_record(record, index, total, source_dir=None):
    """Process a single record: find file, crop, OCR, extract dates."""
    tif_filename = record.get('tif_filename', '')
    location = record.get('location', '')
    csv_date = record.get('date', '')
    csv_end_date = record.get('end_date', '')
    csv_edition = record.get('edition', '')

    result = {
        'index': index,
        'tif_filename': tif_filename,
        'location': location,
        'csv_date': csv_date,
        'csv_end_date': csv_end_date,
        'csv_edition': csv_edition,
        'ocr_effective': None,
        'ocr_to': None,
        'ocr_edition': None,
        'ocr_text': '',
        'status': 'skipped',
        'match': None,
    }

    source_path, filetype = resolve_source_file(record, source_dir)
    if source_path is None:
        result['status'] = 'file_not_found'
        return result

    # Crop candidate date regions
    images = crop_date_regions(source_path, filetype)
    if not images:
        result['status'] = 'crop_failed'
        return result

    best_text = ''
    best_candidate = None
    timed_out = False
    deadline = None
    if OCR_RECORD_TIMEOUT_SEC and OCR_RECORD_TIMEOUT_SEC > 0:
        deadline = monotonic() + OCR_RECORD_TIMEOUT_SEC

    # Evaluate every crop + OCR variant and choose the highest-scoring parse.
    for img in images:
        if deadline is not None and monotonic() >= deadline:
            timed_out = True
            break
        texts = ocr_image_candidates(img, deadline=deadline)
        for text in texts:
            if deadline is not None and monotonic() >= deadline:
                timed_out = True
                break
            if text and len(text) > len(best_text):
                best_text = text

            effective, to_date, edition = parse_dates_from_text(text)
            score = _candidate_parse_score(
                text, effective, to_date, edition,
                expected_effective=csv_date,
                expected_to=csv_end_date,
            )
            candidate = {
                'score': score,
                'text': text,
                'effective': effective,
                'to_date': to_date,
                'edition': edition,
            }

            if (best_candidate is None) or (score > best_candidate['score']):
                best_candidate = candidate

            # Early exit only on very high-confidence full parse.
            if effective and to_date and score >= HIGH_CONFIDENCE_PARSE_SCORE:
                break
            if effective and csv_date and effective == csv_date:
                # If start date matches expected CSV date, stop searching early.
                if not csv_end_date or (to_date and to_date == csv_end_date):
                    break
        if timed_out:
            break
        if best_candidate and best_candidate['effective'] and best_candidate['to_date'] \
                and best_candidate['score'] >= HIGH_CONFIDENCE_PARSE_SCORE:
            break
        if best_candidate and csv_date and best_candidate['effective'] == csv_date:
            if (not csv_end_date) or (best_candidate['to_date'] and best_candidate['to_date'] == csv_end_date):
                break

    if best_candidate:
        result['ocr_text'] = best_candidate['text'].strip()
        result['ocr_effective'] = best_candidate['effective']
        result['ocr_to'] = best_candidate['to_date']
        result['ocr_edition'] = best_candidate['edition']
    else:
        result['ocr_text'] = best_text.strip()
        result['ocr_effective'] = None
        result['ocr_to'] = None
        result['ocr_edition'] = None

    effective = result['ocr_effective']
    to_date = result['ocr_to']
    if effective:
        result['status'] = 'ok'
    elif timed_out:
        result['status'] = 'ocr_timeout'
    else:
        result['status'] = 'no_dates_found'

    # Check if OCR dates match CSV dates
    if effective and csv_date:
        if effective == csv_date:
            if to_date and csv_end_date and to_date == csv_end_date:
                result['match'] = 'full_match'
            elif to_date and csv_end_date:
                result['match'] = 'start_match_end_mismatch'
            else:
                result['match'] = 'start_match'
        else:
            result['match'] = 'mismatch'

    return result


def print_record_result(pos, total, record, result, verbose=False):
    """Print one processed record and return count deltas."""
    location = record.get('location', '')
    edition = record.get('edition', '')
    csv_date = record.get('date', '')
    prefix = f"  [{pos+1}/{total}] {location} ed.{edition} ({csv_date})..."

    if result['status'] == 'ok':
        ocr_eff = result['ocr_effective']
        ocr_to = result['ocr_to']
        ocr_ed = result['ocr_edition'] or '?'

        match_icon = ''
        mismatch_delta = 0
        if result['match'] == 'full_match':
            match_icon = ' [OK]'
        elif result['match'] == 'start_match':
            match_icon = ' [start OK]'
        elif result['match'] == 'mismatch':
            match_icon = ' [MISMATCH]'
            mismatch_delta = 1
        elif result['match'] == 'start_match_end_mismatch':
            match_icon = ' [end MISMATCH]'
            mismatch_delta = 1

        print(f"{prefix} OCR: ed.{ocr_ed} {ocr_eff} to {ocr_to}{match_icon}")

        if verbose:
            for line in result['ocr_text'].split('\n')[:5]:
                if line.strip():
                    print(f"        | {line.strip()}")

        return 1, mismatch_delta, 0, 0

    if result['status'] == 'file_not_found':
        print(f"{prefix} file not found")
        return 0, 0, 1, 0

    if result['status'] == 'no_dates_found':
        print(f"{prefix} no dates in OCR")
        if verbose and result['ocr_text']:
            for line in result['ocr_text'].split('\n')[:3]:
                if line.strip():
                    print(f"        | {line.strip()}")
        return 0, 0, 0, 1

    if result['status'] == 'ocr_timeout':
        print(f"{prefix} OCR timed out")
        if verbose and result['ocr_text']:
            for line in result['ocr_text'].split('\n')[:3]:
                if line.strip():
                    print(f"        | {line.strip()}")
        return 0, 0, 0, 1

    print(f"{prefix} {result['status']}")
    return 0, 0, 0, 0


def main():
    global OCR_TIMEOUT_SEC, OCR_RECORD_TIMEOUT_SEC, OCR_ATTEMPT_TIMEOUT_CAP_SEC

    # Favor OCR quality over throughput by default.
    default_jobs = max(1, min(3, os.cpu_count() or 1))

    parser = argparse.ArgumentParser(
        description="OCR effective dates from FAA sectional chart images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-c', '--csv', type=Path, default=CSV_FILE,
        help=f"CSV file (default: {CSV_FILE})"
    )
    parser.add_argument(
        '-s', '--source', type=Path, default=SOURCE_DIR,
        help=f"Source directory for TIF/ZIP files (default: {SOURCE_DIR})"
    )
    parser.add_argument(
        '-l', '--location', type=str, default=None,
        help="Filter by location name (case-insensitive partial match)"
    )
    parser.add_argument(
        '--missing-only', action='store_true',
        help="Only process records with 'Unknown' edition"
    )
    parser.add_argument(
        '--mismatches-only', action='store_true',
        help="Only display records where OCR dates differ from CSV dates"
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help="Save results to CSV file"
    )
    parser.add_argument(
        '--update', action='store_true',
        help="Update master_dole.csv with OCR dates (for records missing dates)"
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=default_jobs,
        help=f"Number of parallel workers (default: {default_jobs})"
    )
    parser.add_argument(
        '-n', '--limit', type=int, default=None,
        help="Limit number of records to process"
    )
    parser.add_argument(
        '--before-year', type=int, default=None,
        help="Only process records with CSV year earlier than this value (e.g., 2021)"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Show OCR text output"
    )
    parser.add_argument(
        '--ocr-timeout', type=float, default=None,
        help=(
            "Tesseract timeout in seconds per OCR attempt. "
            "Default auto-scales with --jobs to reduce false timeouts."
        )
    )
    parser.add_argument(
        '--record-timeout', type=float, default=None,
        help=(
            "Hard timeout in seconds per record across all crops/variants. "
            "If omitted, no per-record timeout is applied."
        )
    )
    parser.add_argument(
        '--attempt-timeout-cap', type=float, default=OCR_ATTEMPT_TIMEOUT_CAP_SEC,
        help=(
            "Max seconds per individual OCR attempt when --record-timeout is active "
            f"(default: {OCR_ATTEMPT_TIMEOUT_CAP_SEC:.0f})."
        )
    )

    args = parser.parse_args()

    source_dir = args.source
    worker_count = max(1, args.jobs)
    if args.ocr_timeout is not None:
        OCR_TIMEOUT_SEC = max(1.0, args.ocr_timeout)
    else:
        # More workers means less CPU time per OCR process; raise timeout
        # to avoid false 'no dates' caused by scheduler contention.
        OCR_TIMEOUT_SEC = max(DEFAULT_OCR_TIMEOUT_SEC, 12.0 + (worker_count - 1) * 2.0)

    if args.record_timeout is not None:
        OCR_RECORD_TIMEOUT_SEC = max(1.0, args.record_timeout)
    else:
        OCR_RECORD_TIMEOUT_SEC = None

    OCR_ATTEMPT_TIMEOUT_CAP_SEC = max(0.5, args.attempt_timeout_cap)

    # Load CSV
    print(f"Loading {args.csv}...")
    records = load_csv(args.csv)
    print(f"  Loaded {len(records)} records")

    # Check source directory
    if not source_dir.exists():
        print(f"  WARNING: Source directory not found: {source_dir}")
        print(f"  Files will not be available.")
        sys.exit(1)

    # Apply filters
    filtered = []
    for i, record in enumerate(records):
        if args.location:
            loc = record.get('location', '')
            if args.location.lower() not in loc.lower():
                continue
        if args.missing_only:
            if record.get('edition', '') != 'Unknown':
                continue
        if args.before_year is not None:
            year = record_year(record)
            if year is None or year >= args.before_year:
                continue
        filtered.append((i, record))

    if args.limit:
        filtered = filtered[:args.limit]

    print(f"  Processing {len(filtered)} records...")
    print(f"  Workers: {worker_count}")
    print(f"  OCR timeout: {OCR_TIMEOUT_SEC:.1f}s")
    if OCR_RECORD_TIMEOUT_SEC is not None:
        print(f"  Record timeout: {OCR_RECORD_TIMEOUT_SEC:.1f}s")
        print(f"  Attempt timeout cap: {OCR_ATTEMPT_TIMEOUT_CAP_SEC:.1f}s")
    else:
        print("  Record timeout: disabled")
    if args.before_year is not None:
        print(f"  Year filter: < {args.before_year}")
    print()

    if not filtered:
        print("No records matched the current filters.")
        return

    # Process records
    results = []
    ok_count = 0
    mismatch_count = 0
    not_found_count = 0
    no_dates_count = 0

    if worker_count == 1:
        for pos, (idx, record) in enumerate(filtered):
            result = process_record(record, idx, len(filtered), source_dir)
            results.append(result)
            ok_delta, mismatch_delta, not_found_delta, no_dates_delta = print_record_result(
                pos, len(filtered), record, result, verbose=args.verbose
            )
            ok_count += ok_delta
            mismatch_count += mismatch_delta
            not_found_count += not_found_delta
            no_dates_count += no_dates_delta
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(filtered))) as executor:
            future_map = {
                executor.submit(process_record, record, idx, len(filtered), source_dir): (pos, idx, record)
                for pos, (idx, record) in enumerate(filtered)
            }
            for future in as_completed(future_map):
                pos, idx, record = future_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        'index': idx,
                        'tif_filename': record.get('tif_filename', ''),
                        'location': record.get('location', ''),
                        'csv_date': record.get('date', ''),
                        'csv_end_date': record.get('end_date', ''),
                        'csv_edition': record.get('edition', ''),
                        'ocr_effective': None,
                        'ocr_to': None,
                        'ocr_edition': None,
                        'ocr_text': '',
                        'status': f'error: {e}',
                        'match': None,
                    }

                results.append(result)
                ok_delta, mismatch_delta, not_found_delta, no_dates_delta = print_record_result(
                    pos, len(filtered), record, result, verbose=args.verbose
                )
                ok_count += ok_delta
                mismatch_count += mismatch_delta
                not_found_count += not_found_delta
                no_dates_count += no_dates_delta

    # Summary
    print()
    print("=" * 60)
    print(f"Results: {ok_count} dates extracted, {not_found_count} files missing, {no_dates_count} OCR failed")
    if mismatch_count:
        print(f"  {mismatch_count} date MISMATCHES with CSV")
    print("=" * 60)

    # Show mismatches
    mismatches = [r for r in results if r['match'] and 'mismatch' in r['match'].lower()]
    if mismatches:
        print(f"\nDate mismatches ({len(mismatches)}):")
        for r in mismatches:
            print(f"  {r['location']} ed.{r['csv_edition']}:")
            print(f"    CSV:  {r['csv_date']} to {r['csv_end_date']}")
            print(f"    OCR:  {r['ocr_effective']} to {r['ocr_to']}")

    # Save output CSV
    if args.output:
        fieldnames = [
            'location', 'csv_edition', 'csv_date', 'csv_end_date',
            'ocr_edition', 'ocr_effective', 'ocr_to', 'match', 'status', 'tif_filename'
        ]
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, '') for k in fieldnames})
        print(f"\nResults saved to {args.output}")

    # Update CSV
    if args.update:
        print(f"\nUpdating {args.csv} with OCR dates...")
        updated = 0
        for r in results:
            if r['status'] != 'ok':
                continue
            idx = r['index']
            # Only update if CSV is missing data or explicitly mismatched
            rec = records[idx]
            if r['ocr_effective'] and not rec.get('date'):
                rec['date'] = r['ocr_effective']
                updated += 1
            if r['ocr_to'] and not rec.get('end_date'):
                rec['end_date'] = r['ocr_to']
                updated += 1
            if r['ocr_edition'] and rec.get('edition') == 'Unknown':
                rec['edition'] = r['ocr_edition']
                updated += 1

        if updated:
            fieldnames = list(records[0].keys())
            with open(args.csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            print(f"  Updated {updated} fields in {args.csv}")
        else:
            print("  No updates needed")


if __name__ == '__main__':
    main()
