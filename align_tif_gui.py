#!/usr/bin/env python3
"""
Image Aligner TUI - Aligns images to FAA sectional chart references.

Automatically matches input images to reference files in /Volumes/projects/newrawtiffs
based on the location name extracted from the filename.

Usage:
    python align_tif_gui.py <input> [output]           # Single file mode
    python align_tif_gui.py --batch master_dole.csv    # Batch process PDFs from CSV
    python align_tif_gui.py --batch master_dole.csv --resume [log_or_state_path]  # Resume
    python align_tif_gui.py                            # Interactive mode
"""
import sys
import tempfile
import subprocess
import zipfile
import re
import csv
import urllib.request
import time
import json
import shutil
from typing import Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from osgeo import gdal
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

console = Console()

# Reference files directory
REFERENCE_DIR = Path("/Volumes/projects/newrawtiffs")

# Batch output directory
BATCH_OUTPUT_DIR = Path("/Volumes/projects/pdf->tif")

# Temp directory for downloads (configurable)
TEMP_DIR = Path("/Users/ryanhemenway/Desktop/temp")

# Known reference zip files (location name -> zip filename)
REFERENCE_ZIPS = {
    "albuquerque": "web.archive.org_web_20251025034711_aeronav.faa.gov_visual_10-02-2025_sectional-files_Albuquerque.zip",
    "anchorage": "web.archive.org_web_20251025034316_aeronav.faa.gov_visual_10-02-2025_sectional-files_Anchorage.zip",
    "atlanta": "web.archive.org_web_20251025041140_aeronav.faa.gov_visual_10-02-2025_sectional-files_Atlanta.zip",
    "bethel": "web.archive.org_web_20251025044054_aeronav.faa.gov_visual_10-02-2025_sectional-files_Bethel.zip",
    "billings": "web.archive.org_web_20251025041320_aeronav.faa.gov_visual_10-02-2025_sectional-files_Billings.zip",
    "brownsville": "web.archive.org_web_20251025044537_aeronav.faa.gov_visual_10-02-2025_sectional-files_Brownsville.zip",
    "cape_lisburne": "web.archive.org_web_20251025042104_aeronav.faa.gov_visual_10-02-2025_sectional-files_Cape_Lisburne.zip",
    "cape lisburne": "web.archive.org_web_20251025042104_aeronav.faa.gov_visual_10-02-2025_sectional-files_Cape_Lisburne.zip",
    "charlotte": "web.archive.org_web_20251025041136_aeronav.faa.gov_visual_10-02-2025_sectional-files_Charlotte.zip",
    "cheyenne": "web.archive.org_web_20251025042024_aeronav.faa.gov_visual_10-02-2025_sectional-files_Cheyenne.zip",
    "chicago": "web.archive.org_web_20251025042418_aeronav.faa.gov_visual_10-02-2025_sectional-files_Chicago.zip",
    "cincinnati": "web.archive.org_web_20251025041150_aeronav.faa.gov_visual_10-02-2025_sectional-files_Cincinnati.zip",
    "cold_bay": "web.archive.org_web_20251025035029_aeronav.faa.gov_visual_10-02-2025_sectional-files_Cold_Bay.zip",
    "cold bay": "web.archive.org_web_20251025035029_aeronav.faa.gov_visual_10-02-2025_sectional-files_Cold_Bay.zip",
    "dallas-ft_worth": "web.archive.org_web_20251025031444_aeronav.faa.gov_visual_10-02-2025_sectional-files_Dallas-Ft_Worth.zip",
    "dallas-ft worth": "web.archive.org_web_20251025031444_aeronav.faa.gov_visual_10-02-2025_sectional-files_Dallas-Ft_Worth.zip",
    "dallas": "web.archive.org_web_20251025031444_aeronav.faa.gov_visual_10-02-2025_sectional-files_Dallas-Ft_Worth.zip",
    "dawson": "web.archive.org_web_20251025031624_aeronav.faa.gov_visual_10-02-2025_sectional-files_Dawson.zip",
    "denver": "web.archive.org_web_20251025043914_aeronav.faa.gov_visual_10-02-2025_sectional-files_Denver.zip",
    "detroit": "web.archive.org_web_20251025034322_aeronav.faa.gov_visual_10-02-2025_sectional-files_Detroit.zip",
    "dutch_harbor": "web.archive.org_web_20251025040543_aeronav.faa.gov_visual_10-02-2025_sectional-files_Dutch_Harbor.zip",
    "dutch harbor": "web.archive.org_web_20251025040543_aeronav.faa.gov_visual_10-02-2025_sectional-files_Dutch_Harbor.zip",
    "el_paso": "web.archive.org_web_20251025044431_aeronav.faa.gov_visual_10-02-2025_sectional-files_El_Paso.zip",
    "el paso": "web.archive.org_web_20251025044431_aeronav.faa.gov_visual_10-02-2025_sectional-files_El_Paso.zip",
    "fairbanks": "web.archive.org_web_20251025031951_aeronav.faa.gov_visual_10-02-2025_sectional-files_Fairbanks.zip",
    "great_falls": "web.archive.org_web_20251025043152_aeronav.faa.gov_visual_10-02-2025_sectional-files_Great_Falls.zip",
    "great falls": "web.archive.org_web_20251025043152_aeronav.faa.gov_visual_10-02-2025_sectional-files_Great_Falls.zip",
    "green_bay": "web.archive.org_web_20251025041103_aeronav.faa.gov_visual_10-02-2025_sectional-files_Green_Bay.zip",
    "green bay": "web.archive.org_web_20251025041103_aeronav.faa.gov_visual_10-02-2025_sectional-files_Green_Bay.zip",
    "halifax": "web.archive.org_web_20251025042424_aeronav.faa.gov_visual_10-02-2025_sectional-files_Halifax.zip",
    "hawaiian_islands": "web.archive.org_web_20251025040847_aeronav.faa.gov_visual_10-02-2025_sectional-files_Hawaiian_Islands.zip",
    "hawaiian islands": "web.archive.org_web_20251025040847_aeronav.faa.gov_visual_10-02-2025_sectional-files_Hawaiian_Islands.zip",
    "hawaii": "web.archive.org_web_20251025040847_aeronav.faa.gov_visual_10-02-2025_sectional-files_Hawaiian_Islands.zip",
    "houston": "web.archive.org_web_20251025034727_aeronav.faa.gov_visual_10-02-2025_sectional-files_Houston.zip",
    "jacksonville": "web.archive.org_web_20251025034906_aeronav.faa.gov_visual_10-02-2025_sectional-files_Jacksonville.zip",
    "juneau": "web.archive.org_web_20251025042639_aeronav.faa.gov_visual_10-02-2025_sectional-files_Juneau.zip",
    "kansas_city": "web.archive.org_web_20251025031946_aeronav.faa.gov_visual_10-02-2025_sectional-files_Kansas_City.zip",
    "kansas city": "web.archive.org_web_20251025031946_aeronav.faa.gov_visual_10-02-2025_sectional-files_Kansas_City.zip",
    "ketchikan": "web.archive.org_web_20251025034719_aeronav.faa.gov_visual_10-02-2025_sectional-files_Ketchikan.zip",
    "klamath_falls": "web.archive.org_web_20251025041324_aeronav.faa.gov_visual_10-02-2025_sectional-files_Klamath_Falls.zip",
    "klamath falls": "web.archive.org_web_20251025041324_aeronav.faa.gov_visual_10-02-2025_sectional-files_Klamath_Falls.zip",
    "kodiak": "web.archive.org_web_20251025034736_aeronav.faa.gov_visual_10-02-2025_sectional-files_Kodiak.zip",
    "lake_huron": "web.archive.org_web_20251025031506_aeronav.faa.gov_visual_10-02-2025_sectional-files_Lake_Huron.zip",
    "lake huron": "web.archive.org_web_20251025031506_aeronav.faa.gov_visual_10-02-2025_sectional-files_Lake_Huron.zip",
    "las_vegas": "web.archive.org_web_20251025042604_aeronav.faa.gov_visual_10-02-2025_sectional-files_Las_Vegas.zip",
    "las vegas": "web.archive.org_web_20251025042604_aeronav.faa.gov_visual_10-02-2025_sectional-files_Las_Vegas.zip",
    "los_angeles": "web.archive.org_web_20251025043920_aeronav.faa.gov_visual_10-02-2025_sectional-files_Los_Angeles.zip",
    "los angeles": "web.archive.org_web_20251025043920_aeronav.faa.gov_visual_10-02-2025_sectional-files_Los_Angeles.zip",
    "mcgrath": "web.archive.org_web_20251025034933_aeronav.faa.gov_visual_10-02-2025_sectional-files_McGrath.zip",
    "memphis": "web.archive.org_web_20251025031449_aeronav.faa.gov_visual_10-02-2025_sectional-files_Memphis.zip",
    "miami": "web.archive.org_web_20251025040100_aeronav.faa.gov_visual_10-02-2025_sectional-files_Miami.zip",
    "montreal": "web.archive.org_web_20251025034731_aeronav.faa.gov_visual_10-02-2025_sectional-files_Montreal.zip",
    "new_orleans": "web.archive.org_web_20251025035945_aeronav.faa.gov_visual_10-02-2025_sectional-files_New_Orleans.zip",
    "new orleans": "web.archive.org_web_20251025035945_aeronav.faa.gov_visual_10-02-2025_sectional-files_New_Orleans.zip",
    "new_york": "web.archive.org_web_20251025042644_aeronav.faa.gov_visual_10-02-2025_sectional-files_New_York.zip",
    "new york": "web.archive.org_web_20251025042644_aeronav.faa.gov_visual_10-02-2025_sectional-files_New_York.zip",
    "nome": "web.archive.org_web_20251025040236_aeronav.faa.gov_visual_10-02-2025_sectional-files_Nome.zip",
    "omaha": "web.archive.org_web_20251025043420_aeronav.faa.gov_visual_10-02-2025_sectional-files_Omaha.zip",
    "phoenix": "web.archive.org_web_20251025035452_aeronav.faa.gov_visual_10-02-2025_sectional-files_Phoenix.zip",
    "point_barrow": "web.archive.org_web_20251025044542_aeronav.faa.gov_visual_10-02-2025_sectional-files_Point_Barrow.zip",
    "point barrow": "web.archive.org_web_20251025044542_aeronav.faa.gov_visual_10-02-2025_sectional-files_Point_Barrow.zip",
    "salt_lake_city": "web.archive.org_web_20251025031455_aeronav.faa.gov_visual_10-02-2025_sectional-files_Salt_Lake_City.zip",
    "salt lake city": "web.archive.org_web_20251025031455_aeronav.faa.gov_visual_10-02-2025_sectional-files_Salt_Lake_City.zip",
    "san_antonio": "web.archive.org_web_20251025035815_aeronav.faa.gov_visual_10-02-2025_sectional-files_San_Antonio.zip",
    "san antonio": "web.archive.org_web_20251025035815_aeronav.faa.gov_visual_10-02-2025_sectional-files_San_Antonio.zip",
    "san_francisco": "web.archive.org_web_20251025041808_aeronav.faa.gov_visual_10-02-2025_sectional-files_San_Francisco.zip",
    "san francisco": "web.archive.org_web_20251025041808_aeronav.faa.gov_visual_10-02-2025_sectional-files_San_Francisco.zip",
    "seattle": "web.archive.org_web_20251025040649_aeronav.faa.gov_visual_10-02-2025_sectional-files_Seattle.zip",
    "seward": "web.archive.org_web_20251025034327_aeronav.faa.gov_visual_10-02-2025_sectional-files_Seward.zip",
    "st_louis": "web.archive.org_web_20251025032956_aeronav.faa.gov_visual_10-02-2025_sectional-files_St_Louis.zip",
    "st louis": "web.archive.org_web_20251025032956_aeronav.faa.gov_visual_10-02-2025_sectional-files_St_Louis.zip",
    "twin_cities": "web.archive.org_web_20251025044057_aeronav.faa.gov_visual_10-02-2025_sectional-files_Twin_Cities.zip",
    "twin cities": "web.archive.org_web_20251025044057_aeronav.faa.gov_visual_10-02-2025_sectional-files_Twin_Cities.zip",
    "washington": "web.archive.org_web_20251025035531_aeronav.faa.gov_visual_10-02-2025_sectional-files_Washington.zip",
    "western_aleutian_islands": "web.archive.org_web_20251025031957_aeronav.faa.gov_visual_10-02-2025_sectional-files_Western_Aleutian_Islands.zip",
    "western aleutian islands": "web.archive.org_web_20251025031957_aeronav.faa.gov_visual_10-02-2025_sectional-files_Western_Aleutian_Islands.zip",
    "wichita": "web.archive.org_web_20251025040155_aeronav.faa.gov_visual_10-02-2025_sectional-files_Wichita.zip",
}


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _validate_pdf(pdf_path: Path) -> Optional[str]:
    """Return an error string if PDF looks invalid; otherwise None."""
    try:
        size = pdf_path.stat().st_size
    except FileNotFoundError:
        return "file missing"

    if size < 1024:
        return f"file too small ({size} bytes)"

    with open(pdf_path, "rb") as f:
        head = f.read(1024)
    head_lower = head.lower()
    if not head.startswith(b"%PDF-"):
        return "missing PDF header"
    if b"<html" in head_lower or b"<!doctype" in head_lower:
        return "html response"

    with open(pdf_path, "rb") as f:
        f.seek(max(0, size - 2048))
        tail = f.read(2048)
    if b"%%EOF" not in tail:
        return "missing PDF EOF marker"

    return None


def _repair_pdf(pdf_path: Path, update_status=None) -> Path:
    """Attempt to repair a PDF using qpdf if available."""
    qpdf = shutil.which("qpdf")
    if not qpdf:
        raise RuntimeError("qpdf not found for PDF repair")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    repaired_path = TEMP_DIR / f"{pdf_path.stem}_repaired.pdf"
    cmd = [qpdf, "--repair", str(pdf_path), str(repaired_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.strip() or "qpdf repair failed.")

    if update_status:
        update_status("PDF repaired with qpdf")
    return repaired_path


def _convert_pdf_to_tif(pdf_path: Path, update_status=None):
    if update_status:
        update_status("Converting PDF to TIF...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEMP_DIR / (pdf_path.stem + "_converted.tif")
    def run_gdal(input_path: Path):
        cmd = ["gdal_translate", "-of", "GTiff", str(input_path), str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    try:
        run_gdal(pdf_path)
    except FileNotFoundError:
        raise RuntimeError("gdal_translate not found. Install GDAL to convert PDFs.")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        if any(token in err for token in ("Invalid PDF", "Invalid XRef", "Top-level pages object")):
            if update_status:
                update_status("Repairing PDF (qpdf)...")
            repaired = _repair_pdf(pdf_path, update_status)
            try:
                run_gdal(repaired)
                repaired.unlink(missing_ok=True)
            except subprocess.CalledProcessError as e2:
                raise RuntimeError((e2.stderr or "").strip() or "gdal_translate failed after repair.")
        else:
            raise RuntimeError(err or "gdal_translate failed.")
    if update_status:
        update_status("PDF converted to TIF")
    return out_path


def _extract_location_name(filename: str) -> str:
    """Extract location name from filename like 'Houston.pdf' or 'New_York SEC.tif'."""
    # Remove extension
    name = Path(filename).stem
    # Remove common suffixes
    name = re.sub(r'\s*(SEC|TAC|Heli|VFR).*$', '', name, flags=re.IGNORECASE)
    # Normalize
    name = name.strip().lower()
    return name


def _find_reference_zip(location: str) -> Path:
    """Find the reference zip file for a location."""
    location_lower = location.lower()

    if location_lower in REFERENCE_ZIPS:
        zip_path = REFERENCE_DIR / REFERENCE_ZIPS[location_lower]
        if zip_path.exists():
            return zip_path

    # Try fuzzy matching
    for key, zip_name in REFERENCE_ZIPS.items():
        if location_lower in key or key in location_lower:
            zip_path = REFERENCE_DIR / zip_name
            if zip_path.exists():
                return zip_path

    raise RuntimeError(f"No reference found for location: {location}\nAvailable: {', '.join(sorted(set(REFERENCE_ZIPS.values())))}")


def _get_reference_tif(zip_path: Path, update_status=None) -> Path:
    """Extract or find the TIF from a reference zip."""
    if update_status:
        update_status("Loading reference from zip...")

    # Check if already extracted (look for directory with same name as zip)
    extract_dir = zip_path.parent / zip_path.stem
    if extract_dir.exists():
        tifs = list(extract_dir.glob("*.tif")) + list(extract_dir.glob("*.TIF"))
        if tifs:
            # Find the main SEC tif (not overview)
            for tif in tifs:
                if "SEC" in tif.name.upper() and "OVR" not in tif.name.upper():
                    if update_status:
                        update_status(f"Using existing: {tif.name}")
                    return tif
            if update_status:
                update_status(f"Using existing: {tifs[0].name}")
            return tifs[0]

    # Extract from zip
    if update_status:
        update_status("Extracting reference TIF from zip...")
    temp_dir = TEMP_DIR / f"ref_{zip_path.stem}"
    temp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        tif_files = [f for f in zf.namelist() if f.lower().endswith('.tif')]
        if not tif_files:
            raise RuntimeError(f"No TIF files found in {zip_path}")

        # Prefer SEC file
        target = None
        for f in tif_files:
            if "SEC" in f.upper() and "OVR" not in f.upper():
                target = f
                break
        if not target:
            target = tif_files[0]

        zf.extract(target, temp_dir)
        if update_status:
            update_status(f"Extracted: {target}")
        return temp_dir / target


def _load_image(path: Path, update_status=None, label="image"):
    if update_status:
        update_status(f"Loading {label}...")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {label} image: {path}")
    if update_status:
        update_status(f"{label} loaded ({img.shape[1]}x{img.shape[0]})")
    return img


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _copy_georeference(ref_path: Path, output_path: Path, update_status=None):
    """Copy georeference information from reference TIF to output TIF."""
    if update_status:
        update_status("Copying georeference info...")

    # Open reference and output datasets
    ref_ds = gdal.Open(str(ref_path), gdal.GA_ReadOnly)
    if ref_ds is None:
        raise RuntimeError(f"Failed to open reference for georeference: {ref_path}")

    out_ds = gdal.Open(str(output_path), gdal.GA_Update)
    if out_ds is None:
        ref_ds = None
        raise RuntimeError(f"Failed to open output for georeference update: {output_path}")

    # Copy geotransform
    geotransform = ref_ds.GetGeoTransform()
    if geotransform:
        out_ds.SetGeoTransform(geotransform)

    # Copy projection
    projection = ref_ds.GetProjection()
    if projection:
        out_ds.SetProjection(projection)

    # Copy metadata
    metadata = ref_ds.GetMetadata()
    if metadata:
        out_ds.SetMetadata(metadata)

    # Flush and close
    out_ds.FlushCache()
    out_ds = None
    ref_ds = None

    if update_status:
        update_status("Georeference info copied")


def _download_pdf(url: str, dest_path: Path, update_status=None, max_retries=3, retry_delay=30) -> Path:
    """Download a PDF from a Wayback Machine URL with retry logic."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Fix Wayback URL to get raw content (add id_ modifier)
    # Example: /web/20220627003205/ -> /web/20220627003205id_/
    if 'web.archive.org/web/' in url and 'id_' not in url:
        # Insert 'id_' after the timestamp (14-digit number)
        url = re.sub(r'/web/(\d{14})/', r'/web/\1id_/', url)

    for attempt in range(1, max_retries + 1):
        try:
            if update_status:
                if attempt > 1:
                    update_status(f"Downloading (attempt {attempt}/{max_retries})...")
                else:
                    update_status(f"Downloading PDF...")

            urllib.request.urlretrieve(url, dest_path)

            pdf_issue = _validate_pdf(dest_path)
            if pdf_issue:
                raise RuntimeError(f"Downloaded file invalid: {pdf_issue}")

            if update_status:
                update_status("Download complete")

            return dest_path

        except Exception as e:
            if attempt < max_retries:
                if update_status:
                    update_status(f"Download failed, waiting {retry_delay}s...")
                dest_path.unlink(missing_ok=True)
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")

    raise RuntimeError(f"Download failed after {max_retries} attempts")


def align_images(ref_path: Path, input_path: Path, output_path: Path, update_status=None):
    # Load images
    if update_status:
        update_status("Loading reference image...")
    ref = _load_image(ref_path, update_status, "reference")

    if update_status:
        update_status("Loading input image...")
    inp = _load_image(input_path, update_status, "input")

    if update_status:
        update_status("Converting to grayscale...")
    ref_gray = _to_gray(ref)
    inp_gray = _to_gray(inp)

    if update_status:
        update_status("Detecting features (ORB)...")
    orb = cv2.ORB_create(nfeatures=8000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(inp_gray, None)
    if des1 is None or des2 is None:
        raise RuntimeError("Not enough features found to align images.")

    if update_status:
        update_status(f"Matching features ({len(kp1)} ref, {len(kp2)} input)...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 10:
        raise RuntimeError(f"Not enough good matches: {len(good)}")

    if update_status:
        update_status(f"Estimating homography ({len(good)} matches)...")
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    if update_status:
        update_status("Warping to reference dimensions...")
    height, width = ref_gray.shape[:2]
    warped = cv2.warpPerspective(inp, H, (width, height))

    if update_status:
        update_status("Writing aligned output...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), warped):
        raise RuntimeError(f"Failed to write output: {output_path}")

    # Copy georeference information from reference to output
    _copy_georeference(ref_path, output_path, update_status)

    if update_status:
        update_status("Done!")


def _load_batch_state(state_path: Path) -> Optional[dict]:
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_batch_state(state_path: Path, state: dict) -> None:
    state["last_updated"] = datetime.now().isoformat()
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    temp_path.replace(state_path)


def _parse_resume_log(log_path: Path) -> dict:
    processed = {}
    errors = {}
    if not log_path.exists():
        return {"processed": processed, "errors": errors}
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SUCCESS: "):
                fname = line.replace("SUCCESS: ", "", 1).split(" -> ", 1)[0].strip()
                processed[fname] = "success"
            elif line.startswith("SKIP: "):
                fname = line.replace("SKIP: ", "", 1).split(" (", 1)[0].strip()
                processed[fname] = "skipped"
            elif line.startswith("FAILED: "):
                rest = line.replace("FAILED: ", "", 1)
                if " - " in rest:
                    fname, err = rest.split(" - ", 1)
                    processed[fname.strip()] = "failed"
                    errors[fname.strip()] = err.strip()
                else:
                    processed[rest.strip()] = "failed"
    return {"processed": processed, "errors": errors}


def process_batch(csv_path: Path, resume: bool = False, resume_path: Optional[Path] = None):
    """Process all PDF entries from master_dole.csv."""
    console.print(Panel.fit(
        "[bold cyan]Batch PDF Aligner[/bold cyan]\n"
        f"Processing PDFs from: {csv_path}\n"
        f"Output directory: {BATCH_OUTPUT_DIR}\n"
        f"Temp directory: {TEMP_DIR}\n"
        f"[dim]Reference dir: {REFERENCE_DIR}[/dim]",
        border_style="cyan"
    ))

    # Read CSV and filter for PDFs
    pdf_entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '.pdf' in row['download_link'].lower():
                pdf_entries.append(row)

    # Reverse to start from bottom and work up
    pdf_entries.reverse()

    console.print(f"\n[bold]Found {len(pdf_entries)} PDF entries to process (starting from bottom)[/bold]\n")

    # Create output and temp directories
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Track results and state
    results = {"success": [], "failed": [], "skipped": []}
    state_path = BATCH_OUTPUT_DIR / f"batch_state_{csv_path.stem}.json"
    state = None
    processed = {}
    errors = {}

    log_file = BATCH_OUTPUT_DIR / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_mode = "w"

    if resume:
        if resume_path and resume_path.suffix.lower() == ".json":
            state_path = resume_path
        if resume_path and resume_path.exists() and resume_path.suffix.lower() == ".txt":
            parsed = _parse_resume_log(resume_path)
            processed = parsed["processed"]
            errors = parsed["errors"]
            log_file = resume_path
            log_mode = "a"
        else:
            state = _load_batch_state(state_path)
            if state and state.get("csv_path") and state["csv_path"] != str(csv_path.resolve()):
                console.print("[yellow]Warning:[/yellow] batch state CSV mismatch; ignoring saved state.")
                state = None
            if state and isinstance(state.get("processed"), dict):
                for fname, info in state["processed"].items():
                    status = info.get("status")
                    if status:
                        processed[fname] = status
                        if status == "failed" and info.get("error"):
                            errors[fname] = info["error"]
            elif resume_path and resume_path.exists():
                parsed = _parse_resume_log(resume_path)
                processed = parsed["processed"]
                errors = parsed["errors"]
                log_file = resume_path
                log_mode = "a"

        for fname, status in processed.items():
            if status == "success":
                results["success"].append(fname)
            elif status == "failed":
                results["failed"].append((fname, errors.get(fname, "")))
            elif status == "skipped":
                results["skipped"].append(fname)

    if state is None:
        state = {
            "csv_path": str(csv_path.resolve()),
            "started": datetime.now().isoformat(),
            "processed": {},
        }
    if processed:
        for fname, status in processed.items():
            if fname not in state["processed"]:
                entry = {
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                }
                if status == "failed" and errors.get(fname):
                    entry["error"] = errors[fname]
                state["processed"][fname] = entry
        _save_batch_state(state_path, state)

    with open(log_file, log_mode) as log:
        if log_mode == "w":
            log.write(f"Batch processing started: {datetime.now()}\n")
            log.write(f"Total PDFs: {len(pdf_entries)}\n\n")
        else:
            log.write(f"\nResumed batch processing: {datetime.now()}\n\n")

        # Process each PDF
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Starting...", total=len(pdf_entries), completed=len(processed))

            for idx, entry in enumerate(pdf_entries, 1):
                url = entry['download_link']
                filename = entry['filename']
                location = entry['location']
                date = entry['date']

                # Output filename: change .pdf to .tif
                output_filename = filename.replace('.pdf', '.tif')
                output_path = BATCH_OUTPUT_DIR / output_filename

                def update_status(stage: str):
                    desc = f"[{idx}/{len(pdf_entries)}] {location} ({date}) • {stage}"
                    progress.update(task, description=desc)

                # Skip if already processed in a previous run
                if filename in processed:
                    update_status("Skipped (resume)")
                    continue

                # Skip if already exists
                if output_path.exists():
                    results['skipped'].append(filename)
                    log.write(f"SKIP: {filename} (already exists)\n")
                    state["processed"][filename] = {
                        "status": "skipped",
                        "output": str(output_path),
                        "timestamp": datetime.now().isoformat(),
                    }
                    _save_batch_state(state_path, state)
                    update_status("Skipped (already exists)")
                    progress.advance(task)
                    continue

                try:
                    # Download PDF
                    temp_pdf = TEMP_DIR / filename
                    update_status("Downloading PDF...")
                    _download_pdf(url, temp_pdf, update_status)

                    # Wait 7 seconds after download to avoid rate limiting
                    update_status("Waiting 7s (rate limit)...")
                    time.sleep(7)

                    # Convert PDF to TIF
                    update_status("Converting PDF to TIF...")
                    temp_tif = _convert_pdf_to_tif(temp_pdf, update_status)

                    # Find reference
                    update_status("Finding reference chart...")
                    try:
                        ref_zip = _find_reference_zip(location)
                    except RuntimeError:
                        # Try extracting from filename if location field doesn't work
                        loc_from_name = _extract_location_name(filename)
                        ref_zip = _find_reference_zip(loc_from_name)

                    update_status("Extracting reference TIF...")
                    ref_tif = _get_reference_tif(ref_zip, update_status)

                    # Align
                    update_status("Detecting features (ORB)...")
                    align_images(ref_tif, temp_tif, output_path, update_status)

                    # Cleanup temp files
                    temp_pdf.unlink(missing_ok=True)
                    temp_tif.unlink(missing_ok=True)

                    results['success'].append(filename)
                    log.write(f"SUCCESS: {filename} -> {output_filename}\n")
                    log.flush()
                    state["processed"][filename] = {
                        "status": "success",
                        "output": str(output_path),
                        "timestamp": datetime.now().isoformat(),
                    }
                    _save_batch_state(state_path, state)

                    update_status("Complete")
                    progress.advance(task)

                except Exception as e:
                    results['failed'].append((filename, str(e)))
                    log.write(f"FAILED: {filename} - {e}\n")
                    log.flush()
                    state["processed"][filename] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    _save_batch_state(state_path, state)
                    update_status(f"Failed: {str(e)[:50]}")
                    progress.advance(task)
                    continue

    # Print summary
    console.print("\n" + "="*60)
    console.print(f"[bold green]Batch processing complete![/bold green]")

    table = Table(title="Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_row("Success", str(len(results['success'])))
    table.add_row("Failed", str(len(results['failed'])))
    table.add_row("Skipped", str(len(results['skipped'])))
    table.add_row("Total", str(len(pdf_entries)))

    console.print(table)
    console.print(f"\n[dim]Log file: {log_file}[/dim]")

    if results['failed']:
        console.print(f"\n[yellow]Failed entries:[/yellow]")
        for fname, error in results['failed'][:10]:  # Show first 10
            console.print(f"  - {fname}: {error}")
        if len(results['failed']) > 10:
            console.print(f"  ... and {len(results['failed']) - 10} more (see log file)")


def main():
    # Check for batch mode
    args = sys.argv[1:]
    if '--batch' in args:
        batch_idx = args.index('--batch')
        if batch_idx + 1 >= len(args):
            console.print("[red]Error:[/red] --batch requires a CSV file path")
            console.print("[dim]Usage:[/dim] python align_tif_gui.py --batch master_dole.csv [--resume [log_or_state_path]]")
            sys.exit(1)
        csv_path = Path(args[batch_idx + 1])
        if not csv_path.exists():
            console.print(f"[red]Error:[/red] CSV file not found: {csv_path}")
            sys.exit(1)
        resume = '--resume' in args
        resume_path = None
        if resume:
            resume_idx = args.index('--resume')
            if resume_idx + 1 < len(args) and not args[resume_idx + 1].startswith('--'):
                resume_path = Path(args[resume_idx + 1])
        process_batch(csv_path, resume=resume, resume_path=resume_path)
        return

    # Single file mode
    console.print(Panel.fit(
        "[bold cyan]FAA Chart Aligner[/bold cyan]\n"
        "Aligns input images to reference sectional charts\n"
        f"[dim]Reference dir: {REFERENCE_DIR}[/dim]",
        border_style="cyan"
    ))

    # Parse arguments or prompt for input
    if len(sys.argv) >= 2:
        inp_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    elif len(sys.argv) == 1:
        console.print("\n[dim]Enter the image to align (reference will be auto-detected)[/dim]\n")
        inp_path = Prompt.ask("[bold]Input image[/bold] (PDF or TIF)")
        out_path = Prompt.ask("[bold]Output path[/bold]", default="")
        if not out_path:
            out_path = None
    else:
        console.print("[red]Usage:[/red] python align_tif_gui.py <input> [output]")
        console.print("   or: python align_tif_gui.py --batch master_dole.csv [--resume [log_or_state_path]]")
        console.print("   or: python align_tif_gui.py   (interactive mode)")
        sys.exit(1)

    inp = Path(inp_path)
    if not inp.exists():
        console.print(f"[red]Error:[/red] Input file not found: {inp}")
        sys.exit(1)

    # Extract location name from input filename
    location = _extract_location_name(inp.name)
    console.print(f"\n[dim]Detected location:[/dim] [bold]{location}[/bold]")

    # Find reference
    try:
        ref_zip = _find_reference_zip(location)
        console.print(f"[dim]Reference zip:[/dim] {ref_zip.name}")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Default output path
    if out_path is None:
        out_path = inp.parent / f"{inp.stem}_aligned.tif"
    else:
        out_path = Path(out_path)

    console.print(f"[dim]Output:[/dim] {out_path}\n")

    # Run alignment with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...")

        def update_status(message):
            progress.update(task, description=message)

        try:
            # Convert PDF if needed
            if _is_pdf(inp):
                update_status("Converting PDF to TIF...")
                inp = _convert_pdf_to_tif(inp, update_status)

            # Get reference TIF
            update_status("Loading reference...")
            ref_tif = _get_reference_tif(ref_zip, update_status)

            # Align
            update_status("Aligning images...")
            align_images(ref_tif, inp, out_path, update_status)

            console.print(f"\n[green]Success![/green] Aligned image saved to: [bold]{out_path}[/bold]")
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
