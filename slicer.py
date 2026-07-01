#!/usr/bin/env python3
"""
FAA Chart Newslicer - CLI tool for processing sectional charts
Pipeline: download -> warp/cut -> mosaic VRT -> GeoTIFF (one per date).
"""

import os
import sys
import re
import json
import argparse
import zipfile
import shutil
try:
    import requests
except ImportError:
    requests = None
import time
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dole_dataset import load_combined_rows

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL is required but not installed.")
    print("Install with: pip install gdal --break-system-packages")
    sys.exit(1)

import multiprocessing
cpu_count = multiprocessing.cpu_count()
IS_APPLE_SILICON = (sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"})

# Warp outputs whose EPSG:3857 width reaches ~full-world are pathological (bad georef).
WORLD_WIDTH_M = 35_000_000.0


def _get_memory_snapshot_mb(
    default_available_mb: int = 4096,
    default_total_mb: int = 8192
) -> Dict[str, int]:
    """Return a best-effort memory snapshot in MB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_mb = int(vm.available // (1024 * 1024))
        total_mb = int(vm.total // (1024 * 1024))
        return {
            "available_mb": max(1, available_mb),
            "total_mb": max(max(1, available_mb), total_mb),
        }
    except Exception:
        fallback_total = max(default_total_mb, default_available_mb)
        return {
            "available_mb": max(1, default_available_mb),
            "total_mb": max(1, fallback_total),
        }


def _recommended_free_ram_floor_mb(total_ram_mb: int) -> int:
    """RAM headroom to preserve for OS + transient process spikes."""
    total_ram_mb = max(1024, int(total_ram_mb or 0))
    return max(768, min(2048, total_ram_mb // 10))


def _get_available_ram_mb(default_mb: int = 4096) -> int:
    """Return currently available RAM in MB, or a conservative fallback."""
    return _get_memory_snapshot_mb(default_available_mb=default_mb)["available_mb"]


# Set minimal GDAL global defaults (defer thread decisions to phase-specific config)
available_ram_mb = _get_available_ram_mb(default_mb=4096)
cache_size_mb = max(512, min(8192, available_ram_mb * 3 // 4))  # 75% of RAM, max 8GB

# Only set cache at global level - thread settings will be phase-specific
gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')


def _configure_gdal_for_phase(phase: str, workers: int = 1, available_ram_mb: int = None):
    """
    Configure GDAL settings for specific processing phases.

    Args:
        phase: One of 'warp', 'geotiff'
        workers: Number of parallel workers
        available_ram_mb: Available RAM in MB (auto-detect if None)
    """
    if available_ram_mb is None:
        available_ram_mb = _get_available_ram_mb(default_mb=4096)

    if phase == 'warp':
        # Warp phase: single-threaded per worker to prevent oversubscription
        gdal.SetConfigOption('GDAL_NUM_THREADS', '1')
        cache_per_worker = max(256, available_ram_mb // workers)
        gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_per_worker))

    elif phase == 'geotiff':
        # GeoTIFF phase: single-process translate with full CPU
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(max(512, min(8192, available_ram_mb * 3 // 4))))


class ChartSlicer:
    """Process FAA charts and create COGs."""

    @staticmethod
    def _check_zstd_support() -> bool:
        """Check if GDAL supports ZSTD compression."""
        try:
            driver = gdal.GetDriverByName('GTiff')
            if driver:
                metadata = driver.GetMetadata()
                if metadata and 'ZSTD' in metadata.get('DMD_CREATIONOPTIONLIST', ''):
                    return True
        except:
            pass
        return False

    def __init__(self, source_dir: Path, output_dir: Path, csv_file: Path, shape_dir: Path,
                 preview_mode: bool = False, download_delay: float = 8.0, temp_dir: Path = None):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.shape_dir = shape_dir
        self.temp_dir = temp_dir if temp_dir else Path("/Volumes/projects/.temp")
        self.dole_data = defaultdict(list)
        self.srs_cache = {}  # Cache for shapefile SRS lookups
        self.preview_mode = preview_mode  # If True, only report what would be downloaded
        self.download_delay = download_delay  # Seconds to wait between downloads (to avoid rate limiting)
        self.last_download_time = 0  # Track last download time for throttling
        self._date_cache: Dict[str, Optional[datetime.date]] = {}
        self.date_key_map: Dict[str, Tuple[str, str]] = {}
        # GeoTIFF output settings (final pipeline stage: mosaic VRT -> GeoTIFF)
        self.geotiff_compress = 'ZSTD'
        self.num_threads = 'ALL_CPUS'
        self.clip_projwin: Optional[Tuple[float, float, float, float]] = None
        # GeoTIFF progress/ETA tracking (populated in process_all_dates)
        self.geotiff_total_planned = 0
        self.geotiff_completed = 0
        self.geotiff_times: List[float] = []
        self.parallel_warp = 0
        # In threaded warp mode we already parallelize by job; avoid extra internal threads on Apple Silicon.
        self.warp_multithread = not IS_APPLE_SILICON
        self.abbreviations = {
            "hawaiian_is": "hawaiian_islands",
            "dallas_ft_worth": "dallas_ft_worth",
            "mariana_islands_inset": "mariana_islands",
        }
        # Use default PROJ transformer selection to avoid strict-op failures on Alaska/dateline charts.
        self.transformer_options = None

        # File index caches for O(1) lookups
        self.files_by_name: Dict[str, Path] = {}  # Direct TIFs and ZIPs
        self.tifs_by_dir: Dict[str, List[Path]] = {}  # Extracted folder -> TIF list
        self.shapefile_index: Dict[str, Path] = {}  # Normalized location -> shapefile path

    def log(self, msg: str):
        """Print log message with timestamp."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def normalize_name(self, name: str) -> str:
        """Normalize chart name consistently."""
        norm = name.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace(',', '')
        while '__' in norm:
            norm = norm.replace('__', '_')
        return self.abbreviations.get(norm, norm)

    def sanitize_filename(self, name: str) -> str:
        """Remove spaces and special characters from filename."""
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        name = name.replace('.', '_')
        name = name.replace('(', '')
        name = name.replace(')', '')
        name = name.replace(',', '')
        while '__' in name:
            name = name.replace('__', '_')
        return name.lower()

    @staticmethod
    def _row_pick(row: Dict[str, str], keys: List[str]) -> str:
        """Return first non-empty value from a list of possible CSV keys."""
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
        return ""

    def _row_float(self, row: Dict[str, str], keys: List[str]) -> Optional[float]:
        """Parse first available key as float."""
        value = self._row_pick(row, keys)
        if not value:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_lcc_crs(self, lat1: str, lat2: str, lat0: str, lon0: str) -> str:
        if not (lat1 and lat2 and lat0 and lon0):
            return ""
        return (
            f"+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} "
            f"+lon_0={lon0} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
        )

    def _build_gcp_warp_context(self, row: Dict[str, str]) -> Optional[Dict[str, object]]:
        """
        Build per-row GCP warp inputs:
        - pixel corners (tl/tr/br/bl x,y)
        - projected GCP coordinates in source CRS
        - EPSG:4269 cutline polygon
        """
        pixels = []
        for x_key, y_key in (("tl_x", "tl_y"), ("tr_x", "tr_y"), ("br_x", "br_y"), ("bl_x", "bl_y")):
            px = self._row_float(row, [x_key, x_key.upper()])
            py = self._row_float(row, [y_key, y_key.upper()])
            if px is None or py is None:
                return None
            pixels.append((px, py))

        gcp_geo = []
        has_all_gcp_geo = True
        for lat_key, lon_key in (
            ("gcp_tl_lat", "gcp_tl_lon"),
            ("gcp_tr_lat", "gcp_tr_lon"),
            ("gcp_br_lat", "gcp_br_lon"),
            ("gcp_bl_lat", "gcp_bl_lon"),
        ):
            lat = self._row_float(row, [lat_key, lat_key.upper()])
            lon = self._row_float(row, [lon_key, lon_key.upper()])
            if lat is None or lon is None:
                has_all_gcp_geo = False
                break
            gcp_geo.append((lon, lat))

        west = self._row_float(row, ["Left bounds", "Left cut bounds", "left_bounds", "left_cut_bounds"])
        east = self._row_float(row, ["Right bounds", "Right cut bounds", "right_bounds", "right_cut_bounds"])
        north = self._row_float(row, ["Top bounds", "Top cut bounds", "top_bounds", "top_cut_bounds"])
        south = self._row_float(row, ["Bottom bounds", "Bottom cut bounds", "bottom_bounds", "bottom_cut_bounds"])

        bounds_ok = all(v is not None for v in (west, east, north, south))
        if bounds_ok:
            if west > 0:
                west = -west
            if east > 0:
                east = -east
            if west > east:
                west, east = east, west
            if south > north:
                south, north = north, south

        if not has_all_gcp_geo and not bounds_ok:
            return None

        if not has_all_gcp_geo and bounds_ok:
            gcp_geo = [
                (west, north),
                (east, north),
                (east, south),
                (west, south),
            ]

        # Prefer explicit CRS from CSV; otherwise synthesize LCC from projection fields.
        crs_str = self._row_pick(row, ["crs", "CRS"])
        if not crs_str:
            lat1 = self._row_pick(row, ["proj_lat1", "PROJ_LAT1"]) or "33.0"
            lat2 = self._row_pick(row, ["proj_lat2", "PROJ_LAT2"]) or "45.0"
            lat0 = self._row_pick(row, ["proj_lat0", "PROJ_LAT0"]) or "39.0"
            lon0 = self._row_pick(row, ["proj_lon0", "PROJ_LON0"])
            if not lon0:
                if bounds_ok:
                    lon0 = str((west + east) / 2.0)
                else:
                    lon0 = "-96.0"
            crs_str = self._build_lcc_crs(lat1, lat2, lat0, lon0) or \
                "+proj=lcc +lat_1=33.0 +lat_2=45.0 +lat_0=39.0 +lon_0=-96.0 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

        src_srs = osr.SpatialReference()
        dst_srs = osr.SpatialReference()
        try:
            if src_srs.ImportFromEPSG(4269) != 0:
                return None
            if dst_srs.SetFromUserInput(crs_str) != 0:
                return None
        except RuntimeError as e:
            # Malformed CRS (e.g. +lon_0=NaN) raises rather than returning non-zero
            # when GDAL exceptions are enabled. Skip GCP warp; fall back to shapefile.
            self.log(f"    invalid CRS, falling back to shapefile warp: {crs_str!r} ({e})")
            return None
        if hasattr(src_srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        transformer = osr.CoordinateTransformation(src_srs, dst_srs)
        projected = []
        for lon, lat in gcp_geo:
            x, y, _z = transformer.TransformPoint(float(lon), float(lat))
            projected.append((x, y))

        if bounds_ok:
            cutline_coords = [
                [west, north],
                [east, north],
                [east, south],
                [west, south],
                [west, north],
            ]
        else:
            cutline_coords = [[lon, lat] for lon, lat in gcp_geo]
            cutline_coords.append([gcp_geo[0][0], gcp_geo[0][1]])

        return {
            "pixels": pixels,
            "projected": projected,
            "cutline_coords": cutline_coords,
            "source_crs": crs_str,
        }

    def _date_from_str(self, date_str: str) -> Optional[datetime.date]:
        """Parse YYYY-MM-DD date strings with caching."""
        if date_str in self._date_cache:
            return self._date_cache[date_str]
        try:
            parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            parsed = None
        self._date_cache[date_str] = parsed
        return parsed

    def _date_range_key(self, start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
        """Build a stable date key that includes both start and end dates."""
        if not start_date:
            return None
        if not end_date or end_date == start_date:
            return start_date
        return f"{start_date}_to_{end_date}"

    def _date_key_sort_key(self, date_key: str) -> datetime.date:
        """Sort by start date for range keys."""
        start_date = self.date_key_map.get(date_key, (date_key, None))[0]
        parsed = self._date_from_str(start_date)
        return parsed if parsed else datetime.date.min

    def load_dole_data(self):
        """Load CSV data (combined or legacy) into date-keyed groups."""
        self.log(f"Loading CSV: {self.csv_file.name}...")
        self.dole_data = defaultdict(list)
        self.date_key_map = {}
        total_rows = 0
        accepted_rows = 0
        skipped_missing_filename = 0
        undated_rows = 0

        normalized_rows = load_combined_rows(self.csv_file)
        for row_idx, row in enumerate(normalized_rows, start=1):
            total_rows += 1

            filename = (row.get('filename') or '').strip()
            if not filename:
                skipped_missing_filename += 1
                continue

            link = row.get('download_link', '')
            timestamp = None
            if 'web.archive.org/web/' in link:
                match = re.search(r'/web/(\d{14})/', link)
                if match:
                    timestamp = match.group(1)
            row['wayback_ts'] = timestamp

            start_date_raw = str(row.get('date') or '').strip()
            end_date_raw = str(row.get('end_date') or start_date_raw).strip()

            parsed_start = self._date_from_str(start_date_raw) if start_date_raw else None
            parsed_end = self._date_from_str(end_date_raw) if end_date_raw else parsed_start

            start_date = parsed_start.isoformat() if parsed_start else start_date_raw
            end_date = parsed_end.isoformat() if parsed_end else (end_date_raw or start_date)

            if start_date:
                date_key = self._date_range_key(start_date, end_date)
                if not date_key:
                    continue
            else:
                undated_rows += 1
                date_key = f"undated_{Path(filename).stem or f'row{row_idx}'}"

            if not row.get('candidate_group'):
                norm_loc = self.normalize_name(row.get('location', ''))
                if start_date and norm_loc:
                    row['candidate_group'] = f"{start_date}|{norm_loc}"
                else:
                    row['candidate_group'] = f"undated_{Path(filename).stem or f'row{row_idx}'}"

            if not row.get('candidate_rank'):
                row['candidate_rank'] = "3"

            row['start_date'] = start_date
            row['end_date'] = end_date if start_date else ""
            row['date_key'] = date_key
            self.dole_data[date_key].append(row)
            if date_key not in self.date_key_map:
                self.date_key_map[date_key] = (start_date, end_date)
            accepted_rows += 1

        self.log(f"Loaded {len(self.dole_data)} date ranges.")
        self.log(
            f"Accepted {accepted_rows}/{total_rows} rows "
            f"(undated={undated_rows}, missing filename skipped={skipped_missing_filename})."
        )

    def _build_source_index(self):
        """
        Build file index for O(1) lookups instead of repeated os.walk calls.

        Creates:
        - files_by_name: Dict[str, Path] for direct TIFs and ZIPs
        - tifs_by_dir: Dict[str, List[Path]] keyed by extracted folder (ZIP stem)
        """
        self.log("Building source file index...")

        # Walk source_dir once and cache all files
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)

            for filename in files:
                file_path = root_path / filename

                # Cache direct TIF and ZIP files by name
                if filename.endswith('.tif') or filename.endswith('.zip'):
                    self.files_by_name[filename] = file_path

                # For TIFs inside extracted directories, also index by parent directory name
                if filename.endswith('.tif'):
                    parent_dir = root_path.name
                    # Only index if parent is not source_dir itself
                    if root_path != self.source_dir:
                        if parent_dir not in self.tifs_by_dir:
                            self.tifs_by_dir[parent_dir] = []
                        self.tifs_by_dir[parent_dir].append(file_path)

        self.log(f"  Indexed {len(self.files_by_name)} files and {len(self.tifs_by_dir)} directories")

    def _build_shapefile_index(self):
        """
        Build shapefile index for O(1) lookups instead of repeated globbing.

        Creates shapefile_index: Dict[str, Path] keyed by normalized location name.
        """
        self.log("Building shapefile index...")

        sectional_dir = self.shape_dir / "sectional"
        if not sectional_dir.exists():
            self.log(f"  WARNING: Sectional shapefile directory not found at {sectional_dir}")
            return

        # One controlled glob to find all shapefiles
        candidates = list(sectional_dir.glob("**/*.shp"))

        for shp in candidates:
            # Normalize the shapefile name once
            shp_norm = self.normalize_name(shp.stem)
            # Store with normalized key
            self.shapefile_index[shp_norm] = shp

            # Also check for partial matches (for locations that might be substrings)
            # We'll do exact matches first in find_shapefile, then partial matches

        self.log(f"  Indexed {len(self.shapefile_index)} shapefiles")

    def download_file(self, url: str, target_path: Path, timeout: int = 300, max_retries: int = 3) -> bool:
        """
        Download a file from URL to target path with throttling and retry logic.

        Implements:
        - Throttling to avoid rate limits (configurable download_delay)
        - Exponential backoff retry logic on failure
        - Progress reporting

        Returns: True if successful, False otherwise
        """
        if requests is None:
            self.log("ERROR: requests is required for downloading files. Install with: pip install requests")
            return False

        # Throttle downloads to avoid rate limiting
        elapsed = time.time() - self.last_download_time
        if elapsed < self.download_delay:
            wait_time = self.download_delay - elapsed
            self.log(f"    Waiting {wait_time:.1f}s to avoid rate limiting...")
            time.sleep(wait_time)

        retry_count = 0
        backoff_delay = 5  # Start with 5 second backoff

        while retry_count < max_retries:
            try:
                self.log(f"    Downloading: {url}")
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                # Get total file size
                total_size = int(response.headers.get('content-length', 0))

                # Download with progress
                downloaded = 0
                last_logged_percent = -10  # Track last logged percentage
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                current_percent = int((downloaded / total_size) * 100)
                                if current_percent >= last_logged_percent + 10:
                                    self.log(f"      Progress: {current_percent}%")
                                    last_logged_percent = current_percent

                file_size_mb = target_path.stat().st_size / (1024 * 1024)
                self.log(f"    ✓ Downloaded: {target_path.name} ({file_size_mb:.1f} MB)")

                # Update file index with newly downloaded file
                self.files_by_name[target_path.name] = target_path

                # Update last download time for throttling
                self.last_download_time = time.time()
                return True

            except Exception as e:
                retry_count += 1
                error_str = str(e)
                self.log(f"    ✗ Download failed (attempt {retry_count}/{max_retries}): {error_str[:100]}")

                # Detect connection reset errors - server is rejecting us, slow down aggressively
                if 'Connection reset' in error_str or 'Connection aborted' in error_str:
                    # Increase global download delay to back off more for future requests
                    old_delay = self.download_delay
                    self.download_delay = max(self.download_delay * 1.5, self.download_delay + 5)
                    self.log(f"    ⚠ Connection error detected - increasing delay from {old_delay:.1f}s to {self.download_delay:.1f}s")

                if retry_count < max_retries:
                    self.log(f"    Retrying in {backoff_delay} seconds...")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2  # Exponential backoff
                    # Remove partial file before retry
                    if target_path.exists():
                        target_path.unlink()
                else:
                    self.log(f"    Giving up after {max_retries} attempts")
                    if target_path.exists():
                        target_path.unlink()
                    return False

        return False

    def extract_zip(self, zip_path: Path, extract_dir: Path) -> bool:
        """
        Extract ZIP file to a directory with the same name (minus .zip).

        Returns: True if successful, False otherwise
        """
        try:
            if extract_dir.exists():
                self.log(f"    Directory already exists: {extract_dir.name}")
                return True

            self.log(f"    Extracting: {zip_path.name}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Count extracted files and update index
            tif_files = list(extract_dir.glob('**/*.tif'))
            tif_count = len(tif_files)

            # Update tifs_by_dir index with newly extracted files
            dir_name = extract_dir.name
            self.tifs_by_dir[dir_name] = tif_files

            # Also add individual TIF files to files_by_name
            for tif_file in tif_files:
                self.files_by_name[tif_file.name] = tif_file

            self.log(f"    ✓ Extracted {tif_count} files to: {extract_dir.name}")
            return True

        except Exception as e:
            self.log(f"    ✗ Extraction failed: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            return False

    def _download_missing_files(self, allowed_dates: Optional[set] = None):
        """
        Preparation phase: Download all missing files upfront before processing begins.

        This collects all files referenced in self.dole_data that don't exist locally,
        then downloads them sequentially with proper throttling. This prevents burst
        requests to web.archive.org when processing dates with many missing files.

        Respects self.download_delay and self.preview_mode settings.
        """
        # Collect all files that need downloading
        missing_files = []  # List of (filename, download_link, location, date, is_zip)

        for date in self.dole_data:
            if allowed_dates is not None and date not in allowed_dates:
                continue
            for record in self.dole_data[date]:
                filename = record.get('filename', '')
                download_link = record.get('download_link', '')
                location = record.get('location', '')

                if not filename or not download_link:
                    continue

                # Check if file exists locally
                if filename.endswith('.zip'):
                    # ZIP files become directories after extraction
                    dir_name = filename[:-4]
                    dir_path = self.source_dir / dir_name
                    if not dir_path.exists():
                        missing_files.append((filename, download_link, location, date, True))
                else:
                    # Direct TIF files
                    file_path = self.source_dir / filename
                    if not file_path.exists():
                        missing_files.append((filename, download_link, location, date, False))

        if not missing_files:
            self.log("No missing files to download.")
            return

        # Sort by date (latest first to match processing order)
        missing_files.sort(key=lambda x: self._date_key_sort_key(x[3]), reverse=True)

        self.log(f"Found {len(missing_files)} missing files to download (sorted latest→oldest)")
        self.log(f"Download throttle: {self.download_delay}s between requests")

        # Estimate total time
        estimated_secs = len(missing_files) * self.download_delay
        estimated_mins = estimated_secs / 60
        self.log(f"Estimated time: ~{estimated_mins:.0f} minutes (assuming {self.download_delay}s per file)\n")

        # Download each file
        downloaded_count = 0
        failed_count = 0

        for idx, (filename, download_link, location, date, is_zip) in enumerate(missing_files, 1):
            self.log(f"  [{idx}/{len(missing_files)}] {date} - {location}")

            target_path = self.source_dir / filename

            # Download the file
            if self.download_file(download_link, target_path):
                downloaded_count += 1

                # If ZIP, extract it
                if is_zip:
                    dir_name = filename[:-4]
                    extract_dir = self.source_dir / dir_name
                    if self.extract_zip(target_path, extract_dir):
                        self.log(f"    ✓ Downloaded and extracted: {filename}")
                    else:
                        self.log(f"    ⚠ Downloaded but extraction failed: {filename}")
                        failed_count += 1
                else:
                    self.log(f"    ✓ Downloaded: {filename}")

                # Add extra pause after successful download to avoid rate limits
                # This is in addition to the throttling in download_file()
                if idx < len(missing_files):  # Don't pause after last file
                    self.log(f"    Pausing {self.download_delay}s before next request...")
                    time.sleep(self.download_delay)
            else:
                self.log(f"    ✗ Failed to download: {filename}")
                failed_count += 1
                # Still pause before retry to avoid hammering the server
                if idx < len(missing_files):
                    time.sleep(self.download_delay)

        # Summary
        self.log(f"\n=== Download Preparation Complete ===")
        self.log(f"Downloaded: {downloaded_count}/{len(missing_files)}")
        if failed_count > 0:
            self.log(f"Failed: {failed_count}")
        self.log(f"Processing will now use these files as they're available locally.\n")

    def resolve_filename(self, filename: str, download_link: str = '') -> List[Tuple[Path, Optional[str]]]:
        """
        Resolve filename from CSV to actual file path(s) using index cache.

        Supports:
        1. Direct .tif files: "file.tif" → finds file.tif
        2. Unzipped .zip directories: "file.zip" → finds all .tif files in file/ directory
        3. Auto-download if not found locally (when download_link provided)

        Returns: List of (file_path, None) tuples
        """
        if not filename:
            return []

        results = []

        # Case 1: .zip file (after unzipping, this becomes a directory)
        if filename.endswith('.zip'):
            # Remove .zip extension to get directory name
            dir_name = filename[:-4]  # Remove '.zip'

            # Check index FIRST (O(1) lookup)
            if dir_name in self.tifs_by_dir:
                # Found in index - return cached list
                for tif_file in self.tifs_by_dir[dir_name]:
                    results.append((tif_file, None))
                return results

            # Not in index - check if directory exists but wasn't indexed
            direct_path = self.source_dir / dir_name
            if direct_path.exists() and direct_path.is_dir():
                tif_files = list(direct_path.glob('**/*.tif'))
                # Update index for future lookups
                self.tifs_by_dir[dir_name] = tif_files
                for tif_file in tif_files:
                    if tif_file.is_file():
                        results.append((tif_file, None))
                if results:
                    return results

            # If not found, try to download (but not in preview mode)
            if not results and download_link and not self.preview_mode:
                self.log(f"    File not found locally: {filename}")
                zip_path = self.source_dir / filename
                extract_dir = self.source_dir / dir_name

                # Download the ZIP file
                if self.download_file(download_link, zip_path):
                    # Extract it (this will update the index)
                    if self.extract_zip(zip_path, extract_dir):
                        # Get the updated index entry
                        if dir_name in self.tifs_by_dir:
                            for tif_file in self.tifs_by_dir[dir_name]:
                                results.append((tif_file, None))
                        return results
                    else:
                        self.log(f"    Failed to extract: {filename}")
                else:
                    self.log(f"    Failed to download: {download_link}")

        else:
            # Case 2: Direct .tif file
            # Check index FIRST (O(1) lookup)
            if filename in self.files_by_name:
                results.append((self.files_by_name[filename], None))
                return results

            # Not in index - check if file exists but wasn't indexed
            direct_path = self.source_dir / filename
            if direct_path.exists() and direct_path.is_file():
                # Update index for future lookups
                self.files_by_name[filename] = direct_path
                results.append((direct_path, None))
                return results

            # If not found, try to download (but not in preview mode)
            if not results and download_link and not self.preview_mode:
                self.log(f"    File not found locally: {filename}")
                tif_path = self.source_dir / filename

                if self.download_file(download_link, tif_path):
                    results.append((tif_path, None))
                else:
                    self.log(f"    Failed to download: {download_link}")

        return results

    def find_files(self, location: str, edition: str, timestamp: Optional[str], filename: Optional[str] = None, download_link: str = '') -> List[Union[Path, Tuple[Path, str]]]:
        """
        Find files from CSV filename.
        Filename must be provided in master_dole.csv.

        If file not found locally and download_link provided, will attempt download and extract.

        Returns:
            - List of (file_path, None) tuples for files
        """
        if not filename:
            return []

        resolved = self.resolve_filename(filename, download_link)
        return resolved

    def find_shapefile(self, location: str) -> Optional[Path]:
        """Find shapefile for location using index cache (O(1) lookup)."""
        norm_loc = self.normalize_name(location)

        # Check for exact match first (O(1))
        if norm_loc in self.shapefile_index:
            return self.shapefile_index[norm_loc]

        # Check for partial matches (substring search)
        for shp_norm, shp_path in self.shapefile_index.items():
            if norm_loc in shp_norm:
                return shp_path

        return None

    def get_shapefile_srs(self, shapefile: Path) -> str:
        """Get the actual SRS of a shapefile in Proj4 format. Results are cached."""
        # Check cache first
        shapefile_key = str(shapefile)
        if shapefile_key in self.srs_cache:
            return self.srs_cache[shapefile_key]

        try:
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dataset = driver.Open(str(shapefile), 0)
            if not dataset:
                result = 'EPSG:4326'  # Fallback to WGS84
            else:
                layer = dataset.GetLayer(0)
                srs = layer.GetSpatialRef()
                dataset = None

                if srs:
                    proj4 = srs.ExportToProj4()
                    result = proj4 if proj4 else 'EPSG:4326'
                else:
                    result = 'EPSG:4326'  # Fallback
        except Exception:
            result = 'EPSG:4326'  # Fallback on any error

        # Cache the result
        self.srs_cache[shapefile_key] = result
        return result

    def _pick_shapefile_for_source(self, location: str, src_file: Path, default_shp: Optional[Path]) -> Optional[Path]:
        """
        Prefer directional shapefile variants (east/west/north/south) when the source filename
        encodes a directional split chart, e.g. Western Aleutian Islands East/West.
        """
        if default_shp is None:
            return None

        norm_loc = self.normalize_name(location)
        stem_norm = self.normalize_name(src_file.stem)
        for suffix in ("_east", "_west", "_north", "_south"):
            if suffix in stem_norm:
                variant_key = f"{norm_loc}{suffix}"
                variant = self.shapefile_index.get(variant_key)
                if variant:
                    return variant
        return default_shp

    def _antimeridian_split_bounds_3857(self, input_tiff: Path) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Return split output bounds (EPSG:3857) for sources that cross the antimeridian.
        When a source spans +180/-180 in geographic space, a single warp can balloon to nearly
        world width. Splitting into east/west output windows keeps extents narrow.
        """
        try:
            ds = gdal.Open(str(input_tiff))
            if not ds:
                return None

            gt = ds.GetGeoTransform(can_return_null=True)
            projection = ds.GetProjection()
            if gt is None or not projection:
                ds = None
                return None

            x_size = ds.RasterXSize
            y_size = ds.RasterYSize
            ds = None

            src_srs = osr.SpatialReference()
            if src_srs.ImportFromWkt(projection) != 0:
                return None
            geo_srs = osr.SpatialReference()
            if geo_srs.ImportFromEPSG(4326) != 0:
                return None
            merc_srs = osr.SpatialReference()
            if merc_srs.ImportFromEPSG(3857) != 0:
                return None

            if hasattr(src_srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                geo_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                merc_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            to_geo = osr.CoordinateTransformation(src_srs, geo_srs)
            to_merc = osr.CoordinateTransformation(geo_srs, merc_srs)

            corners_px = [
                (0.0, 0.0),
                (float(x_size), 0.0),
                (float(x_size), float(y_size)),
                (0.0, float(y_size)),
            ]
            lon_lat: List[Tuple[float, float]] = []
            for px, py in corners_px:
                x = gt[0] + (px * gt[1]) + (py * gt[2])
                y = gt[3] + (px * gt[4]) + (py * gt[5])
                lon, lat, _ = to_geo.TransformPoint(float(x), float(y))
                while lon > 180.0:
                    lon -= 360.0
                while lon < -180.0:
                    lon += 360.0
                lon_lat.append((lon, lat))

            lons = [p[0] for p in lon_lat]
            lats = [p[1] for p in lon_lat]
            if not lons or not lats:
                return None

            # Detect dateline crossing by large longitudinal jump between neighboring corners.
            max_jump = 0.0
            for idx in range(len(lons)):
                prev = lons[idx - 1]
                curr = lons[idx]
                jump = abs(curr - prev)
                if jump > max_jump:
                    max_jump = jump
            if max_jump < 180.0:
                return None

            pos_lons = [lon for lon in lons if lon >= 0.0]
            neg_lons = [lon for lon in lons if lon < 0.0]
            if not pos_lons or not neg_lons:
                return None

            lat_margin = 0.25
            lon_margin = 0.25
            lat_min = max(-85.0, min(lats) - lat_margin)
            lat_max = min(85.0, max(lats) + lat_margin)
            east_lon_min = max(-180.0, min(pos_lons) - lon_margin)
            west_lon_max = min(180.0, max(neg_lons) + lon_margin)

            lon_windows = [
                (east_lon_min, 180.0),
                (-180.0, west_lon_max),
            ]

            split_bounds: List[Tuple[float, float, float, float]] = []
            for lon_min, lon_max in lon_windows:
                if lon_max <= lon_min:
                    continue
                min_x, min_y, _ = to_merc.TransformPoint(float(lon_min), float(lat_min))
                max_x, max_y, _ = to_merc.TransformPoint(float(lon_max), float(lat_max))
                bounds = (
                    min(min_x, max_x),
                    min(min_y, max_y),
                    max(min_x, max_x),
                    max(min_y, max_y),
                )
                if (bounds[2] - bounds[0]) > 1000 and (bounds[3] - bounds[1]) > 1000:
                    split_bounds.append(bounds)

            return split_bounds if split_bounds else None
        except Exception:
            return None

    def _is_world_width_warp(self, output_raster: Path) -> bool:
        """Detect pathological outputs that balloon to near full-world width in EPSG:3857."""
        try:
            ds = gdal.Open(str(output_raster))
            if not ds:
                return False
            gt = ds.GetGeoTransform(can_return_null=True)
            if gt is None:
                ds = None
                return False
            width_m = abs(float(gt[1])) * float(ds.RasterXSize)
            ds = None
            return width_m >= WORLD_WIDTH_M
        except Exception:
            return False

    def _is_reusable_output(self, path: Path) -> bool:
        """True only if `path` is a fully-readable, non-pathological warp output.

        Used by warp_and_cut to decide whether an existing intermediate can be reused on
        resume instead of re-warping. A corrupt/truncated leftover from an interrupted run
        (bad VRT XML, missing TIFF directory, a VRT whose source is gone, or missing pixel
        tiles) fails gdal.Open or the bottom-strip Checksum probe below and returns False,
        so it gets re-warped rather than silently poisoning the date's mosaic. Near-world-
        width outputs (bad georeferencing) are rejected too.
        """
        ds = None
        try:
            if not (path.exists() and path.stat().st_size > 1024):
                return False
            ds = gdal.Open(str(path))
            if ds is None:
                return False
            gt = ds.GetGeoTransform(can_return_null=True)
            if gt is None:
                return False
            if abs(float(gt[1])) * float(ds.RasterXSize) >= WORLD_WIDTH_M:
                return False
            # Force a read of the last rows so a truncated/partially-written raster
            # (pixel data missing at EOF) raises here instead of at mosaic time.
            strip = min(8, ds.RasterYSize)
            for band_idx in range(1, ds.RasterCount + 1):
                ds.GetRasterBand(band_idx).Checksum(0, ds.RasterYSize - strip, ds.RasterXSize, strip)
            return True
        except Exception:
            return False
        finally:
            ds = None

    def _prepare_warp_job(self, src_file_info: Union[Path, Tuple[Path, Optional[str]]], location: str, edition: str,
                          shp_path: Optional[Path], temp_dir: Path, date: str, record: Optional[Dict] = None) -> List[Dict]:
        """
        Prepare warp job parameters for parallel processing.

        With pre-unzipped files, src_file_info is always a direct Path to a .tif file.
        No ZIP extraction needed.
        """
        norm_loc = self.normalize_name(location)
        jobs = []

        # Unpack the file info (should always be (Path, None) from unzipped structure)
        if isinstance(src_file_info, tuple):
            src_file, internal_file = src_file_info
        else:
            src_file = src_file_info
            internal_file = None

        # With pre-unzipped files, src_file is always a direct TIF path
        tiffs_to_warp = [src_file]

        # Decide intermediate format for warp output (TIFF default, VRT optional)
        warp_out = getattr(self, 'warp_output', 'tif').lower()
        suffix = '.vrt' if warp_out == 'vrt' else '.tif'

        # Create job entries for each TIFF
        for tiff in tiffs_to_warp:
            sanitized_stem = self.sanitize_filename(tiff.stem)
            temp_tif = temp_dir / f"{norm_loc}_{sanitized_stem}{suffix}"
            selected_shp = self._pick_shapefile_for_source(location, tiff, shp_path) if shp_path else None
            jobs.append({
                'input_tiff': tiff,
                'shapefile': selected_shp,
                'output_tiff': temp_tif,
                'location': norm_loc,
                'date': date,
                'record': record,
            })

        return jobs

    def _warp_worker(self, job: Dict) -> Dict:
        """Worker function for parallel warping operations."""
        success = self.warp_and_cut(
            job['input_tiff'],
            job['shapefile'],
            job['output_tiff'],
            record=job.get('record'),
        )

        return {
            'success': success,
            'output_tiff': job['output_tiff'] if success else None,
            'location': job['location'],
            'date': job['date'],
            'attempt_id': job.get('attempt_id')
        }

    def warp_and_cut(self, input_tiff: Path, shapefile: Optional[Path], output_tiff: Path, record: Optional[Dict] = None) -> bool:
        """Warp and cut TIFF using shapefile as cutline.

        If self.warp_output is set to "vrt", write a VRT instead of a TIFF
        to avoid the intermediate write/read of warped GeoTIFFs.
        """
        # Resume: reuse a fully-readable existing output instead of re-warping
        # (see _is_reusable_output). output_tiff already carries the correct
        # .tif/.vrt suffix from _prepare_warp_job.
        if self._is_reusable_output(output_tiff):
            self.log(f"      ↻ Reusing existing warp output {output_tiff.name}")
            return True

        if record is not None:
            context = self._build_gcp_warp_context(record)
            if context:
                if self.warp_and_cut_from_row_gcps(input_tiff, record, output_tiff):
                    return True
                if shapefile:
                    self.log(f"      ⚠ GCP warp failed for {input_tiff.name}; falling back to shapefile warp")
            elif not shapefile:
                self.log(f"      ✗ Missing/invalid GCP metadata and no shapefile fallback for {input_tiff.name}")
                return False

        if not shapefile:
            self.log(f"      ✗ No shapefile available for {input_tiff.name}")
            return False

        try:
            warp_out = getattr(self, 'warp_output', 'tif').lower()
            to_vrt = warp_out == 'vrt'
            self.log(f"    Warping {input_tiff.name} to {'VRT' if to_vrt else 'TIFF'}...")

            # Step 1: Build an RGBA VRT source only for paletted inputs.
            # For non-paletted rasters, warp directly from source TIFF to skip an extra translate pass.
            sanitized_stem = self.sanitize_filename(input_tiff.stem)
            temp_vrt_name = None
            warp_source = str(input_tiff)

            src_ds = gdal.Open(str(input_tiff))
            if not src_ds:
                self.log(f"      ✗ Could not open {input_tiff.name}")
                return False
            first_band = src_ds.GetRasterBand(1)
            has_color_table = bool(first_band and first_band.GetColorTable())
            src_ds = None

            if has_color_table:
                # Keep source VRT on disk only when downstream output is VRT.
                if to_vrt:
                    temp_vrt_path = output_tiff.parent / f"{sanitized_stem}_src.vrt"
                    temp_vrt_name = str(temp_vrt_path)
                else:
                    temp_vrt_name = f"/vsimem/{sanitized_stem}_src.vrt"

                translate_options = gdal.TranslateOptions(format="VRT", rgbExpand="rgba")
                ds_vrt = gdal.Translate(temp_vrt_name, str(input_tiff), options=translate_options)
                if not ds_vrt:
                    self.log(f"      ✗ Failed to prepare RGBA source VRT for {input_tiff.name}")
                    return False
                ds_vrt = None
                warp_source = temp_vrt_name

            # Get the actual SRS of the shapefile (don't hardcode EPSG:4326)
            shapefile_srs = self.get_shapefile_srs(shapefile)
            split_bounds = self._antimeridian_split_bounds_3857(input_tiff)
            if split_bounds:
                self.log(f"      ↺ Antimeridian source detected; splitting warp into {len(split_bounds)} windows")

            # Step 2: Warp with cutline. Write GTiff (default) or VRT when requested
            # Note: Not specifying xRes/yRes allows GDAL to maintain source resolution during warp
            # Don't compress intermediates to reduce CPU load and improve parallel performance
            def _safe_unlink(path: Path) -> None:
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

            def _run_warp(
                transformer_opts,
                output_path: Path,
                output_bounds: Optional[Tuple[float, float, float, float]] = None,
                force_vrt_output: bool = False
            ) -> bool:
                output_format = "VRT" if (to_vrt or force_vrt_output) else "GTiff"
                crop_to_cutline = output_bounds is None
                if output_format == "GTiff":
                    creation_opts = ['TILED=YES', 'BIGTIFF=YES']
                else:
                    creation_opts = None
                _safe_unlink(output_path)
                warp_options = gdal.WarpOptions(
                    format=output_format,
                    dstSRS='EPSG:3857',
                    cutlineDSName=str(shapefile),
                    cutlineSRS=shapefile_srs,
                    cropToCutline=crop_to_cutline,
                    dstAlpha=True,
                    resampleAlg=getattr(self, 'resample_alg', gdal.GRA_Bilinear),
                    creationOptions=creation_opts,
                    multithread=getattr(self, 'warp_multithread', True),
                    transformerOptions=transformer_opts or None,
                    warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
                    outputBounds=output_bounds
                )
                ds_out = gdal.Warp(str(output_path), warp_source, options=warp_options)
                if not ds_out:
                    return False
                ds_out = None
                return output_path.exists() and output_path.stat().st_size > 1024

            def _translate_vrt_to_tiff(src_vrt: Path, dst_tiff: Path) -> bool:
                _safe_unlink(dst_tiff)
                translate_opts = gdal.TranslateOptions(
                    format="GTiff",
                    creationOptions=['TILED=YES', 'BIGTIFF=YES']
                )
                ds_out = gdal.Translate(str(dst_tiff), str(src_vrt), options=translate_opts)
                if not ds_out:
                    return False
                ds_out = None
                return dst_tiff.exists() and dst_tiff.stat().st_size > 1024

            def _raster_pixel_area(path: Path) -> int:
                try:
                    ds_area = gdal.Open(str(path))
                    if not ds_area:
                        return 0
                    area = int(ds_area.RasterXSize) * int(ds_area.RasterYSize)
                    ds_area = None
                    return area
                except Exception:
                    return 0

            def _run_split_warp(transformer_opts, bounds_list: List[Tuple[float, float, float, float]]) -> bool:
                if not bounds_list:
                    return False

                part_vrts: List[Path] = []
                part_items: List[Tuple[Path, Tuple[float, float, float, float]]] = []
                merged_vrt: Optional[Path] = None
                try:
                    for idx, bounds in enumerate(bounds_list):
                        part_vrt = output_tiff.parent / f"{output_tiff.stem}_am_part{idx}.vrt"
                        if _run_warp(transformer_opts, part_vrt, output_bounds=bounds, force_vrt_output=True):
                            part_vrts.append(part_vrt)
                            part_items.append((part_vrt, bounds))
                        else:
                            _safe_unlink(part_vrt)

                    if not part_vrts:
                        return False

                    if len(part_vrts) == 1:
                        if to_vrt:
                            _safe_unlink(output_tiff)
                            part_vrts[0].replace(output_tiff)
                            part_vrts.clear()
                            return output_tiff.exists() and output_tiff.stat().st_size > 0
                        return _translate_vrt_to_tiff(part_vrts[0], output_tiff)

                    # If parts are on opposite sides of the WebMercator antimeridian,
                    # merging them into one VRT yields a full-world canvas. Pick one side.
                    has_pos = any(bounds[0] >= 0.0 for _, bounds in part_items)
                    has_neg = any(bounds[2] <= 0.0 for _, bounds in part_items)
                    if has_pos and has_neg:
                        stem_norm = self.normalize_name(input_tiff.stem)
                        preferred = None
                        if "_east" in stem_norm:
                            preferred = "neg"
                        elif "_west" in stem_norm:
                            preferred = "pos"

                        selected_item: Optional[Tuple[Path, Tuple[float, float, float, float]]] = None
                        if preferred == "neg":
                            neg_items = [item for item in part_items if item[1][2] <= 0.0]
                            if neg_items:
                                selected_item = max(neg_items, key=lambda item: _raster_pixel_area(item[0]))
                        elif preferred == "pos":
                            pos_items = [item for item in part_items if item[1][0] >= 0.0]
                            if pos_items:
                                selected_item = max(pos_items, key=lambda item: _raster_pixel_area(item[0]))

                        if selected_item is None:
                            selected_item = max(part_items, key=lambda item: _raster_pixel_area(item[0]))

                        selected_vrt = selected_item[0]
                        side = "west-hemisphere" if selected_item[1][2] <= 0.0 else "east-hemisphere"
                        self.log(f"      ↺ Antimeridian split spans both world edges; using {side} window for {input_tiff.name}")

                        if to_vrt:
                            _safe_unlink(output_tiff)
                            selected_vrt.replace(output_tiff)
                            part_vrts.remove(selected_vrt)
                            return output_tiff.exists() and output_tiff.stat().st_size > 0
                        return _translate_vrt_to_tiff(selected_vrt, output_tiff)

                    merged_vrt = output_tiff if to_vrt else (output_tiff.parent / f"{output_tiff.stem}_am_merge.vrt")
                    if (not to_vrt) and merged_vrt.exists():
                        _safe_unlink(merged_vrt)

                    merge_ds = gdal.BuildVRT(str(merged_vrt), [str(p) for p in part_vrts])
                    if not merge_ds:
                        return False
                    merge_ds = None

                    if to_vrt:
                        return merged_vrt.exists() and merged_vrt.stat().st_size > 0
                    return _translate_vrt_to_tiff(merged_vrt, output_tiff)
                finally:
                    for part in part_vrts:
                        _safe_unlink(part)
                    if merged_vrt and (not to_vrt):
                        _safe_unlink(merged_vrt)

            def _run_once(transformer_opts) -> bool:
                if split_bounds:
                    return _run_split_warp(transformer_opts, split_bounds)
                return _run_warp(transformer_opts, output_tiff)

            transformer_opts = getattr(self, 'transformer_options', None)
            success = False
            try:
                success = _run_once(transformer_opts)
            except Exception as e:
                if transformer_opts:
                    self.log(f"      ⚠ Warp failed with strict transformer options; retrying with relaxed options...")
                    try:
                        success = _run_once(['ALLOW_BALLPARK=YES'])
                    except Exception:
                        raise e
                else:
                    raise e
            if (not success) and transformer_opts:
                self.log(f"      ⚠ Warp returned no dataset with strict transformer options; retrying with relaxed options...")
                success = _run_once(['ALLOW_BALLPARK=YES'])

            # Safety guard: if output still spans near full-world width, retry with split bounds.
            if success and self._is_world_width_warp(output_tiff):
                self.log("      ⚠ Detected near-world-width warp output; forcing antimeridian split retry...")
                forced_bounds = split_bounds or self._antimeridian_split_bounds_3857(input_tiff)
                if forced_bounds:
                    success = _run_split_warp(['ALLOW_BALLPARK=YES'], forced_bounds)
                if success and self._is_world_width_warp(output_tiff):
                    self.log("      ✗ Warp still produced pathological full-world width after retry")
                    success = False

            # Cleanup
            if temp_vrt_name and (not to_vrt) and temp_vrt_name.startswith("/vsimem/"):
                try:
                    gdal.Unlink(temp_vrt_name)
                except:
                    pass

            if success:
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                size_label = f"{file_size_mb:.1f} MB" if not to_vrt else f"{output_tiff.stat().st_size} bytes"
                self.log(f"      ✓ Created {output_tiff.name} ({size_label})")
                return True

            _safe_unlink(output_tiff)
            self.log(f"      ✗ Warp failed for {input_tiff.name}")
            return False

        except Exception as e:
            try:
                if output_tiff.exists():
                    output_tiff.unlink()
            except Exception:
                pass
            self.log(f"      ✗ Error warping {input_tiff.name}: {e}")
            return False

    def warp_and_cut_from_row_gcps(self, input_tiff: Path, row: Dict, output_tiff: Path) -> bool:
        """Warp and clip using row GCP metadata and inferred bounds."""
        context = self._build_gcp_warp_context(row)
        if not context:
            self.log(f"      ✗ Missing/invalid GCP metadata for {input_tiff.name}")
            return False

        warp_out = getattr(self, 'warp_output', 'tif').lower()
        to_vrt = warp_out == 'vrt'
        self.log(f"    Warping {input_tiff.name} from row GCPs to {'VRT' if to_vrt else 'TIFF'}...")

        sanitized_stem = self.sanitize_filename(input_tiff.stem)
        if to_vrt:
            temp_gcp_vrt = output_tiff.parent / f"{sanitized_stem}_srcgcp.vrt"
        else:
            temp_gcp_vrt = Path(f"/vsimem/{sanitized_stem}_srcgcp.vrt")
        cutline_json = output_tiff.parent / f"{sanitized_stem}_cutline.geojson"

        try:
            gcps = []
            for idx in range(4):
                px, py = context["pixels"][idx]
                mx, my = context["projected"][idx]
                gcps.append(gdal.GCP(float(mx), float(my), 0.0, float(px), float(py)))

            translate_options = gdal.TranslateOptions(
                format="VRT",
                GCPs=gcps,
                outputSRS=str(context["source_crs"])
            )
            ds_vrt = gdal.Translate(str(temp_gcp_vrt), str(input_tiff), options=translate_options)
            if not ds_vrt:
                self.log(f"      ✗ Failed to build GCP VRT for {input_tiff.name}")
                return False
            ds_vrt = None

            with open(cutline_json, "w", encoding="utf-8") as f:
                json.dump({
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [context["cutline_coords"]],
                        }
                    }]
                }, f)

            def _run_warp(transformer_opts):
                warp_options = gdal.WarpOptions(
                    format="VRT" if to_vrt else "GTiff",
                    dstSRS='EPSG:3857',
                    cutlineDSName=str(cutline_json),
                    cutlineSRS='EPSG:4269',
                    cropToCutline=True,
                    dstAlpha=True,
                    resampleAlg=getattr(self, 'resample_alg', gdal.GRA_NearestNeighbour),
                    polynomialOrder=1,
                    creationOptions=None if to_vrt else ['TILED=YES', 'BIGTIFF=YES'],
                    multithread=getattr(self, 'warp_multithread', True),
                    transformerOptions=transformer_opts or None,
                    warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
                )
                return gdal.Warp(str(output_tiff), str(temp_gcp_vrt), options=warp_options)

            transformer_opts = getattr(self, 'transformer_options', None)
            ds = None
            try:
                ds = _run_warp(transformer_opts)
            except Exception:
                if transformer_opts:
                    self.log("      ⚠ Warp failed with strict transformer options; retrying with relaxed options...")
                    ds = _run_warp(['ALLOW_BALLPARK=YES'])
                else:
                    raise

            if ds:
                ds = None
                if output_tiff.exists() and output_tiff.stat().st_size > 1024:
                    if self._is_world_width_warp(output_tiff):
                        self.log(f"      ✗ GCP warp produced pathological full-world width: {output_tiff.name}")
                        return False
                    self.log(f"      ✓ Warp complete: {output_tiff.name}")
                    return True

            self.log(f"      ✗ Warp failed for {input_tiff.name}")
            return False
        except Exception as e:
            self.log(f"      ✗ Error warping {input_tiff.name} from row GCPs: {e}")
            return False
        finally:
            try:
                if str(temp_gcp_vrt).startswith("/vsimem/"):
                    gdal.Unlink(str(temp_gcp_vrt))
                elif (not to_vrt) and Path(temp_gcp_vrt).exists():
                    Path(temp_gcp_vrt).unlink()
            except Exception:
                pass
            try:
                if cutline_json.exists():
                    cutline_json.unlink()
            except Exception:
                pass

    def _make_geotiff_progress_cb(self, start_time: float):
        """Build a GDAL progress callback that logs per-job and overall ETA every 5%.

        Used by create_mosaic_geotiff (Warp) to report progress/ETA.
        """
        progress_state = {'last_pct': -5}

        def progress_cb(complete, message, cb_data):
            pct = int(complete * 100)
            if pct - progress_state['last_pct'] >= 5:
                progress_state['last_pct'] = pct
                elapsed = time.time() - start_time
                # Per-job ETA
                if pct > 0:
                    est_total = elapsed * 100 / pct
                    eta_sec = max(0, est_total - elapsed)
                    eta_min = eta_sec / 60
                    self.log(f"      GeoTIFF progress: {pct}% | ETA {eta_min:.1f} min")
                else:
                    eta_sec = 0
                    self.log(f"      GeoTIFF progress: {pct}%")

                # Overall ETA (based on completed GeoTIFFs)
                # remaining = total_planned - completed (current job counts as in-progress, not completed)
                completed = getattr(self, 'geotiff_completed', 0)
                total_planned = getattr(self, 'geotiff_total_planned', 0)
                remaining_jobs = max(0, total_planned - completed)
                durations = getattr(self, 'geotiff_times', [])
                if durations and remaining_jobs > 0:
                    avg_sec = sum(durations) / len(durations)
                    # Current job remaining + future jobs
                    overall_sec = eta_sec + (remaining_jobs - 1) * avg_sec if pct > 0 else remaining_jobs * avg_sec
                    overall_hours = overall_sec / 3600
                    if overall_hours >= 1:
                        self.log(f"      Overall ETA: {overall_hours:.1f} hours ({remaining_jobs} GeoTIFFs remaining, avg {avg_sec/60:.1f} min each)")
                    else:
                        self.log(f"      Overall ETA: {overall_sec/60:.1f} min ({remaining_jobs} GeoTIFFs remaining)")
            return 1

        return progress_cb

    def create_mosaic_geotiff(self, input_files: List[Path], output_tiff: Path, compress: str = 'LZW') -> Tuple[bool, Optional[float]]:
        """Mosaic the individual warped chart rasters directly into one GeoTIFF.

        Uses gdal.Warp straight from the per-chart VRTs, so there is no
        intermediate mosaic VRT on disk - the charts stay virtual and the
        mosaic is the only materialized artifact. Overlapping charts composite
        via their alpha bands (last opaque source wins), matching the previous
        BuildVRT(resolution='highest') + Translate path.
        """
        try:
            # Validate inputs (skip missing/empty/corrupt-small files)
            valid_files = []
            for f in input_files:
                if not f.exists():
                    self.log(f"      Warning: Input file not found: {f.name}")
                    continue
                file_size = f.stat().st_size
                if file_size < 1024:  # Skip files smaller than 1KB (likely corrupt/empty)
                    self.log(f"      Warning: Skipping {f.name} (too small: {file_size} bytes)")
                    continue
                valid_files.append(f)

            if not valid_files:
                self.log(f"      ✗ No valid input files for mosaic")
                return False, None

            num_threads = getattr(self, 'num_threads', 'ALL_CPUS')
            if compress == 'NONE':
                self.log(f"    Creating mosaic GeoTIFF with no compression ({num_threads} threads)...")
            else:
                self.log(f"    Creating mosaic GeoTIFF with {compress} compression ({num_threads} threads)...")

            # Output resolution: leave xRes/yRes unset so gdal.Warp matches the finest
            # source pixel size on its own (equivalent to BuildVRT resolution='highest').
            # Verified byte-identical to an explicit finest-res pre-scan, so no extra
            # gdal.Open passes are needed here.

            # Configure GDAL for single-process mosaic creation (use all CPUs)
            _configure_gdal_for_phase('geotiff', workers=1)

            creation_opts = [
                'TILED=YES',
                'BIGTIFF=YES',
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                f'NUM_THREADS={num_threads}'
            ]
            if compress != 'NONE':
                creation_opts.extend([
                    f'COMPRESS={compress}',
                    'PREDICTOR=2'
                ])
                if compress == 'ZSTD':
                    creation_opts.append('ZSTD_LEVEL=1')

            # Optional clip window: projWin is ULX ULY LRX LRY -> Warp outputBounds is minX minY maxX maxY
            output_bounds = None
            projwin = getattr(self, 'clip_projwin', None)
            if projwin:
                ulx, uly, lrx, lry = projwin
                output_bounds = (ulx, lry, lrx, uly)
                self.log(f"    Applying clip bounds (minX,minY,maxX,maxY): ({ulx}, {lry}, {lrx}, {uly})")

            start_time = time.time()
            progress_cb = self._make_geotiff_progress_cb(start_time)

            warp_kwargs = dict(
                format='GTiff',
                creationOptions=creation_opts,
                resampleAlg=getattr(self, 'resample_alg', gdal.GRA_NearestNeighbour),
                multithread=True,
                outputBounds=output_bounds,
                callback=progress_cb,
            )

            warp_options = gdal.WarpOptions(**warp_kwargs)
            ds = gdal.Warp(str(output_tiff), [str(f) for f in valid_files], options=warp_options)

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                elapsed = time.time() - start_time
                rate = file_size_mb / elapsed if elapsed > 0 else 0
                self.log(f"      ✓ Mosaic GeoTIFF created: {output_tiff.name} ({file_size_mb:.1f} MB in {elapsed:.1f}s, {rate:.1f} MB/s)")
                return True, elapsed
            else:
                self.log(f"      ✗ Failed to create mosaic GeoTIFF")
                return False, None

        except Exception as e:
            error_str = str(e).lower()
            # If ZSTD compression fails, retry with LZW
            if compress == 'ZSTD' and ('zstd' in error_str or 'frame descriptor' in error_str or 'data corruption' in error_str):
                self.log(f"    ZSTD error detected, retrying mosaic with LZW compression...")
                return self.create_mosaic_geotiff(input_files, output_tiff, compress='LZW')

            self.log(f"    Error creating mosaic GeoTIFF: {e}")
            import traceback
            self.log(f"    {traceback.format_exc()}")
            return False, None

    def _save_review_csv(self, report_data: List[Dict], output_path: Path) -> None:
        """Save review report to CSV file in matrix format (dates × locations)."""
        try:
            import csv as csv_module

            # Build data structures (same as console)
            dates = sorted(set(e['date'] for e in report_data), reverse=True)
            locations = sorted(set(e['location'] for e in report_data))

            # Build lookup: date -> location -> cell content
            matrix = defaultdict(dict)
            for entry in report_data:
                date = entry['date']
                loc = entry['location']

                # Determine cell content based on matches
                if entry['num_files'] > 0:
                    # Extract just filename
                    files = entry['source_files'].split(',')
                    if len(files) > 1:
                        cell = f"({len(files)} files)"
                    else:
                        filename = Path(files[0].strip()).name
                        # Truncate if too long
                        if len(filename) > 12:
                            filename = filename[:9] + "..."
                        cell = filename
                else:
                    cell = "---"

                matrix[date][loc] = cell

            # Write matrix format CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv_module.writer(f)

                # Write header row with location names
                header = ['Date'] + locations
                writer.writerow(header)

                # Write data rows
                for date in dates:
                    row = [date]
                    for loc in locations:
                        cell_value = matrix[date].get(loc, "---")
                        row.append(cell_value)
                    writer.writerow(row)

            self.log(f"Review report saved to: {output_path}")
        except Exception as e:
            self.log(f"Warning: Could not save review CSV: {e}")

    def generate_review_report(self, auto_confirm: bool = False) -> bool:
        """
        Generate review report showing file matches for all dates.
        Saves CSV report and displays formatted summary.
        Returns True if user confirms to proceed, False otherwise.
        When auto_confirm is True, skips the prompt and proceeds.
        """
        self.log("\n=== Generating Processing Review ===")

        # Get all locations from CSV (not shapefiles)
        all_locations = set()
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                if norm_loc:
                    all_locations.add(norm_loc)

        all_locations = sorted(all_locations)

        # Build CSV records lookup: date -> location -> record
        csv_lookup = defaultdict(dict)
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                csv_lookup[date][norm_loc] = rec

        # Build report data for ALL locations on ALL dates
        sorted_dates = sorted(self.dole_data.keys(), key=self._date_key_sort_key, reverse=True)
        report_data = []

        for date in sorted_dates:
            for location in all_locations:
                # Check if this location has a CSV record for this date
                if location in csv_lookup[date]:
                    rec = csv_lookup[date][location]
                    edition = rec.get('edition', '')
                    timestamp = rec.get('wayback_ts')

                    # Find shapefile
                    shp_path = self.find_shapefile(location)
                    shapefile_found = "YES" if shp_path else "NO"

                    # Find source files
                    filename = rec.get('filename', '')
                    download_link = rec.get('download_link', '')
                    found_files = self.find_files(location, edition, timestamp, filename, download_link)

                    if found_files:
                        # Handle both Path and (Path, internal_file) tuple formats
                        file_names = []
                        for f in found_files:
                            if isinstance(f, tuple):
                                path, internal = f
                                if internal:
                                    file_names.append(f"{path.name}/{internal}")
                                else:
                                    file_names.append(path.name)
                            else:
                                file_names.append(f.name)
                        source_files_str = ', '.join(file_names)
                    else:
                        # Check if filename exists in CSV but just not on disk yet
                        if filename:
                            # File is in CSV but not on disk - will be downloaded during processing
                            source_files_str = f"📥 WILL DOWNLOAD: {filename}"
                        else:
                            source_files_str = "NOT IN CSV"

                    report_entry = {
                        'date': date,
                        'location': location,
                        'edition': edition,
                        'shapefile_found': shapefile_found,
                        'source_files': source_files_str,
                        'num_files': len(found_files)
                    }
                else:
                    # No CSV record for this location on this date
                    report_entry = {
                        'date': date,
                        'location': location,
                        'edition': 'N/A',
                        'shapefile_found': 'N/A',
                        'source_files': 'NOT FOUND',
                        'num_files': 0
                    }

                report_data.append(report_entry)

        # Save CSV report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"processing_review_{timestamp}.csv"
        self._save_review_csv(report_data, csv_path)

        # Non-interactive: proceed without prompting
        if auto_confirm:
            self.log("Auto-confirm enabled (--yes): proceeding with processing.")
            return True

        # Get user confirmation
        while True:
            response = input("\nProceed with processing? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _rebuild_vrt_library(self, all_locations: List[str]) -> Dict[str, Dict[str, List[Path]]]:
        """
        Rebuild vrt_library from existing temp VRTs for resume support.

        Scans temp_dir for previously created location VRTs so that
        resuming an interrupted run can reuse existing intermediates.

        Returns:
            Dict mapping location -> date -> list of VRT/TIF paths
        """
        vrt_library: Dict[str, Dict[str, List[Path]]] = defaultdict(dict)

        if not self.temp_dir.exists():
            return vrt_library

        self.log("Rebuilding VRT library from existing temp files...")

        # Scan date subdirectories in temp_dir
        date_dirs = [d for d in self.temp_dir.iterdir() if d.is_dir()]

        found_count = 0
        for date_dir in date_dirs:
            date = date_dir.name

            # Skip non-date directories (simple validation: should contain hyphen)
            if '-' not in date:
                continue

            # Collect per-location warped outputs (TIF/VRT) and fall back to combined VRTs
            combined_vrt_by_loc = {}

            for vrt_file in date_dir.glob("*.vrt"):
                # Skip mosaic VRTs
                if vrt_file.name.startswith("mosaic_"):
                    continue
                # Skip RGBA intermediate VRTs
                if "_rgba.vrt" in vrt_file.name:
                    continue
                # NOTE: world-width (pathological) filtering is deferred to mosaic-input
                # assembly (see process_all_dates) so this rebuild stays pure filesystem
                # globbing with no GDAL opens.

                # Track per-location combined VRTs: {location}_{date}.vrt
                stem = vrt_file.stem
                if stem.endswith(f"_{date}"):
                    location = stem[:-len(f"_{date}")]
                    if location in all_locations or self.normalize_name(location) in all_locations:
                        norm_loc = self.normalize_name(location)
                        combined_vrt_by_loc[norm_loc] = vrt_file
                    continue

                # Treat other VRTs as individual warped outputs: {location}_{original_stem}.vrt
                for loc in all_locations:
                    if stem.startswith(f"{loc}_"):
                        if date not in vrt_library.get(loc, {}):
                            vrt_library[loc][date] = []
                        vrt_library[loc][date].append(vrt_file)
                        found_count += 1
                        break

            # Collect warped TIFs: {location}_{original_stem}.tif
            for tif_file in date_dir.glob("*.tif"):
                stem = tif_file.stem
                for loc in all_locations:
                    if stem.startswith(f"{loc}_"):
                        if date not in vrt_library.get(loc, {}):
                            vrt_library[loc][date] = []
                        vrt_library[loc][date].append(tif_file)
                        found_count += 1
                        break

            # If no individual warped outputs were found, fall back to per-location combined VRTs
            for norm_loc, loc_vrt in combined_vrt_by_loc.items():
                if date in vrt_library.get(norm_loc, {}):
                    continue
                vrt_library[norm_loc][date] = [loc_vrt]
                found_count += 1

        if found_count > 0:
            # Count unique location-date pairs
            total_pairs = sum(len(dates) for dates in vrt_library.values())
            self.log(f"  Restored {total_pairs} location-date entries from {len(vrt_library)} locations")
        else:
            self.log("  No existing VRTs found (fresh run)")

        return vrt_library

    def process_all_dates(self, date_range: Optional[Tuple[datetime.date, datetime.date]] = None):
        """Process date ranges in CSV into mosaicked GeoTIFFs, optionally limited by start date."""
        self.log("\n=== Starting Chart Processing ===")

        # Get all locations from CSV (not shapefiles)
        all_locations = set()
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                if norm_loc:
                    all_locations.add(norm_loc)

        all_locations = sorted(all_locations)

        # Count how many have shapefiles
        shapefile_count = sum(1 for loc in all_locations if self.find_shapefile(loc))
        self.log(f"Found {len(all_locations)} locations from CSV ({shapefile_count} with shapefiles).")

        # Build processing date list (optionally filtered)
        sorted_dates = sorted(self.dole_data.keys(), key=self._date_key_sort_key)
        if date_range:
            start_date, end_date = date_range
            if start_date and end_date and start_date > end_date:
                start_date, end_date = end_date, start_date
            filtered = []
            for date_key in sorted_dates:
                start_str = self.date_key_map.get(date_key, (date_key, None))[0]
                date_obj = self._date_from_str(start_str)
                if not date_obj:
                    continue
                if start_date <= date_obj <= end_date:
                    filtered.append(date_key)
            sorted_dates = filtered

            if not sorted_dates:
                self.log("No dates found in the requested range. Nothing to process.")
                return

            self.log(f"Limiting processing to {len(sorted_dates)} date ranges ({start_date} to {end_date})")

        # === PREPARATION PHASE: Download all missing files upfront ===
        self.log("\n=== Preparing: Downloading missing files ===")
        self._download_missing_files(allowed_dates=set(sorted_dates))

        # Track all warped VRTs per location-date
        # Rebuild from existing temp files to support resume
        vrt_library = self._rebuild_vrt_library(all_locations)

        # Process each date range (earliest to latest)
        total_dates = len(sorted_dates)

        # GeoTIFF progress/ETA tracking (read by the geotiff progress callback)
        self.geotiff_total_planned = total_dates
        self.geotiff_completed = 0
        self.geotiff_times = []

        for date_idx, date in enumerate(sorted_dates, 1):
            self.log(f"\n[{date_idx}/{total_dates}] Processing range: {date}")

            temp_dir = self.temp_dir / date
            temp_dir.mkdir(parents=True, exist_ok=True)

            output_tiff = self.output_dir / f"{date}.tif"

            # Resume: skip dates whose final mosaic GeoTIFF already exists
            if output_tiff.exists() and output_tiff.stat().st_size > 0:
                self.log(f"  ✓ GeoTIFF already exists - skipping {output_tiff.name}")
                self.geotiff_completed += 1
                continue

            records = self.dole_data[date]

            # Group records by candidate_group so early/master overlap rows are alternatives.
            record_groups = []
            record_group_map = {}
            for rec in records:
                location = (rec.get('location') or '').strip()
                edition = (rec.get('edition') or '').strip()
                candidate_group = (rec.get('candidate_group') or '').strip()
                if not candidate_group:
                    norm_loc = self.normalize_name(location)
                    candidate_group = f"{date}|{norm_loc}" if norm_loc else f"{date}|row"
                    rec['candidate_group'] = candidate_group
                key = candidate_group
                if key not in record_group_map:
                    record_group_map[key] = {
                        'location': location,
                        'edition': edition,
                        'candidate_group': candidate_group,
                        'records': []
                    }
                    record_groups.append(record_group_map[key])
                record_group_map[key]['records'].append(rec)

            # Track warped outputs selected per location for this date
            temp_warped_by_location = defaultdict(list)

            # Prepare group state for fallback attempts
            group_states = []
            for group in record_groups:
                group['records'].sort(
                    key=lambda r: int(str(r.get('candidate_rank', '')).strip())
                    if str(r.get('candidate_rank', '')).strip().isdigit() else 9999
                )
                norm_loc = self.normalize_name(group['location'])
                group_states.append({
                    'location': group['location'],
                    'edition': group['edition'],
                    'candidate_group': group['candidate_group'],
                    'norm_loc': norm_loc,
                    'records': group['records'],
                    'next_index': 0,
                    'done': False
                })

            # Execute warp jobs in parallel, with fallback to duplicate rows on failure
            if group_states:
                requested_warp_workers = getattr(self, 'parallel_warp', 0)
                if requested_warp_workers and requested_warp_workers > 0:
                    warp_worker_upper_bound = max(1, min(requested_warp_workers, cpu_count))
                else:
                    warp_worker_upper_bound = max(2, cpu_count - 2)
                    if IS_APPLE_SILICON:
                        # Keep some headroom for system + upload threads on M-series SoCs.
                        warp_worker_upper_bound = min(warp_worker_upper_bound, 6)

                warp_worker_target = warp_worker_upper_bound
                warp_round_rate_ema = None
                warp_mem_per_worker_mb = 1200 if getattr(self, 'warp_output', 'tif').lower() == 'vrt' else 900
                if getattr(self, 'warp_multithread', True):
                    warp_mem_per_worker_mb += 256

                round_idx = 0
                while True:
                    warp_jobs = []
                    attempt_map = {}
                    made_progress = False

                    for state in group_states:
                        if state['done'] or state['next_index'] >= len(state['records']):
                            continue

                        rec = state['records'][state['next_index']]
                        location = (rec.get('location') or '').strip()
                        edition = (rec.get('edition') or '').strip()
                        timestamp = rec.get('wayback_ts')
                        filename = rec.get('filename', '')  # Get filename from CSV

                        attempt_num = state['next_index'] + 1
                        attempt_total = len(state['records'])
                        attempt_label = f" (option {attempt_num}/{attempt_total})" if attempt_total > 1 else ""

                        # Prefer per-row GCP context when available; fallback to shapefile warp.
                        gcp_context = self._build_gcp_warp_context(rec)
                        shp_path = None
                        if not gcp_context:
                            shp_path = self.find_shapefile(location)
                            if not shp_path:
                                self.log(
                                    f"    {location} (edition {edition}){attempt_label}: "
                                    "no valid GCP context and no shapefile fallback"
                                )
                                state['done'] = True
                                made_progress = True
                                continue

                        # Find source files (use filename from CSV if available)
                        download_link = rec.get('download_link', '')
                        found_files = self.find_files(location, edition, timestamp, filename, download_link)
                        if not found_files:
                            self.log(
                                f"    {location} (edition {edition}){attempt_label}: No source files found [CSV: {filename if filename else 'no match'}]"
                            )
                            state['next_index'] += 1
                            made_progress = True
                            if state['next_index'] < len(state['records']):
                                self.log(f"    {location} (edition {edition}): trying next duplicate option")
                            continue

                        self.log(f"    {location} (edition {edition}){attempt_label}: {filename if filename else 'pattern match'}")

                        attempt_id = (state['candidate_group'], state['next_index'])
                        attempt_map[attempt_id] = state

                        # Prepare warp jobs for this record attempt
                        for src_file_info in found_files:
                            jobs = self._prepare_warp_job(
                                src_file_info,
                                location,
                                edition,
                                shp_path,
                                temp_dir,
                                date,
                                record=rec,
                            )
                            for job in jobs:
                                job['attempt_id'] = attempt_id
                            warp_jobs.extend(jobs)

                    if not warp_jobs:
                        if made_progress:
                            continue
                        break

                    round_idx += 1
                    warp_mem_snapshot = _get_memory_snapshot_mb(
                        default_available_mb=8192 if IS_APPLE_SILICON else 4096,
                        default_total_mb=16384 if IS_APPLE_SILICON else 8192
                    )
                    warp_reserve_mb = _recommended_free_ram_floor_mb(warp_mem_snapshot["total_mb"])
                    warp_mem_budget_mb = max(0, warp_mem_snapshot["available_mb"] - warp_reserve_mb)
                    warp_memory_cap = max(
                        1,
                        min(
                            warp_worker_upper_bound,
                            (warp_mem_budget_mb // warp_mem_per_worker_mb) if warp_mem_budget_mb > 0 else 1
                        )
                    )
                    current_warp_workers = max(
                        1,
                        min(warp_worker_target, warp_memory_cap, len(warp_jobs))
                    )
                    if current_warp_workers != warp_worker_target:
                        self.log(
                            f"  ↺ Autotune warp: workers {warp_worker_target}->{current_warp_workers} "
                            f"(free={warp_mem_snapshot['available_mb']}MB, floor={warp_reserve_mb}MB)"
                        )
                    warp_worker_target = current_warp_workers
                    _configure_gdal_for_phase('warp', workers=current_warp_workers)
                    self.log(
                        f"  Processing {len(warp_jobs)} warp jobs with {current_warp_workers} parallel workers "
                        f"(round {round_idx}, free RAM {warp_mem_snapshot['available_mb']}MB)..."
                    )
                    round_start = time.time()
                    results = []

                    with ThreadPoolExecutor(max_workers=current_warp_workers) as executor:
                        futures = {executor.submit(self._warp_worker, job): job for job in warp_jobs}

                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                results.append(result)
                            except Exception as e:
                                self.log(f"    Error in parallel warp: {e}")

                    round_elapsed = max(0.001, time.time() - round_start)
                    round_successes = sum(1 for result in results if result.get('success'))
                    round_rate = round_successes / round_elapsed

                    if warp_round_rate_ema is None:
                        warp_round_rate_ema = round_rate
                    else:
                        if round_successes == 0 and warp_worker_target > 1:
                            next_workers = max(1, warp_worker_target - 1)
                            self.log(
                                f"  ↺ Autotune warp: workers {warp_worker_target}->{next_workers} "
                                f"(no successes in round {round_idx})"
                            )
                            warp_worker_target = next_workers
                        elif round_rate < (warp_round_rate_ema * 0.75) and warp_worker_target > 1:
                            next_workers = max(1, warp_worker_target - 1)
                            self.log(
                                f"  ↺ Autotune warp: workers {warp_worker_target}->{next_workers} "
                                f"(throughput drop to {round_rate:.3f} jobs/s)"
                            )
                            warp_worker_target = next_workers
                        elif (
                            round_rate >= (warp_round_rate_ema * 1.05)
                            and warp_worker_target < warp_worker_upper_bound
                            and len(warp_jobs) > warp_worker_target
                        ):
                            projected_workers = warp_worker_target + 1
                            projected_need_mb = warp_reserve_mb + (projected_workers * warp_mem_per_worker_mb)
                            if warp_mem_snapshot["available_mb"] >= projected_need_mb:
                                self.log(
                                    f"  ↺ Autotune warp: workers {warp_worker_target}->{projected_workers} "
                                    f"(throughput stable at {round_rate:.3f} jobs/s)"
                                )
                                warp_worker_target = projected_workers

                        warp_round_rate_ema = (warp_round_rate_ema * 0.7) + (round_rate * 0.3)

                    # Group results by attempt (validate output files)
                    outputs_by_attempt = defaultdict(list)
                    for result in results:
                        if result['success'] and result['output_tiff']:
                            output_file = result['output_tiff']
                            if output_file.exists() and output_file.stat().st_size > 1024:
                                if self._is_world_width_warp(output_file):
                                    self.log(f"      Warning: Skipping pathological full-world warp: {output_file.name}")
                                    continue
                                outputs_by_attempt[result.get('attempt_id')].append(output_file)
                            else:
                                self.log(f"      Warning: Skipping invalid warped file: {output_file.name}")

                    # Apply results and schedule fallback attempts as needed
                    for attempt_id, state in attempt_map.items():
                        attempt_outputs = outputs_by_attempt.get(attempt_id, [])
                        if attempt_outputs:
                            temp_warped_by_location[state['norm_loc']].extend(attempt_outputs)
                            state['done'] = True
                        else:
                            state['next_index'] += 1
                            made_progress = True
                            if state['next_index'] < len(state['records']):
                                self.log(
                                    f"    {state['location']} (edition {state['edition']}): warp failed; trying next duplicate option"
                                )

                # Register warped outputs directly for mosaic (no per-location VRT combining)
                location_count = 0
                for norm_loc, temp_warped in temp_warped_by_location.items():
                    if not temp_warped:
                        continue
                    location_count += 1
                    if len(temp_warped) > 1:
                        self.log(f"      Using {len(temp_warped)} warped files for {norm_loc}")
                    else:
                        self.log(f"      Single file, using directly")

                    existing = vrt_library.get(norm_loc, {}).get(date, [])
                    if isinstance(existing, list):
                        combined = list(existing)
                    elif existing:
                        combined = [existing]
                    else:
                        combined = []

                    combined.extend(temp_warped)

                    # De-duplicate while preserving order
                    seen = set()
                    deduped = []
                    for p in combined:
                        key = str(p)
                        if key in seen:
                            continue
                        seen.add(key)
                        deduped.append(p)

                    vrt_library[norm_loc][date] = deduped

                if temp_warped_by_location:
                    self.log(f"  Processed {location_count} locations with data")
                else:
                    self.log(f"  No valid locations found with source files")
            else:
                self.log(f"  No valid locations found with source files")

            # Build mosaic inputs from exact date range matches only.
            # Filter pathological (near-world-width) sources here: freshly warped
            # outputs are already screened at warp time (see outputs_by_attempt),
            # so in practice this only opens genuinely orphaned rebuilt temps whose
            # source file could no longer be found.
            mosaic_sources = []
            for location in all_locations:
                entries = vrt_library.get(location, {}).get(date)
                if not entries:
                    continue
                candidates = entries if isinstance(entries, list) else [entries]
                for entry in candidates:
                    if self._is_world_width_warp(entry):
                        self.log(f"      Warning: Skipping pathological full-world source: {entry.name}")
                        continue
                    mosaic_sources.append(entry)

            # Mosaic the per-chart VRTs directly into the final GeoTIFF (no mosaic VRT)
            if mosaic_sources:
                self.log(f"  Mosaicking {len(mosaic_sources)} sources -> {output_tiff.name}...")

                ok, elapsed = self.create_mosaic_geotiff(mosaic_sources, output_tiff, compress=self.geotiff_compress)
                if ok:
                    self.geotiff_completed += 1
                    if elapsed:
                        self.geotiff_times.append(elapsed)
                    self.log(f"  ✓✓✓ {date.upper()} GEOTIFF COMPLETE")
                else:
                    self.log(f"  ✗ Mosaic GeoTIFF creation failed")
            else:
                self.log(f"  Skipping {date} - no data available")

        self.log("\n=== Processing Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="FAA Chart Slicer - Warp, cut, and mosaic sectional charts into GeoTIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s -s /path/to/charts -o /path/to/output -c master_dole_combined.csv -b shapefiles
        """
    )

    parser.add_argument(
        "-s", "--source",
        type=Path,
        default=Path("/Volumes/projects/rawtiffs"),
        help="Source directory containing TIFF/ZIP files (default: /Volumes/projects/rawtiffs)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("/Volumes/drive/sync"),
        help="Output directory for mosaicked GeoTIFFs (one {date}.tif per date) (default: /Volumes/drive/sync)"
    )

    parser.add_argument(
        "-c", "--csv",
        type=Path,
        default=Path("/Users/ryanhemenway/archive.aero/master_dole_scanned.csv"),
        help="Master Dole CSV file (default: /Users/ryanhemenway/archive.aero/master_dole_scanned.csv)"
    )

    parser.add_argument(
        "-b", "--shapefiles",
        type=Path,
        default=Path("/Users/ryanhemenway/archive.aero/shapefiles"),
        help="Directory containing shapefiles (default: /Users/ryanhemenway/archive.aero/shapefiles)"
    )

    parser.add_argument(
        "-t", "--temp-dir",
        type=Path,
        default=Path("/Volumes/projects/.temp"),
        help="Temporary directory for intermediate files (default: /Volumes/projects/.temp)"
    )

    parser.add_argument(
        "-r", "--resample",
        type=str,
        choices=['nearest', 'bilinear', 'cubic', 'cubicspline'],
        default='nearest',
        help="Resampling algorithm for warping (default: nearest). Use 'nearest' for faster processing with charts"
    )

    parser.add_argument(
        "--warp-output",
        type=str,
        choices=['tif', 'vrt'],
        default='vrt',
        help="Intermediate warp output format: 'tif' writes cutlines to disk, 'vrt' (default) keeps them virtual to reduce I/O"
    )

    parser.add_argument(
        "--clip-projwin",
        type=float,
        nargs=4,
        metavar=('ULX', 'ULY', 'LRX', 'LRY'),
        default=None,
        help="Optional projWin window (in target CRS) applied when writing the GeoTIFF. "
             "Order: ULX ULY LRX LRY. Example for CONUS approx: --clip-projwin -13914936 6446276 -7347086 2753408"
    )

    parser.add_argument(
        "--compression",
        type=str,
        choices=['AUTO', 'ZSTD', 'LZW', 'DEFLATE', 'NONE'],
        default='LZW',
        help="GeoTIFF compression (default: LZW). ZSTD is fast lossless at ZSTD_LEVEL=1. "
             "AUTO uses ZSTD when available, else LZW."
    )

    parser.add_argument(
        "--num-threads",
        type=str,
        default='ALL_CPUS',
        help="GTiff NUM_THREADS creation option for GeoTIFF writing (default: ALL_CPUS)"
    )

    parser.add_argument(
        "--parallel-warp",
        type=int,
        default=0,
        help="Number of parallel warp jobs (default: auto; Apple Silicon auto mode caps at 6 for better perf/watt)"
    )

    parser.add_argument(
        "--warp-multithread",
        action="store_true",
        help="Enable GDAL internal multithreading inside each warp job (off by default on Apple Silicon)"
    )

    parser.add_argument(
        "--download-delay",
        type=float,
        default=8.0,
        help="Seconds to wait between downloads to avoid rate limiting (default: 8.0, try 12.0-20.0 if hitting connection errors)"
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the 'Proceed with processing?' prompt (non-interactive)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all dates without prompting (non-interactive; mutually exclusive with --start-date/--end-date)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start of date range to process (use with --end-date for a one-line non-interactive run)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End of date range to process (use with --start-date)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.source.exists():
        print(f"ERROR: Source directory not found: {args.source}")
        sys.exit(1)

    if not args.csv.exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    if not args.shapefiles.exists():
        print(f"ERROR: Shapefiles directory not found: {args.shapefiles}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # Process
    slicer = ChartSlicer(args.source, args.output, args.csv, args.shapefiles,
                         preview_mode=False, download_delay=args.download_delay,
                         temp_dir=args.temp_dir)
    slicer.warp_output = args.warp_output
    slicer.clip_projwin = tuple(args.clip_projwin) if args.clip_projwin else None
    slicer.parallel_warp = args.parallel_warp
    if args.warp_multithread:
        slicer.warp_multithread = True

    # GeoTIFF output settings
    slicer.num_threads = args.num_threads
    geotiff_compress = (args.compression or 'NONE').strip().upper()
    if geotiff_compress == 'AUTO':
        geotiff_compress = 'ZSTD' if ChartSlicer._check_zstd_support() else 'LZW'
    slicer.geotiff_compress = geotiff_compress

    # Map resampling algorithm
    resample_map = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline
    }
    slicer.resample_alg = resample_map[args.resample]

    slicer.load_dole_data()

    # Build indexes for O(1) lookups (after loading CSV data)
    slicer._build_source_index()
    slicer._build_shapefile_index()

    # During review phase, disable downloads (don't try to validate by downloading)
    # This prevents rate limiting during the review report generation
    slicer.preview_mode = True

    # Validate non-interactive date-range args early (before the slow review pass)
    if args.all and (args.start_date or args.end_date):
        print("ERROR: --all cannot be combined with --start-date/--end-date.")
        sys.exit(1)
    if bool(args.start_date) != bool(args.end_date):
        print("ERROR: --start-date and --end-date must be provided together.")
        sys.exit(1)

    cli_date_range = None  # None = unresolved; sentinel handled below
    if args.start_date and args.end_date:
        start_dt = slicer._date_from_str(args.start_date)
        end_dt = slicer._date_from_str(args.end_date)
        if not start_dt or not end_dt:
            print("ERROR: --start-date/--end-date must be valid YYYY-MM-DD dates.")
            sys.exit(1)
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
            print(f"Swapped range to {start_dt} - {end_dt}")
        cli_date_range = (start_dt, end_dt)

    # Generate review report and get user confirmation (without downloading)
    if not slicer.generate_review_report(auto_confirm=args.yes):
        print("Processing cancelled by user.")
        sys.exit(0)

    def _prompt_date_range() -> Optional[Tuple[datetime.date, datetime.date]]:
        while True:
            choice = input("Process all dates or a range? (all/range): ").strip().lower()
            if choice in {"all", "a", ""}:
                return None
            if choice in {"range", "r"}:
                start_str = input("Start date (YYYY-MM-DD): ").strip()
                end_str = input("End date (YYYY-MM-DD): ").strip()
                start_dt = slicer._date_from_str(start_str)
                end_dt = slicer._date_from_str(end_str)
                if not start_dt or not end_dt:
                    print("Please enter valid dates in YYYY-MM-DD format.")
                    continue
                if start_dt > end_dt:
                    start_dt, end_dt = end_dt, start_dt
                    print(f"Swapped range to {start_dt} - {end_dt}")
                return (start_dt, end_dt)
            print("Please enter 'all' or 'range'.")

    # Now enable downloads for actual processing
    slicer.preview_mode = False
    # Resolve date range non-interactively when CLI args supplied; else prompt.
    if args.all:
        date_range = None
    elif cli_date_range is not None:
        date_range = cli_date_range
    else:
        date_range = _prompt_date_range()
    slicer.process_all_dates(date_range=date_range)


if __name__ == '__main__':
    main()
