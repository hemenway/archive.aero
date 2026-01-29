#!/usr/bin/env python3
"""
FAA Chart Newslicer - CLI tool for processing sectional charts
Creates COGs for each date with fallback logic for 56 locations.
"""

import os
import sys
import csv
import re
import argparse
import zipfile
import requests
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL is required but not installed.")
    print("Install with: pip install gdal --break-system-packages")
    sys.exit(1)

import multiprocessing
cpu_count = multiprocessing.cpu_count()

# Set minimal GDAL global defaults (defer thread decisions to phase-specific config)
try:
    import psutil
    available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
    cache_size_mb = max(512, min(8192, available_ram_mb * 3 // 4))  # 75% of RAM, max 8GB
except ImportError:
    cache_size_mb = 2048

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
        try:
            import psutil
            available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
        except ImportError:
            available_ram_mb = 4096

    if phase == 'warp':
        # Warp phase: single-threaded per worker to prevent oversubscription
        gdal.SetConfigOption('GDAL_NUM_THREADS', '1')
        cache_per_worker = max(256, available_ram_mb // workers)
        gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_per_worker))

    elif phase == 'geotiff':
        # GeoTIFF phase: single-process translate with full CPU
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
        gdal.SetConfigOption('GDAL_CACHEMAX', str(max(512, min(8192, available_ram_mb * 3 // 4))))


def create_geotiff_worker(args):
    """Worker for parallel GeoTIFF creation."""
    try:
        input_vrt, output_tiff, date, compress, num_threads, clip_projwin, cache_size_mb = args
    except ValueError:
        input_vrt, output_tiff, date, compress, num_threads, clip_projwin = args
        cache_size_mb = None

    start_time = time.time()
    output_tiff = Path(output_tiff)

    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except Exception as e:
        return {
            'success': False,
            'date': date,
            'output_tiff': str(output_tiff),
            'error': f"GDAL import failed: {e}"
        }

    try:
        if cache_size_mb:
            gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
        if num_threads:
            gdal.SetConfigOption('GDAL_NUM_THREADS', str(num_threads))
    except Exception:
        pass

    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    attempted_lzw = False
    current_compress = compress

    while True:
        try:
            creation_opts = [
                'TILED=YES',
                'BIGTIFF=YES',
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                f'NUM_THREADS={num_threads}'
            ]

            if current_compress != 'NONE':
                creation_opts.extend([
                    f'COMPRESS={current_compress}',
                    'PREDICTOR=2'
                ])

            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=creation_opts,
                projWin=clip_projwin if clip_projwin else None
            )

            ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                elapsed = time.time() - start_time
                rate = file_size_mb / elapsed if elapsed > 0 else 0
                return {
                    'success': True,
                    'date': date,
                    'output_tiff': str(output_tiff),
                    'elapsed': elapsed,
                    'file_size_mb': file_size_mb,
                    'rate': rate,
                    'compress': current_compress
                }

            return {
                'success': False,
                'date': date,
                'output_tiff': str(output_tiff),
                'error': 'gdal.Translate returned None'
            }

        except Exception as e:
            error_str = str(e).lower()
            if current_compress == 'ZSTD' and not attempted_lzw and (
                'zstd' in error_str or 'frame descriptor' in error_str or 'data corruption' in error_str
            ):
                attempted_lzw = True
                current_compress = 'LZW'
                try:
                    if output_tiff.exists():
                        output_tiff.unlink()
                except Exception:
                    pass
                continue

            import traceback
            return {
                'success': False,
                'date': date,
                'output_tiff': str(output_tiff),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'compress': current_compress
            }


def postprocess_tif_worker(args):
    """Convert GeoTIFF to MBTiles/PMTiles and upload PMTiles via rclone."""
    input_tiff, output_dir, remote, rclone_flags = args
    input_tiff = Path(input_tiff)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_tiff.stem
    mbtiles_path = output_dir / f"{stem}.mbtiles"
    pmtiles_path = output_dir / f"{stem}.pmtiles"

    def run(cmd):
        result = subprocess.run(cmd)
        return result.returncode == 0, result.returncode

    if not (pmtiles_path.exists() and pmtiles_path.stat().st_size > 0):
        ok, rc = run(["gdal_translate", "-of", "MBTILES", str(input_tiff), str(mbtiles_path)])
        if not ok:
            return {
                'success': False,
                'input_tiff': str(input_tiff),
                'error': f"gdal_translate failed (exit {rc})"
            }

        ok, rc = run([
            "gdaladdo", "-r", "bilinear", str(mbtiles_path),
            "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"
        ])
        if not ok:
            return {
                'success': False,
                'input_tiff': str(input_tiff),
                'error': f"gdaladdo failed (exit {rc})"
            }

        ok, rc = run(["pmtiles", "convert", str(mbtiles_path), str(pmtiles_path)])
        if not ok:
            return {
                'success': False,
                'input_tiff': str(input_tiff),
                'error': f"pmtiles convert failed (exit {rc})"
            }

    if not (pmtiles_path.exists() and pmtiles_path.stat().st_size > 0):
        return {
            'success': False,
            'input_tiff': str(input_tiff),
            'error': "pmtiles output missing"
        }

    upload_cmd = ["rclone", "copyto", str(pmtiles_path), f"{remote}/{pmtiles_path.name}"] + list(rclone_flags)
    ok, rc = run(upload_cmd)
    if not ok:
        return {
            'success': False,
            'input_tiff': str(input_tiff),
            'error': f"rclone upload failed (exit {rc})"
        }

    return {
        'success': True,
        'input_tiff': str(input_tiff),
        'pmtiles': str(pmtiles_path)
    }


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
        self.temp_dir = temp_dir if temp_dir else output_dir / ".temp"  # Default to output_dir/.temp for backward compatibility
        self.dole_data = defaultdict(list)
        self.srs_cache = {}  # Cache for shapefile SRS lookups
        self.preview_mode = preview_mode  # If True, only report what would be downloaded
        self.download_delay = download_delay  # Seconds to wait between downloads (to avoid rate limiting)
        self.last_download_time = 0  # Track last download time for throttling
        self.fallback_window_days = 180  # Max age for fallback chart reuse
        self._date_cache: Dict[str, Optional[datetime.date]] = {}
        self.upload_root = Path("/Volumes/drive/upload")
        self.upload_remote = "r2:charts/sectionals"
        self.upload_jobs = 6
        self.rclone_flags = [
            "-P",
            "--s3-upload-concurrency", "16",
            "--s3-chunk-size", "128M",
            "--buffer-size", "128M",
            "--s3-disable-checksum",
            "--stats", "1s"
        ]
        self.abbreviations = {
            "hawaiian_is": "hawaiian_islands",
            "dallas_ft_worth": "dallas_ft_worth",
            "mariana_islands_inset": "mariana_islands",
        }

        # File index caches for O(1) lookups
        self.files_by_name: Dict[str, Path] = {}  # Direct TIFs and ZIPs
        self.tifs_by_dir: Dict[str, List[Path]] = {}  # Extracted folder -> TIF list
        self.shapefile_index: Dict[str, Path] = {}  # Normalized location -> shapefile path

        # Determine compression method based on GDAL support
        self.compression = 'ZSTD' if self._check_zstd_support() else 'LZW'
        if self.compression == 'LZW':
            # This will be logged when log() is called for the first time
            self._compression_warning = True

        # GeoTIFF ETA tracking
        self.geotiff_total_planned = 0
        self.geotiff_completed = 0
        self.geotiff_times = []

    def log(self, msg: str):
        """Print log message with timestamp."""
        # Show compression warning on first log call
        if hasattr(self, '_compression_warning') and self._compression_warning:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] WARNING: ZSTD not available, falling back to LZW compression")
            self._compression_warning = False

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def normalize_name(self, name: str) -> str:
        """Normalize chart name consistently."""
        norm = name.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_')
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

    def _run_postprocess_upload_stage(self, input_dir: Path) -> None:
        """Convert GeoTIFF mosaics to MBTiles/PMTiles and upload PMTiles."""
        output_dir = self.upload_root
        output_dir.mkdir(parents=True, exist_ok=True)

        tiff_files = sorted(input_dir.glob("*.tif"))
        jobs = [tiff for tiff in tiff_files if tiff.exists() and tiff.stat().st_size > 1024 * 1024]
        if not jobs:
            self.log(f"  ⊘ No GeoTIFFs found in {input_dir}")
            return

        workers = max(1, min(self.upload_jobs, len(jobs)))
        self.log(f"\n=== Post-Processing: TIFF → MBTiles → PMTiles → Upload ===")
        self.log(f"Input: {input_dir}")
        self.log(f"Output: {output_dir}")
        self.log(f"Remote: {self.upload_remote}")
        self.log(f"Workers: {workers}")

        successes = 0
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    postprocess_tif_worker,
                    (tiff, output_dir, self.upload_remote, self.rclone_flags)
                ): tiff for tiff in jobs
            }
            for future in as_completed(futures):
                tiff = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    failures += 1
                    self.log(f"  ✗ Post-process failed for {tiff.name}: {e}")
                    continue

                if result.get('success'):
                    successes += 1
                    self.log(f"  ✓ Uploaded: {Path(result.get('pmtiles', '')).name}")
                else:
                    failures += 1
                    self.log(f"  ✗ Failed: {tiff.name} - {result.get('error', 'unknown error')}")

        self.log(f"Post-processing complete: {successes} succeeded, {failures} failed")


    def load_dole_data(self):
        """Load CSV data."""
        self.log(f"Loading CSV: {self.csv_file.name}...")
        self.dole_data = defaultdict(list)

        with open(self.csv_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get('date')
                if not date:
                    continue

                link = row.get('download_link', '')
                timestamp = None
                if 'web.archive.org/web/' in link:
                    match = re.search(r'/web/(\d{14})/', link)
                    if match:
                        timestamp = match.group(1)
                row['wayback_ts'] = timestamp
                self.dole_data[date].append(row)

        self.log(f"Loaded {len(self.dole_data)} dates.")

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
                import shutil
                shutil.rmtree(extract_dir)
            return False

    def _download_missing_files(self):
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
        missing_files.sort(key=lambda x: x[3], reverse=True)

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

    def _prepare_warp_job(self, src_file_info: Union[Path, Tuple[Path, Optional[str]]], location: str, edition: str,
                          shp_path: Path, temp_dir: Path, date: str) -> List[Dict]:
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
            jobs.append({
                'input_tiff': tiff,
                'shapefile': shp_path,
                'output_tiff': temp_tif,
                'location': norm_loc,
                'date': date
            })

        return jobs

    def _warp_worker(self, job: Dict) -> Dict:
        """Worker function for parallel warping operations."""
        # Configure GDAL for warp phase (single-threaded per worker)
        _configure_gdal_for_phase('warp', workers=1)

        success = self.warp_and_cut(
            job['input_tiff'],
            job['shapefile'],
            job['output_tiff']
        )

        return {
            'success': success,
            'output_tiff': job['output_tiff'] if success else None,
            'location': job['location'],
            'date': job['date']
        }

    def warp_and_cut(self, input_tiff: Path, shapefile: Path, output_tiff: Path) -> bool:
        """Warp and cut TIFF using shapefile as cutline.

        If self.warp_output is set to "vrt", write a VRT instead of a TIFF
        to avoid the intermediate write/read of warped GeoTIFFs.
        """
        try:
            warp_out = getattr(self, 'warp_output', 'tif').lower()
            to_vrt = warp_out == 'vrt'
            self.log(f"    Warping {input_tiff.name} to {'VRT' if to_vrt else 'TIFF'}...")

            # Step 1: Build a VRT source for warping.
            # Only use rgbExpand when the source is paletted; otherwise GDAL errors.
            sanitized_stem = self.sanitize_filename(input_tiff.stem)
            # Keep the source VRT on disk when outputting VRT so downstream VRTs
            # can reference a persistent source (vsimem would disappear).
            if to_vrt:
                temp_vrt_path = output_tiff.parent / f"{sanitized_stem}_src.vrt"
                temp_vrt_name = str(temp_vrt_path)
            else:
                temp_vrt_name = f"/vsimem/{sanitized_stem}_src.vrt"

            src_ds = gdal.Open(str(input_tiff))
            if not src_ds:
                self.log(f"      ✗ Could not open {input_tiff.name}")
                return False
            first_band = src_ds.GetRasterBand(1)
            has_color_table = bool(first_band and first_band.GetColorTable())
            src_ds = None

            if has_color_table:
                translate_options = gdal.TranslateOptions(format="VRT", rgbExpand="rgba")
            else:
                translate_options = gdal.TranslateOptions(format="VRT")

            ds_vrt = gdal.Translate(temp_vrt_name, str(input_tiff), options=translate_options)
            ds_vrt = None

            # Get the actual SRS of the shapefile (don't hardcode EPSG:4326)
            shapefile_srs = self.get_shapefile_srs(shapefile)

            # Step 2: Warp with cutline. Write GTiff (default) or VRT when requested
            # Note: Not specifying xRes/yRes allows GDAL to maintain source resolution during warp
            # Don't compress intermediates to reduce CPU load and improve parallel performance
            warp_options = gdal.WarpOptions(
                format="VRT" if to_vrt else "GTiff",
                dstSRS='EPSG:3857',
                cutlineDSName=str(shapefile),
                cutlineSRS=shapefile_srs,
                cropToCutline=True,
                dstAlpha=True,
                resampleAlg=getattr(self, 'resample_alg', gdal.GRA_Bilinear),
                creationOptions=None if to_vrt else ['TILED=YES', 'BIGTIFF=YES'],
                multithread=True,
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
            )

            ds = gdal.Warp(str(output_tiff), temp_vrt_name, options=warp_options)

            # Cleanup
            if not to_vrt and temp_vrt_name.startswith("/vsimem/"):
                try:
                    gdal.Unlink(temp_vrt_name)
                except:
                    pass

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                size_label = f"{file_size_mb:.1f} MB" if not to_vrt else f"{output_tiff.stat().st_size} bytes"
                self.log(f"      ✓ Created {output_tiff.name} ({size_label})")
                return True
            else:
                self.log(f"      ✗ Warp failed for {input_tiff.name}")
                return False

        except Exception as e:
            self.log(f"      ✗ Error warping {input_tiff.name}: {e}")
            return False

    def build_vrt(self, input_files: List[Path], output_vrt: Path) -> bool:
        """Combine multiple TIFFs/VRTs into a single VRT."""
        try:
            # Validate input files before building VRT
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
                self.log(f"      ✗ No valid input files for VRT")
                return False

            input_strs = [str(f) for f in valid_files]
            # Don't add alpha - input files already have alpha from warp_and_cut
            vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_Bilinear, resolution='highest')
            ds = gdal.BuildVRT(str(output_vrt), input_strs, options=vrt_options)

            if ds:
                ds = None
                if output_vrt.exists():
                    vrt_size_kb = output_vrt.stat().st_size / 1024
                    self.log(f"      ✓ Built VRT: {output_vrt.name} ({vrt_size_kb:.1f} KB)")
                return True
            else:
                self.log(f"      ✗ Failed to build VRT")
                return False

        except Exception as e:
            self.log(f"      ✗ Error building VRT: {e}")
            import traceback
            self.log(f"      {traceback.format_exc()}")
            return False

    def create_geotiff(self, input_vrt: Path, output_tiff: Path, compress: str = 'LZW') -> Tuple[bool, Optional[float]]:
        """Convert VRT to optimized GeoTIFF using Translate with multithreading.

        Uses gdal.Translate instead of gdal.Warp because:
        - VRT is already georeferenced, no reprojection needed
        - Translate is faster for simple copy+compress operations
        - NUM_THREADS creation option enables parallel compression
        """
        try:
            # Log compression status
            num_threads = getattr(self, 'num_threads', '4')
            if compress == 'NONE':
                self.log(f"    Creating GeoTIFF with no compression ({num_threads} threads)...")
            else:
                self.log(f"    Creating GeoTIFF with {compress} compression ({num_threads} threads)...")

            # Configure GDAL for single-process GeoTIFF creation (use all CPUs)
            _configure_gdal_for_phase('geotiff', workers=1)

            # Build creation options based on compression
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

            # Optional clipping window (projWin) if user supplied --clip-projwin
            projwin = getattr(self, 'clip_projwin', None)
            if projwin:
                ulx, uly, lrx, lry = projwin
                self.log(f"    Applying projWin clip: UL({ulx}, {uly}) LR({lrx}, {lry})")

            # Simple progress reporter for long translates (logs every 5%)
            start_time = time.time()
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

            # Use gdal.Translate - faster than Warp when no reprojection needed
            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=creation_opts,
                projWin=projwin if projwin else None,
                callback=progress_cb
            )

            ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                elapsed = time.time() - start_time
                rate = file_size_mb / elapsed if elapsed > 0 else 0
                self.log(f"      ✓ GeoTIFF created: {output_tiff.name} ({file_size_mb:.1f} MB in {elapsed:.1f}s, {rate:.1f} MB/s)")
                return True, elapsed
            else:
                self.log(f"      ✗ Failed to create GeoTIFF")
                return False, None

        except Exception as e:
            error_str = str(e).lower()
            # If ZSTD decompression fails, try with LZW compression instead
            if compress == 'ZSTD' and ('zstd' in error_str or 'frame descriptor' in error_str or 'data corruption' in error_str):
                self.log(f"    ZSTD decompression error detected, retrying with LZW compression...")
                return self.create_geotiff(input_vrt, output_tiff, compress='LZW')

            self.log(f"    Error creating GeoTIFF: {e}")
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

                # Determine cell content based on matches and fallback
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

                # Override with fallback indicator if using fallback
                if entry['fallback'] != 'NO':
                    fallback_date = entry['fallback'][-5:]  # MM-DD
                    cell = f"(→{fallback_date})"

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

    def _format_review_console(self, report_data: List[Dict], stats: Dict) -> str:
        """Format review report as matrix (dates × locations)."""
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " " * 25 + "PROCESSING REVIEW REPORT" + " " * 31 + "║")
        lines.append("╚" + "═" * 78 + "╝")
        lines.append("")

        # Build data structures
        dates = sorted(set(e['date'] for e in report_data), reverse=True)
        locations = sorted(set(e['location'] for e in report_data))

        # Build lookup: date -> location -> cell content
        matrix = defaultdict(dict)
        for entry in report_data:
            date = entry['date']
            loc = entry['location']

            # Determine cell content based on matches and fallback
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

            # Override with fallback indicator if using fallback
            if entry['fallback'] != 'NO':
                fallback_date = entry['fallback'][-5:]  # MM-DD
                cell = f"(→{fallback_date})"

            matrix[date][loc] = cell

        # Determine column width (min 8, max 14)
        if locations:
            col_width = min(14, max(8, max(len(loc) for loc in locations) + 1))
        else:
            col_width = 12

        # Build header row with location names
        header = "Date        │ " + " │ ".join(
            loc[:col_width-1].ljust(col_width-1) for loc in locations
        )
        lines.append(header)

        # Build separator
        separator = "────────────┼" + "┼".join("─" * col_width for _ in locations)
        lines.append(separator)

        # Build data rows
        for date in dates:
            row = f"{date} │ "
            cells = []
            for loc in locations:
                cell_value = matrix[date].get(loc, "---")
                cells.append(cell_value.ljust(col_width-1))
            row += " │ ".join(cells)
            lines.append(row)

        lines.append("")

        # Legend
        lines.append("Legend:")
        lines.append("  filename.ext    = Direct match for this date")
        lines.append("  ---             = No file found (missing)")
        lines.append(f"  (→MM-DD)        = Using fallback from previous date (<= {self.fallback_window_days} days)")
        lines.append("  (N files)       = Multiple files matched")
        lines.append("")

        # Statistics
        lines.append("STATISTICS:")
        lines.append(f"  Total dates: {stats['total_dates']}")
        lines.append(f"  Total locations: {stats['total_locations']}")
        lines.append(f"  Total matches: {stats['total_matches']}")
        lines.append(f"  Missing matches: {stats['missing_matches']}")
        lines.append(f"  Locations using fallback: {stats['fallback_count']}")
        lines.append("")
        lines.append("═" * 80)

        return "\n".join(lines)

    def generate_review_report(self) -> bool:
        """
        Generate review report showing file matches for all dates.
        Saves CSV report and displays formatted summary.
        Returns True if user confirms to proceed, False otherwise.
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

        # Track file matches per location-date for fallback simulation
        vrt_library_sim = defaultdict(dict)

        # Build CSV records lookup: date -> location -> record
        csv_lookup = defaultdict(dict)
        for date, records in self.dole_data.items():
            for rec in records:
                norm_loc = self.normalize_name(rec.get('location', ''))
                csv_lookup[date][norm_loc] = rec

        # Build report data for ALL locations on ALL dates
        sorted_dates = sorted(self.dole_data.keys(), reverse=True)
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
                        vrt_library_sim[location][date] = True
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
                            # Mark it as found so fallback logic doesn't trigger
                            vrt_library_sim[location][date] = True
                            source_files_str = f"📥 WILL DOWNLOAD: {filename}"
                        else:
                            source_files_str = "NOT IN CSV"

                    report_entry = {
                        'date': date,
                        'location': location,
                        'edition': edition,
                        'shapefile_found': shapefile_found,
                        'source_files': source_files_str,
                        'num_files': len(found_files),
                        'fallback': 'NO'
                    }
                else:
                    # No CSV record for this location on this date
                    report_entry = {
                        'date': date,
                        'location': location,
                        'edition': 'N/A',
                        'shapefile_found': 'N/A',
                        'source_files': 'NOT FOUND',
                        'num_files': 0,
                        'fallback': 'NO'
                    }

                report_data.append(report_entry)

        # Second pass: determine fallback usage (limit to recent history)
        max_fallback_days = self.fallback_window_days
        for date in sorted_dates:
            date_dt = self._date_from_str(date)
            if not date_dt:
                continue
            for location in all_locations:
                if location in vrt_library_sim:
                    available_dates = []
                    for d in vrt_library_sim[location].keys():
                        cand_dt = self._date_from_str(d)
                        if not cand_dt:
                            continue
                        if cand_dt <= date_dt and (date_dt - cand_dt).days <= max_fallback_days:
                            available_dates.append(d)
                    if available_dates:
                        most_recent = max(available_dates, key=self._date_from_str)
                        if most_recent != date:
                            # Mark this entry as using fallback
                            for entry in report_data:
                                if entry['date'] == date and entry['location'] == location:
                                    entry['fallback'] = most_recent
                                    break

        # Calculate statistics
        stats = {
            'total_dates': len(sorted_dates),
            'total_locations': len(all_locations),
            'total_matches': sum(1 for e in report_data if e['num_files'] > 0),
            'missing_matches': sum(1 for e in report_data if e['num_files'] == 0),
            'fallback_count': sum(1 for e in report_data if e['fallback'] != 'NO')
        }

        # Display formatted console output
        console_output = self._format_review_console(report_data, stats)
        print("\n" + console_output)

        # Save CSV report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"processing_review_{timestamp}.csv"
        self._save_review_csv(report_data, csv_path)

        # Get user confirmation
        while True:
            response = input("\nProceed with processing? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _rebuild_vrt_library(self, all_locations: List[str]) -> Dict[str, Dict[str, Path]]:
        """
        Rebuild vrt_library from existing temp VRTs for resume support.

        Scans temp_dir for previously created location VRTs so that fallback
        logic works correctly when resuming an interrupted run.

        Returns:
            Dict mapping location -> date -> VRT/TIF path
        """
        vrt_library: Dict[str, Dict[str, Path]] = defaultdict(dict)

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

            # Look for per-location VRTs: {location}_{date}.vrt
            for vrt_file in date_dir.glob("*_*.vrt"):
                # Skip mosaic VRTs
                if vrt_file.name.startswith("mosaic_"):
                    continue
                # Skip RGBA intermediate VRTs
                if "_rgba.vrt" in vrt_file.name:
                    continue

                # Parse location from filename: {location}_{date}.vrt
                stem = vrt_file.stem
                if stem.endswith(f"_{date}"):
                    location = stem[:-len(f"_{date}")]
                    if location in all_locations or self.normalize_name(location) in all_locations:
                        norm_loc = self.normalize_name(location)
                        vrt_library[norm_loc][date] = vrt_file
                        found_count += 1

            # Also check for single warped TIFs (used when only one file per location)
            # These follow pattern: {location}_{original_stem}.tif
            for tif_file in date_dir.glob("*.tif"):
                # Try to extract location from filename
                stem = tif_file.stem
                for loc in all_locations:
                    if stem.startswith(f"{loc}_"):
                        # Only add if we don't already have a VRT for this location/date
                        if date not in vrt_library.get(loc, {}):
                            vrt_library[loc][date] = tif_file
                            found_count += 1
                        break

        if found_count > 0:
            # Count unique location-date pairs
            total_pairs = sum(len(dates) for dates in vrt_library.values())
            self.log(f"  Restored {total_pairs} location-date entries from {len(vrt_library)} locations")
        else:
            self.log("  No existing VRTs found (fresh run)")

        return vrt_library

    def process_all_dates(self):
        """Process all dates in CSV with fallback logic (GeoTIFF only)."""
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

        # === PREPARATION PHASE: Download all missing files upfront ===
        self.log("\n=== Preparing: Downloading missing files ===")
        self._download_missing_files()

        # Track all warped VRTs per location-date
        # Rebuild from existing temp files to support resume with fallback
        vrt_library = self._rebuild_vrt_library(all_locations)

        # Process each date (earliest to latest)
        # Fallback logic uses most_recent available date <= current date
        sorted_dates = sorted(self.dole_data.keys())
        total_dates = len(sorted_dates)

        # Count dates that need GeoTIFF processing (for ETA calculation)
        dates_needing_geotiff = 0
        for date in sorted_dates:
            geotiff_path = self.output_dir / f"{date}.tif"
            if not (geotiff_path.exists() and geotiff_path.stat().st_size > 1024 * 1024):
                dates_needing_geotiff += 1
        self.geotiff_total_planned = dates_needing_geotiff
        self.geotiff_completed = 0
        self.geotiff_times = []
        if dates_needing_geotiff > 0:
            self.log(f"GeoTIFFs to create: {dates_needing_geotiff} (skipping {total_dates - dates_needing_geotiff} existing)")

        geotiff_jobs = []

        for date_idx, date in enumerate(sorted_dates, 1):
            self.log(f"\n[{date_idx}/{total_dates}] Processing date: {date}")

            # Check if output artifacts already exist to skip expensive processing
            geotiff_path = self.output_dir / f"{date}.tif"
            geotiff_exists = geotiff_path.exists() and geotiff_path.stat().st_size > 1024 * 1024
            if geotiff_exists:
                self.log(f"  ✓ GeoTIFF already exists ({geotiff_path.stat().st_size / (1024*1024):.1f} MB) - skipping")
                continue

            records = self.dole_data[date]
            temp_dir = self.temp_dir / date
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Collect warp jobs for parallel processing
            warp_jobs = []

            for rec in records:
                location = rec.get('location', '')
                edition = rec.get('edition', '')
                timestamp = rec.get('wayback_ts')
                filename = rec.get('filename', '')  # Get filename from CSV

                norm_loc = self.normalize_name(location)

                # Find shapefile
                shp_path = self.find_shapefile(location)
                if not shp_path:
                    self.log(f"    {location}: No shapefile found")
                    continue

                # Find source files (use filename from CSV if available)
                download_link = rec.get('download_link', '')
                found_files = self.find_files(location, edition, timestamp, filename, download_link)
                if not found_files:
                    self.log(f"    {location}: No source files found (edition {edition}) [CSV: {filename if filename else 'no match'}]")
                    continue

                self.log(f"    {location} (edition {edition}): {filename if filename else 'pattern match'}")

                # Prepare warp jobs for this location
                for src_file_info in found_files:
                    jobs = self._prepare_warp_job(src_file_info, location, edition, shp_path, temp_dir, date)
                    warp_jobs.extend(jobs)

            # Execute warp jobs in parallel
            if warp_jobs:
                cpu_count_for_warp = max(2, cpu_count - 2)
                self.log(f"  Processing {len(warp_jobs)} warp jobs with {cpu_count_for_warp} parallel workers...")
                results = []

                with ThreadPoolExecutor(max_workers=cpu_count_for_warp) as executor:
                    futures = {executor.submit(self._warp_worker, job): job for job in warp_jobs}

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.log(f"    Error in parallel warp: {e}")

                # Group results by location (validate output files)
                temp_warped_by_location = defaultdict(list)
                for result in results:
                    if result['success'] and result['output_tiff']:
                        output_file = result['output_tiff']
                        if output_file.exists() and output_file.stat().st_size > 1024:
                            temp_warped_by_location[result['location']].append(output_file)
                        else:
                            self.log(f"      Warning: Skipping invalid warped file: {output_file.name}")

                # Build VRTs for each location
                location_count = 0
                for norm_loc, temp_warped in temp_warped_by_location.items():
                    location_count += 1
                    if len(temp_warped) > 1:
                        self.log(f"      Combining {len(temp_warped)} warped files for {norm_loc}...")
                        loc_vrt = temp_dir / f"{norm_loc}_{date}.vrt"
                        if self.build_vrt(temp_warped, loc_vrt):
                            self.log(f"      ✓ VRT created: {loc_vrt.name}")
                            vrt_library[norm_loc][date] = loc_vrt
                    else:
                        self.log(f"      Single file, using directly")
                        vrt_library[norm_loc][date] = temp_warped[0]

                self.log(f"  Processed {location_count} locations with data")
            else:
                self.log(f"  No valid locations found with source files")

            # Build date matrix with fallback logic
            date_matrix = {}
            date_dt = self._date_from_str(date)
            for location in all_locations:
                if location in vrt_library:
                    # Prefer exact date match, otherwise fall back to most recent available
                    if date in vrt_library[location]:
                        date_matrix[location] = vrt_library[location][date]
                    elif vrt_library[location]:
                        # Use most recent available date <= current date to avoid future data
                        available_dates = []
                        if date_dt:
                            for d in vrt_library[location].keys():
                                cand_dt = self._date_from_str(d)
                                if not cand_dt:
                                    continue
                                if cand_dt <= date_dt and (date_dt - cand_dt).days <= self.fallback_window_days:
                                    available_dates.append(d)
                        if available_dates:
                            most_recent = max(available_dates, key=self._date_from_str)
                            date_matrix[location] = vrt_library[location][most_recent]

            # Create mosaic VRT and output GeoTIFF
            if date_matrix:
                mosaic_vrts = list(date_matrix.values())
                self.log(f"  Creating mosaic VRT from {len(mosaic_vrts)} locations...")

                temp_dir.mkdir(parents=True, exist_ok=True)

                mosaic_vrt = temp_dir / f"mosaic_{date}.vrt"
                if self.build_vrt(mosaic_vrts, mosaic_vrt):
                    self.log(f"  ✓ Mosaic VRT created")

                    geotiff_jobs.append((mosaic_vrt, geotiff_path, date))
                    self.log(f"  Queued GeoTIFF creation")
                    self.log(f"  ✓✓✓ {date.upper()} VRT COMPLETE")
                else:
                    self.log(f"  ✗ Mosaic creation failed")
            else:
                self.log(f"  Skipping {date} - no data available")

        if geotiff_jobs:
            # Determine GeoTIFF worker count (conservative by default)
            requested_workers = getattr(self, 'parallel_geotiff', None)
            if not requested_workers or requested_workers < 1:
                requested_workers = min(cpu_count, len(geotiff_jobs))
            geo_workers = max(1, min(requested_workers, len(geotiff_jobs)))

            try:
                import psutil
                available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
            except ImportError:
                available_ram_mb = 4096

            cache_per_worker = max(256, min(4096, available_ram_mb // geo_workers))

            # Cap per-worker threads to avoid oversubscription
            threads_cap = max(1, cpu_count // geo_workers)
            requested_threads = getattr(self, 'num_threads', '4')
            if isinstance(requested_threads, str) and requested_threads.upper() == 'ALL_CPUS':
                worker_threads = threads_cap
            else:
                try:
                    requested_int = int(requested_threads)
                except (TypeError, ValueError):
                    requested_int = threads_cap
                worker_threads = max(1, min(requested_int, threads_cap))

            self.geotiff_total_planned = len(geotiff_jobs)
            self.geotiff_completed = 0
            self.geotiff_times = []

            self.log(f"\n=== Processing {len(geotiff_jobs)} GeoTIFFs in parallel ({geo_workers} workers) ===")
            self.log(f"GeoTIFF threads/worker: {worker_threads} (requested {requested_threads}), cache/worker: {cache_per_worker}MB")

            geotiff_jobs_with_args = []
            for input_vrt, output_tiff, date in geotiff_jobs:
                geotiff_jobs_with_args.append((
                    input_vrt,
                    output_tiff,
                    date,
                    self.compression,
                    worker_threads,
                    getattr(self, 'clip_projwin', None),
                    cache_per_worker
                ))

            failed_jobs = []
            with ProcessPoolExecutor(max_workers=geo_workers) as executor:
                futures = {executor.submit(create_geotiff_worker, job): job for job in geotiff_jobs_with_args}

                for future in as_completed(futures):
                    job = futures[future]
                    date = job[2]
                    try:
                        result = future.result()
                    except Exception as e:
                        self.log(f"  ✗ GeoTIFF worker crashed for {date}: {e}")
                        failed_jobs.append(job)
                        continue

                    if result.get('success'):
                        self.geotiff_completed += 1
                        elapsed = result.get('elapsed')
                        if elapsed:
                            self.geotiff_times.append(elapsed)
                        file_size_mb = result.get('file_size_mb', 0)
                        rate = result.get('rate', 0)
                        compress_used = result.get('compress', self.compression)
                        if elapsed:
                            self.log(f"  ✓ GeoTIFF created: {Path(result['output_tiff']).name} ({file_size_mb:.1f} MB in {elapsed:.1f}s, {rate:.1f} MB/s, {compress_used})")
                        else:
                            self.log(f"  ✓ GeoTIFF created: {Path(result['output_tiff']).name} ({file_size_mb:.1f} MB, {compress_used})")
                        self.log(f"  ✓✓✓ {date.upper()} COMPLETE")
                    else:
                        self.log(f"  ✗ GeoTIFF failed for {date}: {result.get('error', 'unknown error')}")
                        failed_jobs.append(job)

                    remaining_jobs = max(0, self.geotiff_total_planned - self.geotiff_completed)
                    if self.geotiff_times and remaining_jobs > 0:
                        avg_sec = sum(self.geotiff_times) / len(self.geotiff_times)
                        overall_sec = remaining_jobs * avg_sec
                        overall_hours = overall_sec / 3600
                        if overall_hours >= 1:
                            self.log(f"  Overall ETA: {overall_hours:.1f} hours ({remaining_jobs} GeoTIFFs remaining)")
                        else:
                            self.log(f"  Overall ETA: {overall_sec/60:.1f} min ({remaining_jobs} GeoTIFFs remaining)")

            if failed_jobs:
                self.log(f"\nRetrying {len(failed_jobs)} GeoTIFFs sequentially...")
                for job in failed_jobs:
                    input_vrt, output_tiff, date, compress, _, _, _ = job
                    self.log(f"  Retrying {date} sequentially...")
                    success, elapsed = self.create_geotiff(Path(input_vrt), Path(output_tiff), compress=compress)
                    if success and elapsed:
                        self.geotiff_completed += 1
                        self.geotiff_times.append(elapsed)
                        self.log(f"  ✓✓✓ {date.upper()} COMPLETE")
                    else:
                        self.log(f"  ✗ GeoTIFF failed after retry: {date}")

        # Post-process: convert GeoTIFF mosaics to MBTiles/PMTiles and upload
        self._run_postprocess_upload_stage(input_dir=self.output_dir)

        self.log("\n=== Processing Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="FAA Chart Newslicer - Process sectional charts to COGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s -s /path/to/charts -o /path/to/output -c master_dole.csv -b shapefiles
        """
    )

    parser.add_argument(
        "-s", "--source",
        type=Path,
        default=Path("/Volumes/drive/newrawtiffs"),
        help="Source directory containing TIFF/ZIP files (default: /Volumes/drive/newrawtiffs)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("/Volumes/projects/sync"),
        help="Output directory for COGs (default: /Volumes/projects/sync)"
    )

    parser.add_argument(
        "-c", "--csv",
        type=Path,
        default=Path("/Users/ryanhemenway/archive.aero/master_dole.csv"),
        help="Master Dole CSV file (default: /Users/ryanhemenway/archive.aero/master_dole.csv)"
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
        default='tif',
        help="Intermediate warp output format: 'tif' (default) writes cutlines to disk, 'vrt' keeps them virtual to reduce I/O"
    )

    parser.add_argument(
        "--clip-projwin",
        type=float,
        nargs=4,
        metavar=('ULX', 'ULY', 'LRX', 'LRY'),
        default=None,
        help="Optional projWin window (in target CRS, EPSG:3857) applied at VRT→GeoTIFF translate. "
             "Order: ULX ULY LRX LRY. Example for CONUS approx: --clip-projwin -13914936 6446276 -7347086 2753408"
    )

    parser.add_argument(
        "--compression",
        type=str,
        choices=['AUTO', 'ZSTD', 'LZW', 'DEFLATE', 'NONE'],
        default='AUTO',
        help="Compression for GeoTIFF output: AUTO (detect best), ZSTD, LZW, DEFLATE, or NONE (default: AUTO)"
    )

    parser.add_argument(
        "--num-threads",
        type=str,
        default='4',
        help="Number of threads for GeoTIFF compression: a number (e.g., '4') or 'ALL_CPUS' (default: 4). Lower values reduce mutex contention on network drives."
    )

    parser.add_argument(
        "--parallel-geotiff",
        type=int,
        default=3,
        help="Number of parallel GeoTIFF creation jobs (default: 3, tuned for 10-core/16GB systems)"
    )

    parser.add_argument(
        "--download-delay",
        type=float,
        default=8.0,
        help="Seconds to wait between downloads to avoid rate limiting (default: 8.0, try 12.0-20.0 if hitting connection errors)"
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
    slicer.num_threads = args.num_threads
    slicer.parallel_geotiff = args.parallel_geotiff

    # Map resampling algorithm
    resample_map = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline
    }
    slicer.resample_alg = resample_map[args.resample]

    # Override compression if specified
    if args.compression != 'AUTO':
        slicer.compression = args.compression

    slicer.load_dole_data()

    # Build indexes for O(1) lookups (after loading CSV data)
    slicer._build_source_index()
    slicer._build_shapefile_index()

    # During review phase, disable downloads (don't try to validate by downloading)
    # This prevents rate limiting during the review report generation
    slicer.preview_mode = True

    # Generate review report and get user confirmation (without downloading)
    if not slicer.generate_review_report():
        print("Processing cancelled by user.")
        sys.exit(0)

    # Now enable downloads for actual processing
    slicer.preview_mode = False
    slicer.process_all_dates()


if __name__ == '__main__':
    main()
