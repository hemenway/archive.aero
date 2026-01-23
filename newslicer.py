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
import subprocess
import zipfile
import requests
import time
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

# Set GDAL options
try:
    import psutil
    available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
    cache_size_mb = max(512, min(8192, available_ram_mb * 3 // 4))  # 75% of RAM, max 8GB
except ImportError:
    cache_size_mb = 2048

gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
gdal.SetConfigOption('GDAL_SWATH_SIZE', str(cache_size_mb * 1024 * 1024))  # Convert MB to bytes
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')


def create_mbtiles_worker(args):
    """Worker for parallel MBTiles creation."""
    input_vrt, output_mbtiles = args
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        
        translate_options = gdal.TranslateOptions(
            format='MBTiles',
            creationOptions=[
                'TILE_FORMAT=WEBP',
                'QUALITY=85',
                'ZOOM_LEVEL_STRATEGY=AUTO'
            ]
        )
        
        # Create parent directory if needed (should exist, but safety first)
        output_mbtiles.parent.mkdir(parents=True, exist_ok=True)
        
        ds = gdal.Translate(str(output_mbtiles), str(input_vrt), options=translate_options)
        
        if ds:
            ds = None
            size_mb = output_mbtiles.stat().st_size / (1024 * 1024) if output_mbtiles.exists() else 0
            return (True, output_mbtiles, size_mb, None)
        else:
            return (False, output_mbtiles, 0, "GDAL Translate failed")
            
    except Exception as e:
        return (False, output_mbtiles, 0, str(e))


class ChartSlicer:
    """Process FAA charts and create COGs."""

    # Locations that don't require shapefile cutting
    SHAPEFILE_FREE_LOCATIONS = {
        'hawaiian_islands',
        'mariana_islands',
        'samoan_islands',
        'western_aleutian_islands'
    }

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
                 preview_mode: bool = False, download_delay: float = 8.0):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.shape_dir = shape_dir
        self.dole_data = defaultdict(list)
        self.srs_cache = {}  # Cache for shapefile SRS lookups
        self.preview_mode = preview_mode  # If True, only report what would be downloaded
        self.download_delay = download_delay  # Seconds to wait between downloads (to avoid rate limiting)
        self.last_download_time = 0  # Track last download time for throttling
        self.abbreviations = {
            "hawaiian_is": "hawaiian_islands",
            "dallas_ft_worth": "dallas_ft_worth",
            "mariana_islands_inset": "mariana_islands",
        }

        # Determine compression method based on GDAL support
        self.compression = 'ZSTD' if self._check_zstd_support() else 'LZW'
        if self.compression == 'LZW':
            # This will be logged when log() is called for the first time
            self._compression_warning = True

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

            # Count extracted files
            tif_count = len(list(extract_dir.glob('**/*.tif')))
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
        Resolve filename from CSV to actual file path(s).

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

            # Check direct path FIRST (fast path)
            direct_path = self.source_dir / dir_name
            if direct_path.exists() and direct_path.is_dir():
                for tif_file in direct_path.glob('**/*.tif'):
                    if tif_file.is_file():
                        results.append((tif_file, None))
                if results:
                    return results

            # Fall back to os.walk only if direct path doesn't exist
            for root, dirs, files in os.walk(self.source_dir):
                if dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    # Find all TIF files in this directory
                    for tif_file in dir_path.glob('**/*.tif'):
                        if tif_file.is_file():
                            results.append((tif_file, None))
                    if results:
                        return results  # Found and extracted, return early
                    break

            # If not found, try to download (but not in preview mode)
            if not results and download_link and not self.preview_mode:
                self.log(f"    File not found locally: {filename}")
                zip_path = self.source_dir / filename
                extract_dir = self.source_dir / dir_name

                # Download the ZIP file
                if self.download_file(download_link, zip_path):
                    # Extract it
                    if self.extract_zip(zip_path, extract_dir):
                        # Find extracted TIF files
                        for tif_file in extract_dir.glob('**/*.tif'):
                            if tif_file.is_file():
                                results.append((tif_file, None))
                        # Keep the ZIP file for reference
                        return results
                    else:
                        self.log(f"    Failed to extract: {filename}")
                else:
                    self.log(f"    Failed to download: {download_link}")

        else:
            # Case 2: Direct .tif file
            # Check direct path FIRST (fast path)
            direct_path = self.source_dir / filename
            if direct_path.exists() and direct_path.is_file():
                results.append((direct_path, None))
                return results

            # Fall back to os.walk only if direct path doesn't exist
            for root, _, files in os.walk(self.source_dir):
                for fname in files:
                    if fname == filename and fname.endswith('.tif'):
                        file_path = Path(root) / fname
                        results.append((file_path, None))
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
        """Find shapefile for location - only searches sectional shapefiles."""
        norm_loc = self.normalize_name(location)
        sectional_dir = self.shape_dir / "sectional"

        # Only search sectional directory to avoid TAC, Terminal, Helicopter, etc.
        if not sectional_dir.exists():
            self.log(f"WARNING: Sectional shapefile directory not found at {sectional_dir}")
            return None

        candidates = list(sectional_dir.glob("**/*.shp"))
        for shp in candidates:
            shp_norm = self.normalize_name(shp.stem)
            if shp_norm == norm_loc:
                return shp
            if norm_loc in shp_norm:
                return shp
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

        # Create job entries for each TIFF
        for tiff in tiffs_to_warp:
            sanitized_stem = self.sanitize_filename(tiff.stem)
            temp_tif = temp_dir / f"{norm_loc}_{sanitized_stem}.tif"
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
        """Warp and cut TIFF using shapefile as cutline."""
        try:
            self.log(f"    Warping {input_tiff.name}...")

            # Step 1: Expand to RGBA
            sanitized_stem = self.sanitize_filename(input_tiff.stem)
            temp_vrt_name = f"/vsimem/{sanitized_stem}_rgba.vrt"
            translate_options = gdal.TranslateOptions(format="VRT", rgbExpand="rgba")
            ds_vrt = gdal.Translate(temp_vrt_name, str(input_tiff), options=translate_options)
            ds_vrt = None

            # Get the actual SRS of the shapefile (don't hardcode EPSG:4326)
            shapefile_srs = self.get_shapefile_srs(shapefile)

            # Step 2: Warp with cutline to GTiff with BIGTIFF support
            # Note: Not specifying xRes/yRes allows GDAL to maintain source resolution during warp
            # Compress during warp to reduce intermediate I/O
            warp_options = gdal.WarpOptions(
                format="GTiff",
                dstSRS='EPSG:3857',
                cutlineDSName=str(shapefile),
                cutlineSRS=shapefile_srs,
                cropToCutline=True,
                dstAlpha=True,
                resampleAlg=getattr(self, 'resample_alg', gdal.GRA_Bilinear),
                creationOptions=['TILED=YES', f'COMPRESS={self.compression}', 'PREDICTOR=2', 'BIGTIFF=YES'],
                multithread=True,
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
            )

            ds = gdal.Warp(str(output_tiff), temp_vrt_name, options=warp_options)

            # Cleanup
            try:
                gdal.Unlink(temp_vrt_name)
            except:
                pass

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                self.log(f"      ✓ Created {output_tiff.name} ({file_size_mb:.1f} MB)")
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

    def create_geotiff(self, input_vrt: Path, output_tiff: Path, compress: str = 'LZW') -> bool:
        """Convert VRT to optimized GeoTIFF using Translate with multithreading.

        Uses gdal.Translate instead of gdal.Warp because:
        - VRT is already georeferenced, no reprojection needed
        - Translate is faster for simple copy+compress operations
        - NUM_THREADS creation option enables parallel compression
        """
        try:
            self.log(f"    Creating GeoTIFF with {compress} compression (multithreaded)...")

            # Use gdal.Translate - faster than Warp when no reprojection needed
            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=[
                    'TILED=YES',
                    f'COMPRESS={compress}',
                    'PREDICTOR=2',
                    'BIGTIFF=YES',
                    'BLOCKXSIZE=512',
                    'BLOCKYSIZE=512',
                    'NUM_THREADS=ALL_CPUS'
                ]
            )

            ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                file_size_mb = output_tiff.stat().st_size / (1024 * 1024) if output_tiff.exists() else 0
                self.log(f"      ✓ GeoTIFF created: {output_tiff.name} ({file_size_mb:.1f} MB)")
                return True
            else:
                self.log(f"      ✗ Failed to create GeoTIFF")
                return False

        except Exception as e:
            error_str = str(e).lower()
            # If ZSTD decompression fails, try with LZW compression instead
            if compress == 'ZSTD' and ('zstd' in error_str or 'frame descriptor' in error_str or 'data corruption' in error_str):
                self.log(f"    ZSTD decompression error detected, retrying with LZW compression...")
                return self.create_geotiff(input_vrt, output_tiff, compress='LZW')

            self.log(f"    Error creating GeoTIFF: {e}")
            import traceback
            self.log(f"    {traceback.format_exc()}")
            return False

    def create_mbtiles(self, input_vrt: Path, output_mbtiles: Path) -> bool:
        """Convert VRT to MBTiles with WEBP tiles.

        Uses gdal.Translate to create MBTiles directly from VRT.
        Tiles are stored as WEBP with quality 85.
        ZOOM_LEVEL_STRATEGY=AUTO lets GDAL determine optimal zoom levels.
        """
        try:
            self.log(f"    Creating MBTiles with WEBP tiles...")

            translate_options = gdal.TranslateOptions(
                format='MBTiles',
                creationOptions=[
                    'TILE_FORMAT=WEBP',
                    'QUALITY=85',
                    'ZOOM_LEVEL_STRATEGY=AUTO'
                ]
            )

            ds = gdal.Translate(str(output_mbtiles), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                file_size_mb = output_mbtiles.stat().st_size / (1024 * 1024) if output_mbtiles.exists() else 0
                self.log(f"      ✓ MBTiles created: {output_mbtiles.name} ({file_size_mb:.1f} MB)")
                return True
            else:
                self.log(f"      ✗ Failed to create MBTiles")
                return False

        except Exception as e:
            self.log(f"    Error creating MBTiles: {e}")
            import traceback
            self.log(f"    {traceback.format_exc()}")
            return False

    def generate_tiles(self, input_vrt: Path, output_dir: Path, zoom_levels: str) -> bool:
        """Generate tiles using gdal2tiles.py."""
        try:
            self.log(f"    Generating tiles (zoom: {zoom_levels})...")

            # Check availability of gdal2tiles.py
            gdal2tiles_cmd = subprocess.run(['which', 'gdal2tiles.py'], capture_output=True, text=True)
            if gdal2tiles_cmd.returncode != 0:
                gdal2tiles_cmd = "gdal2tiles.py"
            else:
                gdal2tiles_cmd = gdal2tiles_cmd.stdout.strip()

            # Use reduced CPU count to avoid locking up system
            cpu_count_for_tiles = max(1, cpu_count - 2)

            cmd = [
                gdal2tiles_cmd,
                '--zoom', zoom_levels,
                '--processes', str(cpu_count_for_tiles),
                '--webviewer=none',
                '--exclude',
                '--tiledriver=WEBP',
                '--webp-quality=50',
                str(input_vrt),
                str(output_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            if result.returncode == 0:
                self.log(f"      ✓ Tiles generated in {output_dir.name}")
                return True
            else:
                self.log(f"      ✗ Tile generation failed with return code {result.returncode}")
                if result.stderr:
                    self.log(f"      Error: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"      ✗ Tile generation timed out after 2 hours")
            return False
        except Exception as e:
            self.log(f"      ✗ Error generating tiles: {e}")
            return False

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
        lines.append("  (→MM-DD)        = Using fallback from previous date")
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

                    # Find shapefile (not needed for special locations)
                    if location in self.SHAPEFILE_FREE_LOCATIONS:
                        shapefile_found = "NOT NEEDED"
                        shp_path = None
                    else:
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

        # Second pass: determine fallback usage
        for date in sorted_dates:
            for location in all_locations:
                if location in vrt_library_sim:
                    available_dates = [d for d in vrt_library_sim[location].keys() if d <= date]
                    if available_dates:
                        most_recent = sorted(available_dates)[-1]
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

    def preview_downloads(self):
        """
        Preview what files will be needed for each date and location.
        Shows as a pivot table: dates (rows) x locations (columns)
        Cells show what file/folder will be looked for during processing.
        """
        self.log("\n=== Processing Preview - File Lookup Table ===")
        self.log("Shows what files newslicer will look for when processing each date/location combo.\n")

        # Build pivot table: date -> location -> filename
        pivot = defaultdict(lambda: defaultdict(str))
        all_locations = set()
        missing_count = 0
        found_count = 0

        for date in sorted(self.dole_data.keys(), reverse=True):
            records = self.dole_data[date]

            for rec in records:
                location = rec.get('location', '')
                filename = rec.get('filename', '')
                norm_loc = self.normalize_name(location)

                if not location or not filename:
                    continue

                all_locations.add(norm_loc)

                # Check if file exists locally
                if filename.endswith('.zip'):
                    dir_name = filename[:-4]
                    local_path = self.source_dir / dir_name
                else:
                    local_path = self.source_dir / filename

                # Show status: local file, remote URL, or missing
                if local_path.exists():
                    pivot[date][norm_loc] = f"✓ {filename[:40]}"
                    found_count += 1
                else:
                    pivot[date][norm_loc] = f"📥 {filename[:40]}"  # Will download
                    missing_count += 1

        # Print pivot table (latest to oldest)
        sorted_dates = sorted(pivot.keys(), reverse=True)
        sorted_locations = sorted(all_locations)

        # Header row
        header = f"{'Date':<15} "
        for loc in sorted_locations[:10]:  # Show first 10 locations
            header += f"{loc[:18]:18} "
        if len(sorted_locations) > 10:
            header += f"... +{len(sorted_locations)-10} more"
        self.log(header)
        self.log("=" * 200)

        # Data rows
        for date in sorted_dates:
            row = f"{date:<15} "
            for loc in sorted_locations[:10]:
                cell = pivot[date].get(loc, "---")
                row += f"{cell:<18} "
            self.log(row)

        self.log("\n=== Legend ===")
        self.log("✓ = File exists locally (no download needed)")
        self.log("📥 = File needs to be downloaded")
        self.log("--- = Not in CSV for this date/location combo")

        self.log(f"\n=== Summary ===")
        self.log(f"Total dates: {len(sorted_dates)}")
        self.log(f"Total locations: {len(sorted_locations)}")
        self.log(f"Files already local: {found_count}")
        self.log(f"Files to download: {missing_count}")

        if missing_count > 0:
            self.log(f"\nWhen you run newslicer for real:")
            self.log(f"- It will process each date in order")
            self.log(f"- For each date, it will look for these files per location")
            self.log(f"- If a file is missing, it will download with --download-delay {self.download_delay}s between requests")
            self.log(f"- Estimated time: {missing_count * self.download_delay / 60:.0f} minutes at {self.download_delay}s delay")

    def process_all_dates(self):
        """Process all dates in CSV with fallback logic."""
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
        self.log(f"Found {len(all_locations)} locations from CSV ({shapefile_count} with shapefiles, {len(self.SHAPEFILE_FREE_LOCATIONS)} shapefile-free).")

        # === PREPARATION PHASE: Download all missing files upfront ===
        self.log("\n=== Preparing: Downloading missing files ===")
        self._download_missing_files()

        # Track all warped VRTs per location-date
        vrt_library: Dict[str, Dict[str, Path]] = defaultdict(dict)

        # Queue for MBTiles jobs
        mbtiles_jobs = []

        # Process each date (earliest to latest)
        # Fallback logic uses most_recent available date, so processing order doesn't matter
        sorted_dates = sorted(self.dole_data.keys())
        total_dates = len(sorted_dates)

        for date_idx, date in enumerate(sorted_dates, 1):
            self.log(f"\n[{date_idx}/{total_dates}] Processing date: {date}")

            records = self.dole_data[date]
            temp_dir = self.output_dir / ".temp" / date
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Collect warp jobs for parallel processing
            warp_jobs = []
            location_metadata = {}  # Track location/edition for each job

            for rec in records:
                location = rec.get('location', '')
                edition = rec.get('edition', '')
                timestamp = rec.get('wayback_ts')
                filename = rec.get('filename', '')  # Get filename from CSV

                norm_loc = self.normalize_name(location)

                # Check if this is a shapefile-free location
                if norm_loc in self.SHAPEFILE_FREE_LOCATIONS:
                    # No shapefile needed - find source files directly
                    download_link = rec.get('download_link', '')
                    found_files = self.find_files(location, edition, timestamp, filename, download_link)
                    if not found_files:
                        self.log(f"    {location}: No source files found (edition {edition}) [CSV: {filename if filename else 'no match'}]")
                        continue

                    self.log(f"    {location} (shapefile-free, edition {edition}): {filename if filename else 'pattern match'}")

                    # Warp TIF files to EPSG:3857 for mosaic compatibility
                    # (With pre-unzipped files, all found_files are direct TIF paths)
                    warped_files = []
                    for src_file_info in found_files:
                        # Unpack the file info (always (Path, None) with pre-unzipped structure)
                        if isinstance(src_file_info, tuple):
                            src_file, _ = src_file_info
                        else:
                            src_file = src_file_info

                        # Warp each TIF file to EPSG:3857 without shapefile cutline
                        for tiff in [src_file]:
                            sanitized_stem = self.sanitize_filename(tiff.stem)
                            warped_tiff = temp_dir / f"{norm_loc}_{sanitized_stem}_warped.tif"

                            try:
                                self.log(f"      Warping {tiff.name}...")

                                # Expand to RGBA
                                temp_vrt_name = f"/vsimem/{sanitized_stem}_rgba.vrt"
                                translate_options = gdal.TranslateOptions(format="VRT", rgbExpand="rgba")
                                ds_vrt = gdal.Translate(temp_vrt_name, str(tiff), options=translate_options)
                                ds_vrt = None

                                # Warp to EPSG:3857 (no cutline needed for these locations)
                                # Compress during warp to reduce intermediate I/O
                                warp_options = gdal.WarpOptions(
                                    format="GTiff",
                                    dstSRS='EPSG:3857',
                                    dstAlpha=True,
                                    resampleAlg=getattr(self, 'resample_alg', gdal.GRA_Bilinear),
                                    creationOptions=['TILED=YES', f'COMPRESS={self.compression}', 'PREDICTOR=2', 'BIGTIFF=YES'],
                                    multithread=True
                                )

                                ds = gdal.Warp(str(warped_tiff), temp_vrt_name, options=warp_options)

                                # Cleanup
                                try:
                                    gdal.Unlink(temp_vrt_name)
                                except:
                                    pass

                                if ds:
                                    ds = None
                                    file_size_mb = warped_tiff.stat().st_size / (1024 * 1024) if warped_tiff.exists() else 0
                                    self.log(f"        ✓ Warped {tiff.name} ({file_size_mb:.1f} MB)")
                                    warped_files.append(warped_tiff)
                                else:
                                    self.log(f"        ✗ Warp failed for {tiff.name}")
                            except Exception as e:
                                self.log(f"        ✗ Error warping {tiff.name}: {e}")

                    # Build VRT from warped files
                    if warped_files:
                        if len(warped_files) > 1:
                            loc_vrt = temp_dir / f"{norm_loc}_{date}.vrt"
                            if self.build_vrt(warped_files, loc_vrt):
                                vrt_library[norm_loc][date] = loc_vrt
                        else:
                            vrt_library[norm_loc][date] = warped_files[0]

                    location_metadata[norm_loc] = {'edition': edition}
                    continue  # Skip shapefile processing

                # Normal shapefile-based processing
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

                # Store location metadata
                location_metadata[norm_loc] = {'edition': edition}

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
            for location in all_locations:
                if location in vrt_library:
                    # Prefer exact date match, otherwise fall back to most recent available
                    if date in vrt_library[location]:
                        date_matrix[location] = vrt_library[location][date]
                    elif vrt_library[location]:
                        # Use most recent available date (works regardless of processing order)
                        available_dates = sorted(vrt_library[location].keys())
                        most_recent = available_dates[-1]
                        date_matrix[location] = vrt_library[location][most_recent]

            # Create mosaic VRT and output products
            if date_matrix:
                mosaic_vrts = list(date_matrix.values())
                self.log(f"  Creating mosaic VRT from {len(mosaic_vrts)} locations...")

                temp_dir = self.output_dir / ".temp" / date
                temp_dir.mkdir(parents=True, exist_ok=True)

                mosaic_vrt = temp_dir / f"mosaic_{date}.vrt"
                if self.build_vrt(mosaic_vrts, mosaic_vrt):
                    self.log(f"  ✓ Mosaic VRT created")

                    # Generate output based on configured format
                    output_format = getattr(self, 'output_format', 'geotiff')
                    zoom_levels = getattr(self, 'zoom_levels', '0-11')

                    geotiff_path = self.output_dir / f"{date}.tif"
                    geotiff_exists = geotiff_path.exists() and geotiff_path.stat().st_size > 1024 * 1024  # At least 1 MB

                    mbtiles_path = self.output_dir / f"{date}.mbtiles"
                    mbtiles_exists = mbtiles_path.exists() and mbtiles_path.stat().st_size > 1024 * 1024  # At least 1 MB

                    tiles_dir = self.output_dir / f"{date}_tiles" if output_format in ['tiles', 'both'] else None
                    tiles_exist = tiles_dir and tiles_dir.exists() and list(tiles_dir.glob('**/*.webp'))

                    # Check if this date is already complete
                    if output_format == 'geotiff' and geotiff_exists:
                        self.log(f"  ⊘ GeoTIFF already exists ({geotiff_path.stat().st_size / (1024*1024):.1f} MB) - skipping")
                    elif output_format == 'mbtiles' and mbtiles_exists:
                        self.log(f"  ⊘ MBTiles already exists ({mbtiles_path.stat().st_size / (1024*1024):.1f} MB) - skipping")
                    elif output_format == 'tiles' and tiles_exist:
                        self.log(f"  ⊘ Tiles already exist - skipping")
                    elif output_format == 'both' and geotiff_exists and tiles_exist:
                        self.log(f"  ⊘ GeoTIFF and tiles already exist - skipping")
                    else:
                        if output_format in ['geotiff', 'both'] and not geotiff_exists:
                            self.log(f"  Creating GeoTIFF (this may take several minutes)...")
                            self.create_geotiff(mosaic_vrt, geotiff_path, compress=self.compression)

                        if output_format == 'mbtiles' and not mbtiles_exists:
                            self.log(f"  Queuing MBTiles creation...")
                            mbtiles_jobs.append((mosaic_vrt, mbtiles_path))

                        if output_format in ['tiles', 'both'] and not tiles_exist:
                            self.log(f"  Generating tiles...")
                            tiles_output_dir = self.output_dir / f"{date}_tiles"
                            self.generate_tiles(mosaic_vrt, tiles_output_dir, zoom_levels)

                    self.log(f"  ✓✓✓ {date.upper()} COMPLETE")
                else:
                    self.log(f"  ✗ Mosaic creation failed")
            else:
                self.log(f"  Skipping {date} - no data available")

        # Process queued MBTiles jobs
        if mbtiles_jobs:
            workers = getattr(self, 'parallel_mbtiles', 6)
            self.log(f"\n=== Processing {len(mbtiles_jobs)} MBTiles jobs in parallel ({workers} workers) ===")
            
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all jobs
                futures = {executor.submit(create_mbtiles_worker, job): job for job in mbtiles_jobs}
                
                # Process results as they complete
                for i, future in enumerate(as_completed(futures), 1):
                    input_job = futures[future]
                    output_path = input_job[1]
                    try:
                        success, path, size_mb, error = future.result()
                        if success:
                            self.log(f"[{i}/{len(mbtiles_jobs)}] ✓ MBTiles created: {path.name} ({size_mb:.1f} MB)")
                        else:
                            self.log(f"[{i}/{len(mbtiles_jobs)}] ✗ Failed: {path.name} - {error}")
                    except Exception as e:
                        self.log(f"[{i}/{len(mbtiles_jobs)}] ✗ Error processing {output_path.name}: {e}")

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
        default=Path("/Volumes/drive/sync"),
        help="Output directory for COGs (default: /Volumes/drive/sync)"
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
        "-f", "--format",
        type=str,
        choices=['geotiff', 'mbtiles', 'tiles', 'both'],
        default='geotiff',
        help="Output format: geotiff (LZW compressed), mbtiles (WEBP tiles), tiles (WEBP), or both (default: geotiff)"
    )

    parser.add_argument(
        "-z", "--zoom",
        type=str,
        default='0-11',
        help="Zoom levels for tile generation, e.g., '0-11' (default: 0-11)"
    )

    parser.add_argument(
        "-r", "--resample",
        type=str,
        choices=['nearest', 'bilinear', 'cubic', 'cubicspline'],
        default='bilinear',
        help="Resampling algorithm for warping (default: bilinear). Use 'nearest' for faster processing with charts"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview downloads without actually downloading (dry-run mode)"
    )

    parser.add_argument(
        "--download-delay",
        type=float,
        default=8.0,
        help="Seconds to wait between downloads to avoid rate limiting (default: 8.0, try 12.0-20.0 if hitting connection errors)"
    )

    parser.add_argument(
        "--parallel-mbtiles",
        type=int,
        default=6,
        help="Number of parallel MBTiles creation jobs (default: 6)"
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

    # Process
    slicer = ChartSlicer(args.source, args.output, args.csv, args.shapefiles,
                         preview_mode=args.preview, download_delay=args.download_delay)
    slicer.output_format = args.format
    slicer.zoom_levels = args.zoom
    slicer.parallel_mbtiles = args.parallel_mbtiles

    # Map resampling algorithm
    resample_map = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline
    }
    slicer.resample_alg = resample_map[args.resample]

    slicer.load_dole_data()

    # If preview mode, show what would be downloaded and exit
    if args.preview:
        slicer.preview_downloads()
        sys.exit(0)

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
