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
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from osgeo import gdal
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
gdal.SetConfigOption('GDAL_NUM_THREADS', '8')
gdal.SetConfigOption('GDAL_SWATH_SIZE', str(cache_size_mb))
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')


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

    def __init__(self, source_dir: Path, output_dir: Path, csv_file: Path, shape_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.shape_dir = shape_dir
        self.file_index = {}
        self.dole_data = defaultdict(list)
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

    def scan_source_dir(self):
        """Recursively scan directory for charts and index them."""
        self.log(f"Scanning directory: {self.source_dir}...")
        self.file_index = {}

        count = 0
        for root, _, files in os.walk(self.source_dir):
            for name in files:
                if name.lower().endswith(('.zip', '.tif', '.tiff')):
                    current_path = Path(root) / name
                    stem = self.normalize_name(current_path.stem)
                    parts = stem.split('_')

                    if parts[-1].isdigit():
                        edition = parts[-1]
                        location = "_".join(parts[:-1])
                        key = ('EDITION', location, edition)
                        if key not in self.file_index:
                            self.file_index[key] = []
                        self.file_index[key].append(current_path)
                        count += 1
                    elif parts[0].isdigit() and len(parts[0]) >= 14:
                        timestamp = parts[0]
                        key = ('TS', timestamp)
                        if key not in self.file_index:
                            self.file_index[key] = []
                        self.file_index[key].append(current_path)
                        count += 1

        self.log(f"Indexed {count} files across {len(self.file_index)} unique keys.")

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

    def find_files(self, location: str, edition: str, timestamp: Optional[str]) -> List[Path]:
        """Find files matching location, edition, and timestamp."""
        results = []

        if timestamp:
            res = self.file_index.get(('TS', timestamp))
            if res:
                results.extend(res)

        norm_loc = self.normalize_name(location)
        candidate_locs = [
            norm_loc,
            f"{norm_loc}_sec",
            f"{norm_loc}_tac",
            f"{norm_loc}_sectional",
            f"{norm_loc}_terminal",
        ]

        variants = ["north", "south", "east", "west"]
        for v in variants:
            var_base = f"{norm_loc}_{v}"
            candidate_locs.append(var_base)
            candidate_locs.append(f"{var_base}_sec")
            candidate_locs.append(f"{var_base}_tac")

        for loc in candidate_locs:
            key = ('EDITION', loc, edition)
            res = self.file_index.get(key)
            if res:
                results.extend(res)

        return list(set(results))

    def find_shapefile(self, location: str) -> Optional[Path]:
        """Find shapefile for location."""
        norm_loc = self.normalize_name(location)
        candidates = list(self.shape_dir.glob("*.shp"))
        for shp in candidates:
            shp_norm = self.normalize_name(shp.stem)
            if shp_norm == norm_loc:
                return shp
            if norm_loc in shp_norm:
                return shp
        return None

    def unzip_file(self, zip_path: Path, output_dir: Path) -> List[Path]:
        """Extract TIFFs from ZIP file."""
        extracted_tiffs = []
        try:
            if not zip_path.exists():
                self.log(f"  [ERROR] Zip file not found: {zip_path}")
                return []

            with zipfile.ZipFile(zip_path, 'r') as z:
                tiffs = [n for n in z.namelist() if n.lower().endswith(('.tif', '.tiff'))]
                if not tiffs:
                    self.log(f"  [ERROR] No TIFFs found in {zip_path.name}")
                    return []

                extract_path = output_dir / zip_path.stem
                extract_path.mkdir(parents=True, exist_ok=True)

                for target in tiffs:
                    z.extract(target, path=extract_path)
                    full_path = extract_path / target
                    if full_path.exists():
                        extracted_tiffs.append(full_path)

                        # Extract matching TFW if exists
                        target_stem = Path(target).stem
                        tfw_name = f"{target_stem}.tfw"
                        if tfw_name in z.namelist():
                            z.extract(tfw_name, path=extract_path)

            return extracted_tiffs
        except Exception as e:
            self.log(f"  Error unzipping: {e}")
            return []

    def _prepare_warp_job(self, src_file: Path, location: str, edition: str,
                          shp_path: Path, temp_dir: Path, date: str) -> List[Dict]:
        """Prepare warp job parameters for parallel processing."""
        norm_loc = self.normalize_name(location)
        jobs = []

        # Handle ZIP extraction
        if src_file.suffix.lower() == '.zip':
            extracted = self.unzip_file(src_file, temp_dir / "extract")
            tiffs_to_warp = extracted
        else:
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

            # Step 2: Warp with cutline to GTiff with BIGTIFF support
            # Note: Not specifying xRes/yRes allows GDAL to maintain source resolution during warp
            warp_options = gdal.WarpOptions(
                format="GTiff",
                dstSRS='EPSG:3857',
                cutlineDSName=str(shapefile),
                cutlineSRS='EPSG:4326',
                cropToCutline=True,
                dstAlpha=True,
                resampleAlg=gdal.GRA_Bilinear,
                creationOptions=['TILED=YES', f'COMPRESS={self.compression}', 'BIGTIFF=YES'],
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
            input_strs = [str(f) for f in input_files]
            # Don't add alpha - input files already have alpha from warp_and_cut
            vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_Bilinear)
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
        """Convert VRT to optimized GeoTIFF (tiled, compressed)."""
        try:
            self.log(f"    Creating GeoTIFF with {compress} compression...")

            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=['TILED=YES', f'COMPRESS={compress}', 'BIGTIFF=YES', 'PREDICTOR=2']
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
            self.log(f"    Error creating GeoTIFF: {e}")
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
                    found_files = self.find_files(location, edition, timestamp)

                    if found_files:
                        vrt_library_sim[location][date] = True
                        source_files_str = ', '.join([str(f.name) for f in found_files])
                    else:
                        source_files_str = "NOT FOUND"

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

        # Track all warped VRTs per location-date
        vrt_library: Dict[str, Dict[str, Path]] = defaultdict(dict)

        # Process each date (latest to earliest)
        sorted_dates = sorted(self.dole_data.keys(), reverse=True)
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

                norm_loc = self.normalize_name(location)

                # Check if this is a shapefile-free location
                if norm_loc in self.SHAPEFILE_FREE_LOCATIONS:
                    # No shapefile needed - find source files directly
                    found_files = self.find_files(location, edition, timestamp)
                    if not found_files:
                        self.log(f"    {location}: No source files found (edition {edition})")
                        continue

                    self.log(f"    {location} (shapefile-free, edition {edition}): Processing {len(found_files)} file(s)...")

                    # Extract ZIPs if needed, add files directly to VRT library
                    for src_file in found_files:
                        if src_file.suffix.lower() == '.zip':
                            self.log(f"      Extracting {src_file.name}...")
                            extracted = self.unzip_file(src_file, temp_dir / "extract")
                            tiffs_to_add = extracted
                            self.log(f"      Extracted {len(tiffs_to_add)} TIFFs")
                        else:
                            tiffs_to_add = [src_file]

                        # Add directly to vrt_library without warping
                        if tiffs_to_add:
                            if len(tiffs_to_add) > 1:
                                # Build VRT for multiple files
                                loc_vrt = temp_dir / f"{norm_loc}_{date}.vrt"
                                if self.build_vrt(tiffs_to_add, loc_vrt):
                                    vrt_library[norm_loc][date] = loc_vrt
                            else:
                                # Single file, use directly
                                vrt_library[norm_loc][date] = tiffs_to_add[0]

                    location_metadata[norm_loc] = {'edition': edition}
                    continue  # Skip shapefile processing

                # Normal shapefile-based processing
                # Find shapefile
                shp_path = self.find_shapefile(location)
                if not shp_path:
                    self.log(f"    {location}: No shapefile found")
                    continue

                # Find source files
                found_files = self.find_files(location, edition, timestamp)
                if not found_files:
                    self.log(f"    {location}: No source files found (edition {edition})")
                    continue

                self.log(f"    {location} (edition {edition}): Preparing {len(found_files)} file(s)...")

                # Prepare warp jobs for this location
                for src_file in found_files:
                    jobs = self._prepare_warp_job(src_file, location, edition, shp_path, temp_dir, date)
                    warp_jobs.extend(jobs)

                # Store location metadata
                location_metadata[norm_loc] = {'edition': edition}

            # Execute warp jobs in parallel (4 workers)
            if warp_jobs:
                self.log(f"  Processing {len(warp_jobs)} warp jobs with 4 parallel workers...")
                results = []

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(self._warp_worker, job): job for job in warp_jobs}

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.log(f"    Error in parallel warp: {e}")

                # Group results by location
                temp_warped_by_location = defaultdict(list)
                for result in results:
                    if result['success']:
                        temp_warped_by_location[result['location']].append(result['output_tiff'])

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
                    available_dates = [d for d in vrt_library[location].keys() if d <= date]
                    if available_dates:
                        most_recent = sorted(available_dates)[-1]
                        date_matrix[location] = vrt_library[location][most_recent]

            # Create mosaic VRT and output products
            if date_matrix:
                mosaic_vrts = list(date_matrix.values())
                self.log(f"  Creating mosaic VRT from {len(mosaic_vrts)} locations...")

                date_out_dir = self.output_dir / date
                date_out_dir.mkdir(parents=True, exist_ok=True)

                mosaic_vrt = date_out_dir / f"mosaic_{date}.vrt"
                if self.build_vrt(mosaic_vrts, mosaic_vrt):
                    self.log(f"  ✓ Mosaic VRT created")

                    # Generate output based on configured format
                    output_format = getattr(self, 'output_format', 'geotiff')
                    zoom_levels = getattr(self, 'zoom_levels', '0-11')

                    if output_format in ['geotiff', 'both']:
                        geotiff_path = date_out_dir / f"mosaic_{date}.tif"
                        self.log(f"  Creating GeoTIFF (this may take several minutes)...")
                        self.create_geotiff(mosaic_vrt, geotiff_path, compress=self.compression)

                    if output_format in ['tiles', 'both']:
                        tiles_dir = date_out_dir / 'tiles'
                        self.log(f"  Generating tiles...")
                        self.generate_tiles(mosaic_vrt, tiles_dir, zoom_levels)

                    self.log(f"  ✓✓✓ {date.upper()} COMPLETE")
                else:
                    self.log(f"  ✗ Mosaic creation failed")
            else:
                self.log(f"  Skipping {date} - no data available")

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
        choices=['geotiff', 'tiles', 'both'],
        default='geotiff',
        help="Output format: geotiff (LZW compressed), tiles (WEBP), or both (default: geotiff)"
    )

    parser.add_argument(
        "-z", "--zoom",
        type=str,
        default='0-11',
        help="Zoom levels for tile generation, e.g., '0-11' (default: 0-11)"
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
    slicer = ChartSlicer(args.source, args.output, args.csv, args.shapefiles)
    slicer.output_format = args.format
    slicer.zoom_levels = args.zoom
    slicer.scan_source_dir()
    slicer.load_dole_data()

    # Generate review report and get user confirmation
    if not slicer.generate_review_report():
        print("Processing cancelled by user.")
        sys.exit(0)

    slicer.process_all_dates()


if __name__ == '__main__':
    main()
