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
    cache_size_mb = max(512, min(4096, available_ram_mb // 2))
except ImportError:
    cache_size_mb = 2048

gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
gdal.SetConfigOption('GDAL_NUM_THREADS', '8')
gdal.SetConfigOption('GDAL_SWATH_SIZE', str(cache_size_mb))
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')


class ChartSlicer:
    """Process FAA charts and create COGs."""

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

    def log(self, msg: str):
        """Print log message with timestamp."""
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
                creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES'],
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

    def process_all_dates(self):
        """Process all dates in CSV with fallback logic."""
        self.log("\n=== Starting Chart Processing ===")

        # Get all locations from shapefiles
        all_locations = []
        for shp in self.shape_dir.glob("*.shp"):
            norm_loc = self.normalize_name(shp.stem)
            all_locations.append(norm_loc)

        all_locations = sorted(set(all_locations))
        self.log(f"Found {len(all_locations)} locations with shapefiles.")

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

            # Process each location for this date
            location_count = 0
            for rec in records:
                location = rec.get('location', '')
                edition = rec.get('edition', '')
                timestamp = rec.get('wayback_ts')

                norm_loc = self.normalize_name(location)

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

                location_count += 1
                self.log(f"    {location} (edition {edition}): Processing {len(found_files)} file(s)...")

                # Process files
                temp_warped = []
                for src_file in found_files:
                    if src_file.suffix.lower() == '.zip':
                        self.log(f"      Extracting {src_file.name}...")
                        extracted = self.unzip_file(src_file, temp_dir / "extract")
                        tiffs_to_warp = extracted
                        self.log(f"      Extracted {len(tiffs_to_warp)} TIFFs")
                    else:
                        tiffs_to_warp = [src_file]

                    for tiff in tiffs_to_warp:
                        sanitized_stem = self.sanitize_filename(tiff.stem)
                        temp_tif = temp_dir / f"{norm_loc}_{sanitized_stem}.tif"
                        if self.warp_and_cut(tiff, shp_path, temp_tif):
                            temp_warped.append(temp_tif)

                # Combine warped files for this location-date
                if temp_warped:
                    if len(temp_warped) > 1:
                        self.log(f"      Combining {len(temp_warped)} warped files into VRT...")
                        loc_vrt = temp_dir / f"{norm_loc}_{date}.vrt"
                        if self.build_vrt(temp_warped, loc_vrt):
                            self.log(f"      ✓ VRT created: {loc_vrt.name}")
                            vrt_library[norm_loc][date] = loc_vrt
                    else:
                        self.log(f"      Single file, using directly")
                        vrt_library[norm_loc][date] = temp_warped[0]
                else:
                    self.log(f"      No files were successfully warped")

            self.log(f"  Processed {location_count} locations with data")

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
                        self.create_geotiff(mosaic_vrt, geotiff_path, compress='LZW')

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
    slicer.process_all_dates()


if __name__ == '__main__':
    main()
