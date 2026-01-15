#!/usr/bin/env python3
"""
FAA Chart Tool - Unified
Combines chart cropping (using shapefile cutlines), merging, and tile generation.
"""

import os
import sys
import threading
import queue
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import csv
import re
import datetime
import zipfile
from collections import defaultdict
import time
import json

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL is required but not installed.")
    print("Install with: pip install gdal --break-system-packages")
    sys.exit(1)

# Configure GDAL for optimal performance on multi-core systems
import multiprocessing
cpu_count = multiprocessing.cpu_count()

# Set GDAL cache to use up to 50% of available RAM (minimum 512MB, maximum 4GB)
try:
    import psutil
    available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
    cache_size_mb = max(512, min(4096, available_ram_mb // 2))
except ImportError:
    cache_size_mb = 2048  # Default to 2GB if psutil not available

gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))
gdal.SetConfigOption('GDAL_NUM_THREADS', str(cpu_count))
gdal.SetConfigOption('GDAL_SWATH_SIZE', str(cache_size_mb))
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')

print(f"GDAL Performance Settings: {cpu_count} threads, {cache_size_mb}MB cache")


class ProgressTracker:
    """Helper class to track progress and calculate ETA."""

    def __init__(self, total_items):
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()

    def update(self, completed):
        """Update completed count."""
        self.completed_items = completed

    def get_eta_string(self):
        """Calculate and format ETA."""
        if self.completed_items == 0:
            return "Calculating ETA..."

        elapsed = time.time() - self.start_time
        avg_time_per_item = elapsed / self.completed_items
        remaining_items = self.total_items - self.completed_items
        eta_seconds = avg_time_per_item * remaining_items

        if eta_seconds < 60:
            return f"ETA: {int(eta_seconds)}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            seconds = int(eta_seconds % 60)
            return f"ETA: {minutes}m {seconds}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"ETA: {hours}h {minutes}m"

    def get_progress_string(self):
        """Get formatted progress string."""
        percentage = (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
        return f"{self.completed_items}/{self.total_items} ({percentage:.1f}%) - {self.get_eta_string()}"


class ChartProcessor:
    """Core processing logic using GDAL."""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
    
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def warp_and_cut(self, input_tiff: Path, shapefile: Path, output_tiff: Path, format="GTiff") -> bool:
        """
        Warp and cut the TIFF using the shapefile as a cutline.
        Now converts to RGBA first to ensure consistent color tables and allow Lanczos resampling.
        Supports VRT output to avoid intermediate large files.
        """
        try:
            self.log(f"Processing: {input_tiff.name}")
            self.log(f"  Shapefile: {shapefile.name}")

            # Ensure output directory exists
            output_tiff.parent.mkdir(parents=True, exist_ok=True)

            # Get source resolution to preserve it
            src_ds = gdal.Open(str(input_tiff))
            if not src_ds:
                self.log(f"✗ Could not open {input_tiff.name}")
                return False

            src_gt = src_ds.GetGeoTransform()
            # Get approximate resolution in source CRS
            src_res_x = abs(src_gt[1])
            src_res_y = abs(src_gt[5])
            src_width = src_ds.RasterXSize
            src_height = src_ds.RasterYSize
            self.log(f"  Source: {src_width}x{src_height} pixels, res: {src_res_x:.6f} x {src_res_y:.6f}")
            src_ds = None

            # Step 1: Expand to RGBA
            # If output is VRT, write temp VRT to disk (so it can be referenced)
            # Otherwise use in-memory VRT
            if format == "VRT":
                temp_vrt_name = str(output_tiff.parent / f"{input_tiff.stem}_rgba_temp.vrt")
                cleanup_temp = True
            else:
                temp_vrt_name = f"/vsimem/{input_tiff.stem}_rgba.vrt"
                cleanup_temp = True

            translate_options = gdal.TranslateOptions(
                format="VRT",
                rgbExpand="rgba"
            )

            # Create temp RGBA VRT
            ds_vrt = gdal.Translate(temp_vrt_name, str(input_tiff), options=translate_options)
            ds_vrt = None # Flush/Close

            # Step 2: Warp with Cutline
            creation_opts = []
            if format == 'GTiff':
                 creation_opts = ['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_NEEDED']

            # Calculate target resolution in EPSG:3857 (meters)
            # FAA charts are typically around 42 pixels per nautical mile
            # For EPSG:3857, we want similar visual resolution
            # Use approximately 4-5 meters per pixel for sectional charts
            target_res = 5.0  # meters per pixel in EPSG:3857

            warp_options = gdal.WarpOptions(
                format=format,
                dstSRS='EPSG:3857',
                cutlineDSName=str(shapefile),
                cutlineSRS='EPSG:4326',  # Assume shapefile is in WGS84 (common for FAA)
                cropToCutline=True,
                dstAlpha=True,
                xRes=target_res,
                yRes=target_res,
                resampleAlg=gdal.GRA_Lanczos,
                creationOptions=creation_opts,
                multithread=True,
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
            )

            ds = gdal.Warp(str(output_tiff), temp_vrt_name, options=warp_options)

            # Cleanup temp VRT only if output is NOT VRT
            # (VRT outputs reference the temp file, so we need to keep it)
            if format != "VRT":
                if temp_vrt_name.startswith("/vsimem/"):
                    gdal.Unlink(temp_vrt_name)
                else:
                    try:
                        Path(temp_vrt_name).unlink()
                    except:
                        pass

            if ds:
                # Log output dimensions
                out_width = ds.RasterXSize
                out_height = ds.RasterYSize
                self.log(f"  Output: {out_width}x{out_height} pixels")
                ds = None # Close dataset
                self.log(f"✓ Created {output_tiff.name}")
                return True
            else:
                self.log(f"✗ Failed to create {output_tiff.name}")
                # Check for GDAL error
                err = gdal.GetLastErrorMsg()
                if err:
                    self.log(f"  GDAL Error: {err}")
                return False

        except Exception as e:
            self.log(f"✗ Error processing {input_tiff.name}: {e}")
            # Try to cleanup if failed
            try:
                if temp_vrt_name.startswith("/vsimem/"):
                    gdal.Unlink(temp_vrt_name)
                else:
                    Path(temp_vrt_name).unlink()
            except:
                pass
            return False

    def build_vrt(self, input_files: List[Path], output_vrt: Path) -> bool:
        """Combine multiple TIFFs into a single VRT."""
        try:
            self.log(f"Building VRT: {output_vrt.name}...")
            self.log(f"  Input files: {len(input_files)}")

            # Check if all input files exist
            missing = []
            for f in input_files:
                if not f.exists():
                    missing.append(str(f))
                else:
                    self.log(f"  - {f.name} ✓")

            if missing:
                self.log(f"✗ Missing input files:")
                for m in missing:
                    self.log(f"  - {m}")
                return False

            input_strs = [str(f) for f in input_files]

            # Ensure output directory exists
            output_vrt.parent.mkdir(parents=True, exist_ok=True)

            # gdal.BuildVRT options
            vrt_options = gdal.BuildVRTOptions(
                resampleAlg=gdal.GRA_Lanczos,
                addAlpha=True,
            )

            ds = gdal.BuildVRT(str(output_vrt), input_strs, options=vrt_options)

            if ds:
                ds = None
                self.log(f"✓ VRT created.")
                return True
            else:
                self.log(f"✗ Failed to create VRT - GDAL returned None")
                # Get last GDAL error
                err = gdal.GetLastErrorMsg()
                if err:
                    self.log(f"  GDAL Error: {err}")
                return False

        except Exception as e:
            self.log(f"✗ Error building VRT: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def expand_vrt_to_rgba(self, input_vrt: Path, output_vrt: Path) -> bool:
        """Expand VRT to RGBA using gdal_translate."""
        try:
            self.log(f"Expanding VRT to RGBA: {output_vrt.name}...")
            # gdal.Translate options
            translate_options = gdal.TranslateOptions(
                format='VRT',
                rgbExpand='rgba'
            )
            
            ds = gdal.Translate(str(output_vrt), str(input_vrt), options=translate_options)
            
            if ds:
                ds = None
                self.log(f"✓ Expanded VRT created.")
                return True
            else:
                self.log(f"✗ Failed to expand VRT.")
                return False
                
        except Exception as e:
            self.log(f"✗ Error expanding VRT: {e}")
            return False

    def create_combined_tiff(self, input_vrt: Path, output_tiff: Path) -> bool:
        """Convert VRT to a single large GeoTIFF (BigTIFF) with optimized parallel compression."""
        try:
            self.log(f"Creating Combined GeoTIFF: {output_tiff.name}...")
            self.log("This may take a while for large areas...")

            # Optimized creation options for M4 Mac and modern multi-core systems
            # ZSTD is 2-3x faster than LZW with better compression
            # NUM_THREADS=ALL_CPUS for parallel compression
            # Larger block size (512) for better I/O performance on large files
            creation_options = [
                'TILED=YES',
                'COMPRESS=ZSTD',           # Much faster than LZW
                'ZSTD_LEVEL=6',            # Balance between speed and compression (1-22, default 9)
                'BIGTIFF=YES',
                'PREDICTOR=2',             # Horizontal differencing for better compression
                'NUM_THREADS=ALL_CPUS',    # Use all available CPU cores
                'BLOCKXSIZE=512',          # Larger blocks for better performance
                'BLOCKYSIZE=512'
            ]

            # Progress callback for real-time feedback
            def progress_callback(complete, message, user_data):
                if complete > 0:
                    percent = int(complete * 100)
                    if percent % 10 == 0:  # Log every 10%
                        self.log(f"  Progress: {percent}%")
                return 1  # Return 1 to continue, 0 to cancel

            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=creation_options,
                callback=progress_callback
            )

            ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)

            if ds:
                ds = None
                self.log(f"✓ Created Combined GeoTIFF.")
                return True
            else:
                self.log(f"✗ Failed to create GeoTIFF.")
                return False

        except Exception as e:
            self.log(f"✗ Error creating GeoTIFF: {e}")
            # If ZSTD not available, fallback to LZW with threading
            if "ZSTD" in str(e):
                self.log("ZSTD not available, falling back to LZW with multi-threading...")
                try:
                    creation_options = [
                        'TILED=YES',
                        'COMPRESS=LZW',
                        'BIGTIFF=YES',
                        'PREDICTOR=2',
                        'NUM_THREADS=ALL_CPUS',
                        'BLOCKXSIZE=512',
                        'BLOCKYSIZE=512'
                    ]
                    translate_options = gdal.TranslateOptions(
                        format='GTiff',
                        creationOptions=creation_options
                    )
                    ds = gdal.Translate(str(output_tiff), str(input_vrt), options=translate_options)
                    if ds:
                        ds = None
                        self.log(f"✓ Created Combined GeoTIFF with LZW.")
                        return True
                except Exception as e2:
                    self.log(f"✗ Fallback also failed: {e2}")
            return False

    def generate_tiles(self, input_vrt: Path, output_dir: Path, zoom_levels: str = "", tile_size: int = 1024) -> bool:
        """Generate tiles using gdal2tiles.py with real-time output capturing."""
        try:
            zoom_info = f"Zoom: {zoom_levels}" if zoom_levels else "Zoom: auto"
            self.log(f"Generating Tiles ({zoom_info}, Size: {tile_size})...")
            self.log("This may take a while. Progress will be shown below...")

            # Check availability of gdal2tiles.py
            gdal2tiles_cmd = shutil.which("gdal2tiles.py")
            if not gdal2tiles_cmd:
                gdal2tiles_cmd = "gdal2tiles.py"

            import multiprocessing
            # Reduce CPU count slightly to avoid locking up system
            cpu_count = max(1, multiprocessing.cpu_count() - 2)

            cmd = [
                gdal2tiles_cmd,
                '--processes', str(cpu_count),
                '--webviewer=none',
                '--exclude',
                '--tiledriver=WEBP',
                '--webp-quality=90',
                '--tilesize', str(tile_size),
                '--xyz',
                str(input_vrt),
                str(output_dir)
            ]

            # Only add zoom if specified (otherwise GDAL auto-determines)
            if zoom_levels and zoom_levels.strip():
                cmd.insert(1, '--zoom')
                cmd.insert(2, zoom_levels)
            
            self.log(f"Running command: {' '.join(cmd)}")
            
            # Use PTY to capture output properly (unbuffered)
            import pty
            master, slave = pty.openpty()
            
            process = subprocess.Popen(
                cmd, 
                stdout=slave, 
                stderr=slave,
                close_fds=True
            )
            os.close(slave) # Close slave in parent
            
            # Read from master file descriptor
            import select
            
            output_buffer = ""
            
            while True:
                # Check if process ended
                if process.poll() is not None:
                     # Process ended, check if any data remains
                     r, _, _ = select.select([master], [], [], 0.1)
                     if not r:
                         break

                r, _, _ = select.select([master], [], [], 0.1)
                if r:
                    try:
                        data = os.read(master, 1024).decode(errors='replace')
                        if not data:
                            break
                        
                        # Handle progress bars (carriage returns)
                        # We will append to buffer and log complete lines
                        # For \r, we might want to just log it if it's a progress update
                        
                        output_buffer += data
                        
                        while '\n' in output_buffer or '\r' in output_buffer:
                            # Find first separator
                            n_idx = output_buffer.find('\n')
                            r_idx = output_buffer.find('\r')
                            
                            # Determine which is first (if both exist)
                            if n_idx != -1 and (r_idx == -1 or n_idx < r_idx):
                                line = output_buffer[:n_idx]
                                output_buffer = output_buffer[n_idx+1:]
                                if line.strip(): self.log(f"  > {line.strip()}")
                            elif r_idx != -1:
                                line = output_buffer[:r_idx]
                                output_buffer = output_buffer[r_idx+1:]
                                if line.strip(): self.log(f"  > {line.strip()}")
                                
                    except OSError:
                        break
            
            os.close(master)
            process.wait()
            
            if process.returncode == 0:
                self.log("✓ Tiles generated successfully.")
                return True
            else:
                self.log(f"✗ Tile generation failed (Code {process.returncode})")
                return False
                
        except Exception as e:
            self.log(f"✗ Error generating tiles: {e}")
            return False

    # --- COG & Upload Methods (Merged) ---

    def scan_files(self, in_root: Path) -> List[Path]:
        """Find mosaic TIFFs recursively."""
        self.log(f"Scanning {in_root} for mosaic TIFFs...")
        results = []
        for root, _, files in os.walk(in_root):
             for name in files:
                 if 'mosaic' in name.lower() and name.lower().endswith(('.tif', '.tiff')):
                     results.append(Path(root) / name)
        return results

    def extract_iso_date(self, path: Path) -> Optional[str]:
        """Extract ISO date (YYYY-MM-DD) from filename or path."""
        # Try filename first
        match = re.search(r'\d{4}-\d{2}-\d{2}', path.name)
        if match: return match.group(0)
        
        # Try full path
        match = re.search(r'\d{4}-\d{2}-\d{2}', str(path))
        if match: return match.group(0)
        
        return None

    def convert_to_cog(self, input_path: Path, output_path: Path) -> bool:
        """Convert VRT or TIFF to COG using GDAL with optimized parallel compression."""
        try:
            # Check modification times if output exists
            if output_path.exists():
                # logic tricky with VRTs. Assuming force overwrite or simple time check if input is file
                if input_path.exists() and not input_path.suffix == '.vrt':
                    in_mtime = input_path.stat().st_mtime
                    out_mtime = output_path.stat().st_mtime
                    if out_mtime > in_mtime:
                        self.log(f"Skipping (Up to date): {output_path.name}")
                        return True

            self.log(f"Converting to COG: {output_path.name}...")

            # Optimized COG creation options
            # ZSTD is faster than DEFLATE with similar compression
            # Use ALL_CPUS instead of hardcoded 8 threads
            creation_options = [
                'COMPRESS=ZSTD',
                'ZSTD_LEVEL=6',
                'PREDICTOR=2',
                'BIGTIFF=IF_SAFER',
                'NUM_THREADS=8',
                'BLOCKSIZE=512',
                'RESAMPLING=LANCZOS'
            ]

            # Progress callback for real-time feedback
            def progress_callback(complete, message, user_data):
                if complete > 0:
                    percent = int(complete * 100)
                    if percent % 10 == 0:  # Log every 10%
                        self.log(f"  Progress: {percent}%")
                return 1

            translate_options = gdal.TranslateOptions(
                format='COG',
                creationOptions=creation_options,
                callback=progress_callback
            )

            ds = gdal.Translate(str(output_path), str(input_path), options=translate_options)

            if ds:
                ds = None
                self.log(f"✓ COG Created.")
                return True
            else:
                self.log(f"✗ COG Creation failed.")
                return False

        except Exception as e:
            self.log(f"✗ Error converting to COG: {e}")
            # If ZSTD not available, fallback to DEFLATE with all cores
            if "ZSTD" in str(e):
                self.log("ZSTD not available, falling back to DEFLATE with multi-threading...")
                try:
                    creation_options = [
                        'COMPRESS=DEFLATE',
                        'PREDICTOR=2',
                        'BIGTIFF=IF_SAFER',
                        'NUM_THREADS=8',
                        'BLOCKSIZE=512',
                        'RESAMPLING=LANCZOS'
                    ]
                    translate_options = gdal.TranslateOptions(
                        format='COG',
                        creationOptions=creation_options
                    )
                    ds = gdal.Translate(str(output_path), str(input_path), options=translate_options)
                    if ds:
                        ds = None
                        self.log(f"✓ COG Created with DEFLATE.")
                        return True
                except Exception as e2:
                    self.log(f"✗ Fallback also failed: {e2}")
            return False

    def generate_pmtiles(self, input_vrt: Path, output_pmtiles: Path, zoom_levels: str = "", tile_size: int = 1024) -> bool:
        """Generate PMTiles from a VRT using gdal2tiles and pmtiles convert."""
        try:
            self.log(f"Generating PMTiles: {output_pmtiles.name}...")
            zoom_info = zoom_levels if zoom_levels else "auto"
            self.log(f"  Zoom levels: {zoom_info}, Tile size: {tile_size}")

            # Create a temporary directory for XYZ tiles
            temp_tiles_dir = output_pmtiles.parent / f"_temp_tiles_{output_pmtiles.stem}"
            temp_tiles_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Generate XYZ tiles using gdal2tiles
            self.log("  Step 1/2: Generating XYZ tiles...")
            gdal2tiles_cmd = shutil.which("gdal2tiles.py")
            if not gdal2tiles_cmd:
                gdal2tiles_cmd = "gdal2tiles.py"

            import multiprocessing
            cpu_count = max(1, multiprocessing.cpu_count() - 2)

            cmd_tiles = [
                gdal2tiles_cmd,
                '--processes', str(cpu_count),
                '--webviewer=none',
                '--exclude',
                '--tiledriver=WEBP',
                '--webp-quality=90',
                '--tilesize', str(tile_size),
                '--xyz',
                str(input_vrt),
                str(temp_tiles_dir)
            ]

            # Only add zoom if specified (otherwise GDAL auto-determines)
            if zoom_levels and zoom_levels.strip():
                cmd_tiles.insert(1, '--zoom')
                cmd_tiles.insert(2, zoom_levels)

            self.log(f"  Running: {' '.join(cmd_tiles)}")

            # Run gdal2tiles with PTY for progress
            import pty
            import select
            master, slave = pty.openpty()

            process = subprocess.Popen(
                cmd_tiles,
                stdout=slave,
                stderr=slave,
                close_fds=True
            )
            os.close(slave)

            output_buffer = ""
            while True:
                if process.poll() is not None:
                    r, _, _ = select.select([master], [], [], 0.1)
                    if not r:
                        break

                r, _, _ = select.select([master], [], [], 0.1)
                if r:
                    try:
                        data = os.read(master, 1024).decode(errors='replace')
                        if not data:
                            break

                        output_buffer += data
                        while '\n' in output_buffer or '\r' in output_buffer:
                            n_idx = output_buffer.find('\n')
                            r_idx = output_buffer.find('\r')

                            if n_idx != -1 and (r_idx == -1 or n_idx < r_idx):
                                line = output_buffer[:n_idx]
                                output_buffer = output_buffer[n_idx+1:]
                                if line.strip():
                                    self.log(f"    > {line.strip()}")
                            elif r_idx != -1:
                                line = output_buffer[:r_idx]
                                output_buffer = output_buffer[r_idx+1:]
                                if line.strip():
                                    self.log(f"    > {line.strip()}")
                    except OSError:
                        break

            os.close(master)
            process.wait()

            if process.returncode != 0:
                self.log(f"✗ Tile generation failed (Code {process.returncode})")
                return False

            # Step 2: Convert XYZ tiles to PMTiles
            self.log("  Step 2/2: Converting to PMTiles...")

            pmtiles_cmd = shutil.which("pmtiles")
            if not pmtiles_cmd:
                self.log("✗ pmtiles CLI not found. Install with: go install github.com/protomaps/go-pmtiles/cmd/pmtiles@latest")
                self.log("  Or: pip install pmtiles")
                # Try Python pmtiles as fallback
                try:
                    from pmtiles.convert import convert_xyz_to_pmtiles
                    self.log("  Using Python pmtiles package...")
                    convert_xyz_to_pmtiles(str(temp_tiles_dir), str(output_pmtiles))
                    self.log(f"✓ PMTiles created: {output_pmtiles.name}")
                    # Cleanup temp tiles
                    shutil.rmtree(temp_tiles_dir, ignore_errors=True)
                    return True
                except ImportError:
                    self.log("✗ Python pmtiles package not found either.")
                    return False

            # For raster tiles (WEBP/PNG/JPEG), we need to go through MBTiles
            # Note: tile-join from tippecanoe is for VECTOR tiles only, not raster!

            # Use mb-util to create MBTiles from XYZ directory, then pmtiles convert
            mb_util_cmd = shutil.which("mb-util")
            if mb_util_cmd:
                mbtiles_file = output_pmtiles.parent / f"{output_pmtiles.stem}.mbtiles"

                # mb-util converts XYZ directory to MBTiles
                # Note: mb-util expects TMS scheme by default, but gdal2tiles with --xyz uses XYZ scheme
                # We need to use --scheme=xyz to match
                cmd_mbtiles = [
                    mb_util_cmd,
                    str(temp_tiles_dir),
                    str(mbtiles_file),
                    '--image_format=webp',
                    '--scheme=xyz'
                ]
                self.log(f"  Running: {' '.join(cmd_mbtiles)}")
                result = subprocess.run(cmd_mbtiles, capture_output=True, text=True)

                if result.returncode == 0:
                    self.log(f"  MBTiles created, converting to PMTiles...")
                    # Now convert MBTiles to PMTiles
                    cmd_convert = [
                        pmtiles_cmd, 'convert',
                        str(mbtiles_file),
                        str(output_pmtiles)
                    ]
                    self.log(f"  Running: {' '.join(cmd_convert)}")
                    result = subprocess.run(cmd_convert, capture_output=True, text=True)

                    # Cleanup intermediate MBTiles
                    try:
                        mbtiles_file.unlink()
                    except:
                        pass

                    if result.returncode == 0:
                        self.log(f"✓ PMTiles created: {output_pmtiles.name}")
                        shutil.rmtree(temp_tiles_dir, ignore_errors=True)
                        return True
                    else:
                        self.log(f"✗ PMTiles conversion failed: {result.stderr}")
                        return False
                else:
                    self.log(f"✗ mb-util failed: {result.stderr}")
                    self.log(f"  stdout: {result.stdout}")

            # Final fallback: manual Python-based conversion
            self.log("✗ mb-util not found. Attempting manual conversion...")
            try:
                return self._convert_xyz_to_pmtiles_manual(temp_tiles_dir, output_pmtiles)
            except Exception as e:
                self.log(f"✗ Manual conversion failed: {e}")
                self.log("  Install mb-util: pip install mbutil")
                return False

        except Exception as e:
            self.log(f"✗ Error generating PMTiles: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def _convert_xyz_to_pmtiles_manual(self, tiles_dir: Path, output_pmtiles: Path) -> bool:
        """
        Manual conversion of XYZ tile directory to PMTiles using Python.
        This is a fallback when mb-util is not available.
        """
        import sqlite3
        import json

        self.log("  Creating MBTiles manually from XYZ tiles...")

        # Create intermediate MBTiles file
        mbtiles_file = output_pmtiles.parent / f"{output_pmtiles.stem}.mbtiles"

        try:
            # Create MBTiles database
            conn = sqlite3.connect(str(mbtiles_file))
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    name TEXT,
                    value TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tiles (
                    zoom_level INTEGER,
                    tile_column INTEGER,
                    tile_row INTEGER,
                    tile_data BLOB
                )
            ''')
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles (zoom_level, tile_column, tile_row)
            ''')

            # Set metadata
            cursor.execute("INSERT INTO metadata VALUES ('name', 'tiles')")
            cursor.execute("INSERT INTO metadata VALUES ('type', 'overlay')")
            cursor.execute("INSERT INTO metadata VALUES ('version', '1.0')")
            cursor.execute("INSERT INTO metadata VALUES ('format', 'webp')")

            # Find min/max zoom levels and bounds
            min_zoom = 99
            max_zoom = 0
            tile_count = 0

            # Scan for tiles and insert into database
            for zoom_dir in tiles_dir.iterdir():
                if not zoom_dir.is_dir() or not zoom_dir.name.isdigit():
                    continue

                zoom = int(zoom_dir.name)
                min_zoom = min(min_zoom, zoom)
                max_zoom = max(max_zoom, zoom)

                for x_dir in zoom_dir.iterdir():
                    if not x_dir.is_dir() or not x_dir.name.isdigit():
                        continue

                    x = int(x_dir.name)

                    for tile_file in x_dir.iterdir():
                        if not tile_file.is_file():
                            continue

                        # Extract y from filename (e.g., "123.webp" -> 123)
                        y_str = tile_file.stem
                        if not y_str.isdigit():
                            continue

                        y = int(y_str)

                        # Read tile data
                        tile_data = tile_file.read_bytes()

                        # Convert from XYZ to TMS (flip y)
                        # MBTiles uses TMS scheme where y=0 is at bottom
                        tms_y = (2 ** zoom) - 1 - y

                        cursor.execute(
                            "INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?)",
                            (zoom, x, tms_y, tile_data)
                        )
                        tile_count += 1

                        if tile_count % 1000 == 0:
                            self.log(f"    Processed {tile_count} tiles...")

            # Update metadata with zoom levels
            cursor.execute(f"INSERT INTO metadata VALUES ('minzoom', '{min_zoom}')")
            cursor.execute(f"INSERT INTO metadata VALUES ('maxzoom', '{max_zoom}')")

            conn.commit()
            conn.close()

            self.log(f"  MBTiles created with {tile_count} tiles (zoom {min_zoom}-{max_zoom})")

            # Now convert to PMTiles
            pmtiles_cmd = shutil.which("pmtiles")
            if pmtiles_cmd:
                cmd_convert = [pmtiles_cmd, 'convert', str(mbtiles_file), str(output_pmtiles)]
                self.log(f"  Running: {' '.join(cmd_convert)}")
                result = subprocess.run(cmd_convert, capture_output=True, text=True)

                # Cleanup MBTiles
                try:
                    mbtiles_file.unlink()
                except:
                    pass

                if result.returncode == 0:
                    self.log(f"✓ PMTiles created: {output_pmtiles.name}")
                    shutil.rmtree(tiles_dir, ignore_errors=True)
                    return True
                else:
                    self.log(f"✗ PMTiles conversion failed: {result.stderr}")
                    return False
            else:
                self.log("✗ pmtiles CLI not found")
                self.log(f"  MBTiles file saved at: {mbtiles_file}")
                return False

        except Exception as e:
            self.log(f"✗ Manual MBTiles creation failed: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def sync_to_r2(self, local_dir: Path, remote_dest: str) -> bool:
        """Sync directory to R2 using rclone."""
        self.log(f"Syncing {local_dir} to {remote_dest}...")
        
        rclone_cmd = shutil.which("rclone")
        if not rclone_cmd:
            self.log("✗ rclone not found in PATH.")
            return False
            
        cmd = [
            rclone_cmd, "sync", str(local_dir), remote_dest,
            "-P",
            "--s3-upload-concurrency", "16",
            "--s3-chunk-size", "128M",
            "--buffer-size", "128M",
            "--s3-disable-checksum",
            "--stats", "1s"
        ]
        
        try:
             # Capture output
             import pty
             import select
             master, slave = pty.openpty()
             
             process = subprocess.Popen(
                cmd, 
                stdout=slave, 
                stderr=slave,
                close_fds=True
             )
             os.close(slave)
             
             output_buffer = ""
             while True:
                if process.poll() is not None:
                     r, _, _ = select.select([master], [], [], 0.1)
                     if not r: break

                r, _, _ = select.select([master], [], [], 0.1)
                if r:
                    try:
                        data = os.read(master, 1024).decode(errors='replace')
                        if not data: break
                        
                        output_buffer += data
                        while '\n' in output_buffer or '\r' in output_buffer:
                            n_idx = output_buffer.find('\n')
                            r_idx = output_buffer.find('\r')
                            
                            if n_idx != -1 and (r_idx == -1 or n_idx < r_idx):
                                line = output_buffer[:n_idx]
                                output_buffer = output_buffer[n_idx+1:]
                                if line.strip(): self.log(f"  > {line.strip()}")
                            elif r_idx != -1:
                                line = output_buffer[:r_idx]
                                output_buffer = output_buffer[r_idx+1:]
                                if line.strip() and "Transferred" in line:
                                     self.log(f"  > {line.strip()}") 
                    except OSError:
                        break
                        
             os.close(master)
             process.wait()
             
             if process.returncode == 0:
                 self.log("✓ Upload complete.")
                 return True
             else:
                 self.log(f"✗ Upload failed (Code {process.returncode})")
                 return False
                 
        except Exception as e:
            self.log(f"✗ Error running rclone: {e}")
            return False




class BatchProcessor:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.file_index = {} # Key -> Path
        self.dole_data = defaultdict(list)
        self.vrt_library = {}  # {location: {date: {'vrt_path': Path, ...}}}
        self.vrt_library_dir = None
        self.abbreviations = {
            "hawaiian_is": "hawaiian_islands",
            "dallas_ft_worth": "dallas_ft_worth", # Identity
            "mariana_islands_inset": "mariana_islands", # Maybe?
        }

    def log(self, msg):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

    def normalize_name(self, name):
        """Normalize chart name consistently."""
        # Replace spaces, dashes, dots with underscores
        norm = name.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_')
        # Clean up double underscores
        while '__' in norm:
            norm = norm.replace('__', '_')
        return self.abbreviations.get(norm, norm)

    def scan_source_dir(self, source_dir: Path):
        """Recursively scan directory for charts and index them."""
        self.log(f"Scanning directory: {source_dir}...")
        self.file_index = {}
        
        count = 0
        try:
            for root, _, files in os.walk(source_dir):
                for name in files:
                    if name.lower().endswith(('.zip', '.tif', '.tiff')):
                        current_path = Path(root) / name
                        # Normalize the entire stem before splitting to handle "Seattle SEC 105.tif"
                        stem = self.normalize_name(current_path.stem)
                        parts = stem.split('_')
                        
                        if parts[-1].isdigit():
                            edition = parts[-1]
                            location = "_".join(parts[:-1])
                            key = ('EDITION', location, edition)
                            if key not in self.file_index: self.file_index[key] = []
                            self.file_index[key].append(current_path)
                            count += 1
                        elif parts[0].isdigit() and len(parts[0]) >= 14:
                            timestamp = parts[0]
                            key = ('TS', timestamp)
                            if key not in self.file_index: self.file_index[key] = []
                            self.file_index[key].append(current_path)
                            count += 1

            self.log(f"Indexed {count} files across {len(self.file_index)} unique keys.")
            
        except Exception as e:
            self.log(f"Error scanning directory: {e}")

    def load_dole_data(self, csv_file: Path):
        self.log(f"Loading CSV: {csv_file.name}...")
        self.dole_data = defaultdict(list)
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date = row.get('date')
                    if not date: continue
                    
                    link = row.get('download_link', '')
                    timestamp = None
                    if 'web.archive.org/web/' in link:
                        match = re.search(r'/web/(\d{14})/', link)
                        if match:
                            timestamp = match.group(1)
                    row['wayback_ts'] = timestamp
                    self.dole_data[date].append(row)
            self.log(f"Loaded {len(self.dole_data)} dates.")
        except Exception as e:
            self.log(f"Error loading CSV: {e}")

    def find_files(self, location, edition, timestamp):
        results = []
        
        # 1. Try Timestamp matches (High confidence)
        if timestamp:
            res = self.file_index.get(('TS', timestamp))
            if res: results.extend(res)
            
        # 2. Try variations of location
        norm_loc = self.normalize_name(location)
        
        # Build variations to try
        # 1. Exact match
        # 2. With common suffixes
        # 3. Primary suffixes (SEC, TAC)
        # 4. Long forms (Sectional, Terminal)
        candidate_locs = [
            norm_loc,
            f"{norm_loc}_sec",
            f"{norm_loc}_tac",
            f"{norm_loc}_sectional",
            f"{norm_loc}_terminal",
        ]
        
        # Also try variants (North/South) combined with suffixes
        variants = ["north", "south", "east", "west"]
        for v in variants:
            var_base = f"{norm_loc}_{v}"
            candidate_locs.append(var_base)
            candidate_locs.append(f"{var_base}_sec")
            candidate_locs.append(f"{var_base}_tac")

        # Search each candidate
        for loc in candidate_locs:
            key = ('EDITION', loc, edition)
            res = self.file_index.get(key)
            if res:
                results.extend(res)

        # Deduplicate
        return list(set(results))
        
    def find_shapefile(self, location, shape_dir: Path):
        norm_loc = self.normalize_name(location)
        candidates = list(shape_dir.glob("*.shp"))
        for shp in candidates:
            shp_norm = self.normalize_name(shp.stem)
            if shp_norm == norm_loc: return shp
            # Partial match for "Western Aleutian..."
            if norm_loc in shp_norm: return shp # Naive, handles east/west?
        return None

    def unzip_file(self, zip_path: Path, output_dir: Path) -> List[Path]:
        """Unzips all TIFFs and TFWs, returns list of TIFF paths."""
        extracted_tiffs = []
        try:
            if not zip_path.exists():
                self.log(f"  [ERROR] Zip file not found: {zip_path}")
                return []
                
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find the tif files inside
                tiffs = [n for n in z.namelist() if n.lower().endswith(('.tif', '.tiff'))]
                if not tiffs: 
                    self.log(f"  [ERROR] No TIFFs found in {zip_path.name}")
                    return []
                
                # We also want to extract .tfw files for these tiffs
                # Usually name matches stem.
                tfws = [n for n in z.namelist() if n.lower().endswith(('.tfw', '.tifw'))]
                
                extract_path = output_dir / zip_path.stem
                extract_path.mkdir(parents=True, exist_ok=True)
                
                for target in tiffs:
                    z.extract(target, path=extract_path)
                    full_path = extract_path / target
                    if full_path.exists():
                        extracted_tiffs.append(full_path)
                        
                        # Check for matching tfw and extract if exists
                        target_stem = Path(target).stem
                        for tfw in tfws:
                            if Path(tfw).stem == target_stem:
                                z.extract(tfw, path=extract_path)
                                self.log(f"    Extracted companion world file: {tfw}")

                return extracted_tiffs
        except Exception as e:
            self.log(f"Error unzipping {zip_path.name}: {e}")
            return []

    def build_vrt_library(self, source_dir: Path, shape_dir: Path, vrt_library_dir: Path, processor, progress_callback=None):
        """
        Build/update the VRT library by warping all source files with shapefiles.
        Reuses existing VRTs and only creates missing ones.

        Returns:
            vrt_library dict structure
        """
        try:
            self.vrt_library_dir = vrt_library_dir
            vrt_library_dir.mkdir(parents=True, exist_ok=True)

            # Try to load existing cache
            cache_file = vrt_library_dir / "vrt_library_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_library = json.load(f)
                self.vrt_library = self._deserialize_vrt_library(cached_library)
                self.log(f"Loaded cached VRT library with {len(self.vrt_library)} locations")

            # Count total items to process
            total_items = sum(len(records) for records in self.dole_data.values())
            current_item = 0

            # Iterate through CSV data
            for date, records in sorted(self.dole_data.items()):
                for rec in records:
                    location = rec.get('location', '')
                    edition = rec.get('edition', '')
                    timestamp = rec.get('wayback_ts')

                    norm_loc = self.normalize_name(location)
                    vrt_filename = f"{norm_loc}_{date}.vrt"
                    vrt_path = vrt_library_dir / vrt_filename

                    current_item += 1

                    # Check if VRT already exists
                    if vrt_path.exists():
                        if norm_loc not in self.vrt_library:
                            self.vrt_library[norm_loc] = {}
                        self.vrt_library[norm_loc][date] = {
                            'vrt_path': vrt_path,
                            'source_files': [],
                            'created': datetime.datetime.fromtimestamp(vrt_path.stat().st_mtime).isoformat(),
                            'shapefile': ''
                        }
                        if progress_callback:
                            progress_callback(current_item, total_items, f"Cached: {norm_loc} {date}")
                        continue

                    # Find shapefile
                    shp_path = self.find_shapefile(location, shape_dir)
                    if not shp_path:
                        if progress_callback:
                            progress_callback(current_item, total_items, f"Skip {norm_loc} {date} (no shapefile)")
                        continue

                    # Find source files
                    found_files = self.find_files(location, edition, timestamp)
                    if not found_files:
                        if progress_callback:
                            progress_callback(current_item, total_items, f"Skip {norm_loc} {date} (no files)")
                        continue

                    # Process files (handle ZIP extraction)
                    temp_warped_files = []
                    for src_file in found_files:
                        if src_file.suffix.lower() == '.zip':
                            temp_extract = vrt_library_dir / "temp_extract"
                            extracted = self.unzip_file(src_file, temp_extract)
                            tiffs_to_process = extracted
                        else:
                            tiffs_to_process = [src_file]

                        # Warp each TIFF
                        for tiff in tiffs_to_process:
                            temp_vrt_name = f"{norm_loc}_{date}_{tiff.stem}_temp.vrt"
                            temp_vrt_path = vrt_library_dir / temp_vrt_name

                            if processor.warp_and_cut(tiff, shp_path, temp_vrt_path, format="VRT"):
                                temp_warped_files.append(temp_vrt_path)

                    # Combine or finalize VRT
                    if len(temp_warped_files) > 1:
                        # Multiple files (e.g., North/South), combine into single VRT
                        if processor.build_vrt(temp_warped_files, vrt_path):
                            # Clean up intermediate VRTs
                            for temp_vrt in temp_warped_files:
                                try:
                                    temp_vrt.unlink()
                                except:
                                    pass
                        else:
                            if progress_callback:
                                progress_callback(current_item, total_items, f"Error {norm_loc} {date} (VRT build failed)")
                            continue
                    elif len(temp_warped_files) == 1:
                        # Single file, just rename
                        try:
                            temp_warped_files[0].rename(vrt_path)
                        except Exception as e:
                            self.log(f"  Error renaming VRT: {e}")
                            continue
                    else:
                        if progress_callback:
                            progress_callback(current_item, total_items, f"Error {norm_loc} {date} (no warped files)")
                        continue

                    # Add to library
                    if norm_loc not in self.vrt_library:
                        self.vrt_library[norm_loc] = {}
                    self.vrt_library[norm_loc][date] = {
                        'vrt_path': vrt_path,
                        'source_files': [f.name for f in found_files],
                        'created': datetime.datetime.now().isoformat(),
                        'shapefile': shp_path.name
                    }

                    if progress_callback:
                        progress_callback(current_item, total_items, f"Built: {norm_loc} {date}")

            # Save cache
            cache_data = self._serialize_vrt_library(self.vrt_library)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            self.log(f"VRT library complete: {len(self.vrt_library)} locations")
            return self.vrt_library

        except Exception as e:
            self.log(f"Error building VRT library: {e}")
            import traceback
            self.log(traceback.format_exc())
            return {}

    def build_date_matrix(self, shape_dir: Path):
        """
        Build date matrix with fallback logic.
        For each date, find most recent VRT for each location (fallback if needed).

        Returns:
            date_matrix: {date: {location: {'vrt': Path, 'fallback': bool, 'actual_date': str, 'shp': Path}}}
        """
        try:
            date_matrix = {}

            # Get all unique dates from CSV (sorted)
            all_dates = sorted(self.dole_data.keys())

            # Get all locations from shapefiles
            all_locations = []
            for shp in shape_dir.glob("*.shp"):
                norm_loc = self.normalize_name(shp.stem)
                all_locations.append(norm_loc)

            # Build matrix
            for date in all_dates:
                date_matrix[date] = {}

                for location in all_locations:
                    # Find shapefile
                    shp_path = self.find_shapefile(location, shape_dir)
                    if not shp_path:
                        continue

                    # Find most recent VRT where VRT_date <= current_date
                    if location in self.vrt_library:
                        available_dates = [
                            d for d in self.vrt_library[location].keys()
                            if d <= date
                        ]

                        if available_dates:
                            # Get most recent
                            most_recent = sorted(available_dates)[-1]
                            vrt_info = self.vrt_library[location][most_recent]

                            date_matrix[date][location] = {
                                'vrt': vrt_info['vrt_path'],
                                'fallback': (most_recent != date),
                                'actual_date': most_recent,
                                'shp': shp_path
                            }

            return date_matrix

        except Exception as e:
            self.log(f"Error building date matrix: {e}")
            import traceback
            self.log(traceback.format_exc())
            return {}

    def _serialize_vrt_library(self, library):
        """Convert Path objects to strings for JSON serialization."""
        result = {}
        for loc, dates in library.items():
            result[loc] = {}
            for date, info in dates.items():
                result[loc][date] = {
                    'vrt_path': str(info['vrt_path']),
                    'source_files': info.get('source_files', []),
                    'created': info.get('created', ''),
                    'shapefile': info.get('shapefile', '')
                }
        return result

    def _deserialize_vrt_library(self, cache_data):
        """Convert strings back to Path objects."""
        result = {}
        for loc, dates in cache_data.items():
            result[loc] = {}
            for date, info in dates.items():
                result[loc][date] = {
                    'vrt_path': Path(info['vrt_path']),
                    'source_files': info.get('source_files', []),
                    'created': info.get('created', ''),
                    'shapefile': info.get('shapefile', '')
                }
        return result


class BatchReviewWindow(tk.Toplevel):
    def __init__(self, parent, review_data, shape_dir, process_callback):
        super().__init__(parent)
        self.title("Batch Review")
        self.geometry("900x600")
        self.review_data = review_data # Dict: Date -> List of Items
        self.shape_dir = shape_dir
        self.process_callback = process_callback
        self.columns = []
        self.location_columns = []
        self.location_by_id = {}
        self.location_id_by_label = {}
        self.row_ids = {}
        self.current_cell = {}
        
        self.create_ui()
        
    def create_ui(self):
        # Top: Instructions
        lbl = ttk.Label(
            self,
            text="Review charts by date (rows) and location (columns). Right-click a cell to add/remove files or change shapefiles.",
            wraplength=800
        )
        lbl.pack(pady=10)
        
        # Treeview
        self.build_location_columns()
        self.columns = ["date"] + [col["id"] for col in self.location_columns]
        self.tree = ttk.Treeview(self, columns=self.columns, show="headings", selectmode="browse")

        self.tree.heading("date", text="Date")
        self.tree.column("date", width=120, minwidth=100, stretch=False)

        for col in self.location_columns:
            self.tree.heading(col["id"], text=col["label"])
            self.tree.column(col["id"], width=220, minwidth=140, stretch=True)
        
        # Scrollbars
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
        
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Populate
        self.populate_tree()
        
        # Context Menu
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Add File to Item...", command=self.add_file)
        self.menu.add_command(label="Change Shapefile...", command=self.change_shapefile)
        self.menu.add_separator()
        self.menu.add_command(label="Remove Selected Item", command=self.remove_item)
        
        self.tree.bind("<Button-2>", self.show_context_menu) # Mac Right Click
        self.tree.bind("<Button-3>", self.show_context_menu) # Windows/Linux Right Click
        self.tree.bind("<Button-1>", self.capture_cell)
        
        # Bottom Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=10)
        ttk.Button(btn_frame, text="RUN BATCH PROCESS", command=self.run_batch).pack(side=tk.RIGHT, padx=10)

    def build_location_columns(self):
        locations = sorted({
            item['loc']
            for items in self.review_data.values()
            for item in items
            if item.get('loc')
        })
        self.location_columns = []
        for i, loc in enumerate(locations):
            col_id = f"loc_{i}"
            self.location_columns.append({"id": col_id, "label": loc})
            self.location_by_id[col_id] = loc
            self.location_id_by_label[loc] = col_id

    def format_files(self, files):
        if not files:
            return "MISSING"
        return ", ".join([f.name for f in files])

    def populate_tree(self):
        # Clear
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.row_ids = {}

        # Sort dates
        dates = sorted(self.review_data.keys())
        items_by_date = {}
        for date in dates:
            items_by_date[date] = {item['loc']: item for item in self.review_data[date]}

        for date in dates:
            values = [date]
            for col in self.location_columns:
                loc = col["label"]
                item = items_by_date.get(date, {}).get(loc)
                if item and item.get('files'):
                    # Check if this is a fallback
                    if item.get('fallback', False):
                        actual_date = item.get('actual_date', 'unknown')
                        file_name = item['files'][0].name if item['files'] else 'unknown'
                        display = f"⟳ {actual_date} [Fallback]"
                    else:
                        display = f"✓ {date} [Source]"
                    values.append(display)
                else:
                    values.append("—")  # No data
            row_id = self.tree.insert("", tk.END, values=values)
            self.row_ids[date] = row_id
                
    def capture_cell(self, event):
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if row_id:
            self.tree.selection_set(row_id)
            self.current_cell = {"row_id": row_id, "col_id": col_id}

    def get_selected_item(self):
        row_id = self.current_cell.get("row_id")
        col_id = self.current_cell.get("col_id")
        if not row_id or not col_id:
            return None, None, None

        if not col_id.startswith("#"):
            return None, None, None

        col_index = int(col_id.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(self.columns):
            return None, None, None

        col_key = self.columns[col_index]
        if col_key == "date":
            return None, None, None

        values = self.tree.item(row_id, "values")
        if not values:
            return None, None, None
        date_text = values[0]
        loc = self.location_by_id.get(col_key)
        if not loc:
            return None, None, None

        items = self.review_data.get(date_text, [])
        for item in items:
            if item['loc'] == loc:
                return date_text, item, row_id

        return None, None, None

    def add_file(self):
        date, item, node_id = self.get_selected_item()
        if not item: return
        
        f = filedialog.askopenfilename(title="Select Chart File", filetypes=[("Chart", "*.zip *.tif *.tiff")])
        if f:
            path = Path(f)
            item['files'].append(path)
            self.update_cell(date, item['loc'])

    def change_shapefile(self):
        date, item, node_id = self.get_selected_item()
        if not item: return
        
        f = filedialog.askopenfilename(title="Select Shapefile", initialdir=self.shape_dir, filetypes=[("Shapefile", "*.shp")])
        if f:
            path = Path(f)
            item['shp'] = path
            self.update_cell(date, item['loc'])

    def remove_item(self):
        date, item, node_id = self.get_selected_item()
        if not item: 
            # maybe selected a date?
            return
            
        if messagebox.askyesno("Confirm", f"Remove {item['loc']} from processing?"):
            self.review_data[date].remove(item)
            self.update_cell(date, item['loc'])


    def show_context_menu(self, event):
        try:
            row_id = self.tree.identify_row(event.y)
            col_id = self.tree.identify_column(event.x)
            if row_id:
                self.tree.selection_set(row_id)
                self.current_cell = {"row_id": row_id, "col_id": col_id}
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def update_cell(self, date, loc):
        self.populate_tree()

    def find_fallback_item(self, dates, items_by_date, target_date, loc):
        try:
            idx = dates.index(target_date)
        except ValueError:
            return None
        for i in range(idx - 1, -1, -1):
            prev_date = dates[i]
            prev_item = items_by_date.get(prev_date, {}).get(loc)
            if prev_item and prev_item.get('files'):
                return {"date": prev_date, "item": prev_item}
        return None

    def run_batch(self):
        # Validation ?
        self.process_callback(self.review_data)
        self.destroy()


class UnifiedAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FAA Chart Tool - Unified")
        self.root.geometry("1000x800")
        
        self.processor = ChartProcessor(log_callback=self.safe_log)
        
        # State
        self.charts: List[Dict] = [] # List of dicts: {'path': Path, 'shapefile': Path, 'status': str}
        self.output_dir = tk.StringVar(value="/Volumes/projects/trial")
        self.zoom_levels = tk.StringVar(value="")  # Blank = auto-determine by GDAL
        self.tile_size = tk.StringVar(value="1024")  # Default tile size
        self.output_format = tk.StringVar(value="geotiff")  # Options: 'tiles', 'geotiff', 'cog', 'both', 'pmtiles'
        self.optimize_vrt = tk.BooleanVar(value=True) # Pipeline optimization
        self.r2_enabled = tk.BooleanVar(value=False)
        self.r2_destination = tk.StringVar(value="r2:sectionals")
        
        self.processing = False
        self.log_queue = queue.Queue()
        
        # Batch vars
        self.batch_source_dir = tk.StringVar(value="/Volumes/drive/newrawtiffs")
        self.batch_csv_file = tk.StringVar(value="master_dole.csv")
        self.batch_shape_dir = tk.StringVar(value="shapefiles")
        self.batch_vrt_lib_dir = tk.StringVar(value="/Volumes/projects/warped_vrts")
        self.batch_date_filter = tk.StringVar()
        
        self.batch_processor = BatchProcessor(log_callback=self.safe_log)
        
        # COG/Upload vars (Tab 3 specific)
        self.cog_in_root = tk.StringVar(value="/Volumes/projects/trial")
        self.cog_out_dir = tk.StringVar(value="/Users/ryanhemenway/Desktop/upload")
        # self.r2_dest reused
        # self.cog_processor removed, using self.processor

        self.setup_ui()
        self.check_log_queue()
        
    def setup_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Tab 1: Manual
        self.tab_manual = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab_manual, text="Manual Mode")
        self.setup_manual_tab()
        
        # Tab 2: Batch
        self.tab_batch = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab_batch, text="Batch Mode")
        self.setup_batch_tab()
        
        # Tab 3: COG & Upload
        self.tab_cog = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab_cog, text="COG & Upload")
        self.setup_cog_tab()
        
        # --- Progress Section ---
        progress_frame = ttk.LabelFrame(main, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 5))

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(fill=tk.X)

        # --- Bottom Section: Log ---
        log_frame = ttk.LabelFrame(main, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_manual_tab(self):
        # --- Top Section: Chart Manager ---
        top_frame = ttk.LabelFrame(self.tab_manual, text="Chart Manager", padding="5")
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview
        columns = ("chart", "shapefile", "status")
        self.tree = ttk.Treeview(top_frame, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("chart", text="Note: Input Chart File")
        self.tree.heading("shapefile", text="Linked Shapefile (Cutline)")
        self.tree.heading("status", text="Status")
        
        self.tree.column("chart", width=300)
        self.tree.column("shapefile", width=300)
        self.tree.column("status", width=100)
        
        scrollbar = ttk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Toolbar
        toolbar = ttk.Frame(self.tab_manual)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="Add Charts...", command=self.add_charts).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Link Shapefile...", command=self.link_shapefile).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Auto-Link (Name Match)", command=self.auto_link).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Remove Selected", command=self.remove_charts).pack(side=tk.LEFT, padx=5)
        
        self.setup_processing_options(self.tab_manual, self.start_processing)

    def setup_batch_tab(self):
        frame = ttk.Frame(self.tab_batch)
        frame.pack(fill=tk.X, pady=10)
        
        # File Inputs
        grid_opts = {'padx': 5, 'pady': 5, 'sticky': tk.W}
        
        ttk.Label(frame, text="Source Directory:").grid(row=0, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.batch_source_dir, width=40).grid(row=0, column=1, **grid_opts)
        ttk.Button(frame, text="...", command=lambda: self.browse_dir(self.batch_source_dir)).grid(row=0, column=2, **grid_opts)
        
        ttk.Label(frame, text="Master Dole (csv):").grid(row=1, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.batch_csv_file, width=40).grid(row=1, column=1, **grid_opts)
        ttk.Button(frame, text="...", command=lambda: self.browse_file(self.batch_csv_file)).grid(row=1, column=2, **grid_opts)
        
        ttk.Label(frame, text="Shapefiles Dir:").grid(row=2, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.batch_shape_dir, width=40).grid(row=2, column=1, **grid_opts)
        ttk.Button(frame, text="...", command=lambda: self.browse_dir(self.batch_shape_dir)).grid(row=2, column=2, **grid_opts)

        ttk.Label(frame, text="VRT Library Dir:").grid(row=3, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.batch_vrt_lib_dir, width=40).grid(row=3, column=1, **grid_opts)
        ttk.Button(frame, text="...", command=lambda: self.browse_dir(self.batch_vrt_lib_dir)).grid(row=3, column=2, **grid_opts)

        ttk.Label(frame, text="Filter Date (YYYY-MM-DD):").grid(row=4, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.batch_date_filter, width=20).grid(row=4, column=1, **grid_opts)
        ttk.Label(frame, text="(Leave empty for all)").grid(row=4, column=2, **grid_opts)

        self.setup_processing_options(self.tab_batch, self.start_batch_processing)

    def setup_processing_options(self, parent, command):
        # --- Options ---
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output Directory
        ttk.Label(options_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(options_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(options_frame, text="Browse...", command=self.browse_output).grid(row=0, column=2, padx=5, pady=5)
        
        # Tiling Options
        ttk.Label(options_frame, text="Tile Zoom Levels:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(options_frame, textvariable=self.zoom_levels, width=20).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(options_frame, text="(blank = auto, or e.g. 0-11)").grid(row=1, column=2, sticky=tk.W, padx=5)

        # Tile Size Option
        tile_size_frame = ttk.Frame(options_frame)
        tile_size_frame.grid(row=1, column=3, sticky=tk.W, padx=15)
        ttk.Label(tile_size_frame, text="Tile Size:").pack(side=tk.LEFT)
        tile_size_combo = ttk.Combobox(tile_size_frame, textvariable=self.tile_size, values=["256", "512", "1024", "2048"], width=6)
        tile_size_combo.pack(side=tk.LEFT, padx=5)
        
        # Output Format Options
        ttk.Label(options_frame, text="Output Format:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        format_frame = ttk.Frame(options_frame)
        format_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(format_frame, text="Generate Tiles", variable=self.output_format, value="tiles").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="PMTiles", variable=self.output_format, value="pmtiles").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Combined GeoTIFF", variable=self.output_format, value="geotiff").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="COG (Optimized)", variable=self.output_format, value="cog").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Both", variable=self.output_format, value="both").pack(side=tk.LEFT, padx=5)
        
        # Optimization Checkbox
        ttk.Checkbutton(options_frame, text="Optimize Pipeline (Use VRTs)", variable=self.optimize_vrt).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # R2 Upload
        r2_frame = ttk.LabelFrame(options_frame, text="Upload", padding=5)
        r2_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Checkbutton(r2_frame, text="Sync to R2 after processing", variable=self.r2_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Entry(r2_frame, textvariable=self.r2_destination, width=30).pack(side=tk.LEFT, padx=5)
        
        # Process Button
        self.btn_process = ttk.Button(parent, text="START PROCESSING", command=command) # Store ref
        self.btn_process.pack(fill=tk.X, pady=10)

    def setup_cog_tab(self):
        frame = ttk.Frame(self.tab_cog)
        frame.pack(fill=tk.X, pady=10)
        
        grid_opts = {'padx': 5, 'pady': 5, 'sticky': tk.W}
        
        ttk.Label(frame, text="Search Root (Mosaic TIFFs):").grid(row=0, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.cog_in_root, width=40).grid(row=0, column=1, **grid_opts)
        ttk.Button(frame, text="...", command=lambda: self.browse_dir(self.cog_in_root)).grid(row=0, column=2, **grid_opts)
        
        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.cog_out_dir, width=40).grid(row=1, column=1, **grid_opts)
        ttk.Button(frame, text="...", command=lambda: self.browse_dir(self.cog_out_dir)).grid(row=1, column=2, **grid_opts)
        
        ttk.Label(frame, text="R2 Destination:").grid(row=2, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.r2_destination, width=40).grid(row=2, column=1, **grid_opts)
        
        # Actions
        btn_frame = ttk.Frame(self.tab_cog)
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(btn_frame, text="START PIPELINE (Convert + Upload)", command=self.start_cog_pipeline, width=30).pack(pady=5)

    def browse_file(self, var):
        f = filedialog.askopenfilename()
        if f: var.set(f)
    
    def browse_dir(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)

        
    def safe_log(self, message):
        self.log_queue.put(message)

    def update_progress(self, current, total, status_text="Processing..."):
        """Thread-safe progress update with ETA calculation."""
        def _update():
            if total > 0:
                percentage = (current / total) * 100
                self.progress_bar['value'] = percentage
                self.progress_label['text'] = status_text
            else:
                self.progress_bar['value'] = 0
                self.progress_label['text'] = status_text
        self.root.after(0, _update)

    def reset_progress(self):
        """Reset progress bar to initial state."""
        def _reset():
            self.progress_bar['value'] = 0
            self.progress_label['text'] = "Ready"
        self.root.after(0, _reset)

    def check_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.check_log_queue)
        
    def add_charts(self):
        files = filedialog.askopenfilenames(title="Select Chart TIFFs", filetypes=[("TIFF Files", "*.tif")])
        for f in files:
            path = Path(f)
            # Check duplicates
            if any(c['path'] == path for c in self.charts):
                continue
            
            item = {'path': path, 'shapefile': None, 'status': 'Pending'}
            self.charts.append(item)
            self.refresh_tree()
            
    def remove_charts(self):
        selected = self.tree.selection()
        if not selected:
            return
            
        indices = [self.tree.index(s) for s in selected]
        # Remove in reverse order
        for i in sorted(indices, reverse=True):
            del self.charts[i]
        self.refresh_tree()
        
    def link_shapefile(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select Charts", "Please select chart(s) to link.")
            return
            
        shp_file = filedialog.askopenfilename(title="Select Shapefile", filetypes=[("Shapefile", "*.shp")])
        if not shp_file:
            return
            
        shp_path = Path(shp_file)
        
        for item_id in selected:
            idx = self.tree.index(item_id)
            self.charts[idx]['shapefile'] = shp_path
            
        self.refresh_tree()
        self.safe_log(f"Linked {shp_path.name} to {len(selected)} charts.")

    def auto_link(self):
        """Try to find shapefiles in a folder that match chart names."""
        if not self.charts:
            return
        
        shp_dir = filedialog.askdirectory(title="Select Directory Containing Shapefiles")
        if not shp_dir:
            return
        
        count = 0
        shp_dir_path = Path(shp_dir)
        
        for chart in self.charts:
            if chart['shapefile']: continue # Skip already linked
            
            # Simple heuristic: Chart name (e.g. "Seattle_SEC_...") starts with "Seattle"
            # Shapefile might be "Seattle.shp"
            
            # Try exact match of stem prefix
            chart_stem = chart['path'].name.lower()
            
            # Naive approach: check if any shapefile name is effectively contained in chart name
            # Or assume standard naming convention "Name SEC.tif" -> "Name.shp"
            
            # Let's clean the name: "Seattle SEC 105.tif" -> "Seattle"
            parts = chart['path'].stem.split('_') # Assuming underscores? Or spaces?
            # User's sample code had underscores or spaces. Let's try splitting by typical separators
            
            candidates = list(shp_dir_path.glob("*.shp"))
            found = None
            
            for shp in candidates:
                # Check if shapefile stem appears in chart filename
                # e.g. shp="Seattle", chart="Seattle SEC.tif"
                if shp.stem.lower() in chart['path'].name.lower():
                    found = shp
                    break
            
            if found:
                chart['shapefile'] = found
                count += 1
                
        self.refresh_tree()
        self.safe_log(f"Auto-linked {count} charts.")

    def browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)
            
    def refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        for c in self.charts:
            s_name = c['shapefile'].name if c['shapefile'] else "❌ Not Linked"
            self.tree.insert("", tk.END, values=(c['path'].name, s_name, c['status']))

    def start_processing(self):
        if self.processing: return

        # Validation
        if not self.charts:
            messagebox.showwarning("Warning", "No charts added.")
            return
        if not self.output_dir.get():
            messagebox.showwarning("Warning", "Select output directory.")
            return

        # Check all have shapefiles
        missing = [c['path'].name for c in self.charts if not c['shapefile']]
        if missing:
            messagebox.showwarning("Warning", f"Missing shapefiles for:\n{', '.join(missing)}")
            return

        self.processing = True
        self.btn_process.config(state='disabled')
        self.reset_progress()  # Reset progress bar at start

        thread = threading.Thread(target=self.run_process_thread, daemon=True)
        thread.start()
        
    def run_process_thread(self):
        try:
            out_dir = Path(self.output_dir.get())
            temp_dir = out_dir / "warped_temp"
            tiles_dir = out_dir / "tiles"
            vrt_file = out_dir / "combined.vrt"

            out_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(exist_ok=True)

            # Initialize progress tracker
            total_steps = len(self.charts)
            tracker = ProgressTracker(total_steps)

            # 1. Warp/Cut individual files
            processed_files = []

            for i, chart in enumerate(self.charts):
                chart['status'] = "Processing..."
                tracker.update(i)
                self.update_progress(i, total_steps, f"Processing charts - {tracker.get_progress_string()}")

                out_name = f"{chart['path'].stem}_warped.tif"
                out_path = temp_dir / out_name

                success = self.processor.warp_and_cut(chart['path'], chart['shapefile'], out_path)

                if success:
                    processed_files.append(out_path)
                    chart['status'] = "Done"
                else:
                     chart['status'] = "Failed"

            # Update to 100% for warping phase
            tracker.update(total_steps)
            self.update_progress(total_steps, total_steps, f"Chart warping complete - {tracker.get_progress_string()}")

            if not processed_files:
                self.safe_log("No files processed successfully. Stopping.")
                self.reset_progress()
                return

            # 2. Build VRT
            self.safe_log("Building VRT...")
            self.update_progress(0, 0, "Building VRT mosaic...")
            if self.processor.build_vrt(processed_files, vrt_file):
                self.safe_log(f"VRT created at {vrt_file}")
            else:
                self.safe_log("VRT creation failed. Stopping.")
                self.reset_progress()
                return

            # 3. Generate output based on selected format
            output_fmt = self.output_format.get()

            if output_fmt in ["geotiff", "both"]:
                tiff_file = out_dir / "combined.tif"
                self.safe_log("Creating Combined GeoTIFF...")
                self.update_progress(0, 0, "Creating combined GeoTIFF...")
                self.processor.create_combined_tiff(vrt_file, tiff_file)

            if output_fmt in ["cog", "both"]:
                cog_file = out_dir / "combined_cog.tif"
                self.safe_log("Creating Optimized COG...")
                self.update_progress(0, 0, "Creating COG...")
                self.processor.convert_to_cog(vrt_file, cog_file)

            if output_fmt in ["tiles", "both"]:
                zoom = self.zoom_levels.get()
                tile_size = int(self.tile_size.get() or 1024)
                zoom_info = zoom if zoom else "auto"
                self.safe_log(f"Generating Tiles (zoom: {zoom_info}, size: {tile_size})...")
                self.update_progress(0, 0, f"Generating tiles (zoom {zoom_info})...")
                self.processor.generate_tiles(vrt_file, tiles_dir, zoom, tile_size)

            if output_fmt == "pmtiles":
                zoom = self.zoom_levels.get()
                tile_size = int(self.tile_size.get() or 1024)
                zoom_info = zoom if zoom else "auto"
                pmtiles_dir = out_dir / "pmtiles"
                pmtiles_dir.mkdir(parents=True, exist_ok=True)
                pmtiles_file = pmtiles_dir / "combined.pmtiles"
                self.safe_log(f"Generating PMTiles (zoom: {zoom_info}, size: {tile_size})...")
                self.update_progress(0, 0, f"Generating PMTiles (zoom {zoom_info})...")
                self.processor.generate_pmtiles(vrt_file, pmtiles_file, zoom, tile_size)

            self.update_progress(100, 100, "Processing complete!")
            self.safe_log("ALL TASKS COMPLETED.")
            messagebox.showinfo("Done", "Processing Complete!")

        except Exception as e:
            self.safe_log(f"CRITICAL ERROR: {e}")
            import traceback
            self.safe_log(traceback.format_exc())

        finally:
            self.processing = False
            self.root.after(0, lambda: self.btn_process.config(state='normal'))
            self.root.after(3000, self.reset_progress)  # Reset after 3 seconds
            # Clean up temp? Maybe keep for debugging.

    def start_batch_processing(self):
        if self.processing: return
        self.processing = True
        self.reset_progress()  # Reset progress bar at start

        # Inputs
        source_dir = Path(self.batch_source_dir.get())
        csv_file = Path(self.batch_csv_file.get())
        shape_dir = Path(self.batch_shape_dir.get())
        out_root = Path(self.output_dir.get())
        vrt_library_dir = Path(self.batch_vrt_lib_dir.get())
        date_filter = self.batch_date_filter.get().strip()

        # Start a thread to PREPARE the data, then show UI
        thread = threading.Thread(
            target=self.prepare_batch_data,
            args=(source_dir, csv_file, shape_dir, out_root, date_filter, vrt_library_dir),
            daemon=True
        )
        thread.start()

    def prepare_batch_data(self, source_dir, csv_file, shape_dir, out_root, date_filter, vrt_library_dir):
        try:
            self.safe_log("=== PHASE 1: Building VRT Library ===")

            # 1. Scan and Load Data
            self.batch_processor.scan_source_dir(source_dir)
            self.batch_processor.load_dole_data(csv_file)

            # 2. Build/Update VRT Library
            def vrt_progress(current, total, msg):
                self.update_progress(current, total, f"VRT Library: {msg}")
                self.safe_log(f"  [{current}/{total}] {msg}")

            vrt_library = self.batch_processor.build_vrt_library(
                source_dir=source_dir,
                shape_dir=shape_dir,
                vrt_library_dir=vrt_library_dir,
                processor=self.processor,
                progress_callback=vrt_progress
            )

            self.safe_log("\n=== PHASE 2: Building Date Matrix ===")

            # 3. Build Date Matrix with Fallback
            date_matrix = self.batch_processor.build_date_matrix(shape_dir)

            # 4. Apply date filter if specified
            if date_filter:
                date_matrix = {date_filter: date_matrix[date_filter]} if date_filter in date_matrix else {}

            # 5. Convert matrix to review_data format
            review_data = self._convert_matrix_to_review_data(date_matrix)

            self.safe_log(f"Matrix built: {len(review_data)} dates ready for review")

            # 6. Launch Review Window (on Main Thread)
            self.root.after(0, lambda: BatchReviewWindow(self.root, review_data, shape_dir, self.execute_batch_from_review))

        except Exception as e:
            self.safe_log(f"Error preparing batch: {e}")
            import traceback
            self.safe_log(traceback.format_exc())
            self.processing = False

    def _convert_matrix_to_review_data(self, date_matrix):
        """
        Convert date_matrix to review_data format for BatchReviewWindow.
        """
        review_data = defaultdict(list)

        for date, locations in date_matrix.items():
            for loc, info in locations.items():
                # VRT path becomes the "file"
                review_data[date].append({
                    'loc': loc,
                    'files': [info['vrt']],  # VRT path
                    'shp': info['shp'],
                    'fallback': info['fallback'],
                    'actual_date': info['actual_date'],
                    'ed': ''
                })

        return dict(review_data)

    def execute_batch_from_review(self, review_data):
        self.safe_log("Review complete. Starting execution...")
        out_root = Path(self.output_dir.get())
        
        thread = threading.Thread(
            target=self.run_batch_execution, 
            args=(review_data, out_root), 
            daemon=True
        )
        thread.start()

    def run_batch_execution(self, review_data, out_root):
        try:
            self.safe_log("\n=== PHASE 3: Generating COGs ===")

            total_dates = len(review_data)
            date_index = 0

            sorted_dates = sorted(review_data.keys())

            for date in sorted_dates:
                items = review_data[date]
                date_index += 1

                self.safe_log(f"\n--- Processing Date: {date} ({date_index}/{total_dates}) ---")
                self.update_progress(date_index - 1, total_dates, f"Processing {date}")

                # Collect all VRT paths for this date
                vrt_files = []
                for item in items:
                    loc = item['loc']
                    files = item.get('files', [])

                    if not files:
                        continue

                    # files[0] is the VRT path from matrix
                    vrt_path = files[0]

                    if vrt_path.exists():
                        vrt_files.append(vrt_path)

                        # Log fallback info
                        if item.get('fallback', False):
                            actual_date = item.get('actual_date', 'unknown')
                            self.safe_log(f"  {loc}: using {actual_date} [Fallback]")
                        else:
                            self.safe_log(f"  {loc}: using {date} [Source]")
                    else:
                        self.safe_log(f"  [ERROR] VRT not found: {vrt_path}")

                if not vrt_files:
                    self.safe_log(f"  [SKIP] No VRTs available for {date}")
                    continue

                # Create output directory
                date_out_dir = out_root / date
                date_out_dir.mkdir(parents=True, exist_ok=True)

                # Build mosaic VRT
                self.update_progress(date_index - 0.7, total_dates, f"{date} - Building mosaic VRT")
                mosaic_vrt = date_out_dir / f"mosaic_{date}.vrt"
                self.safe_log(f"  Building mosaic VRT from {len(vrt_files)} locations...")

                if not self.processor.build_vrt(vrt_files, mosaic_vrt):
                    self.safe_log(f"  [ERROR] Failed to build mosaic VRT for {date}")
                    continue

                # Convert to COG
                self.update_progress(date_index - 0.5, total_dates, f"{date} - Converting to COG")
                cog_output = date_out_dir / f"mosaic_{date}_cog.tif"
                self.safe_log(f"  Converting to COG...")

                if self.processor.convert_to_cog(mosaic_vrt, cog_output):
                    self.safe_log(f"  ✓ COG created: {cog_output.name}")
                else:
                    self.safe_log(f"  [ERROR] COG creation failed for {date}")
                    continue

                # Optional: R2 Sync
                if self.r2_enabled.get():
                    self.update_progress(date_index - 0.1, total_dates, f"{date} - Uploading")
                    dest = self.r2_destination.get().rstrip('/') + f"/{date}"
                    self.safe_log(f"  Syncing to R2: {dest}")
                    self.processor.sync_to_r2(date_out_dir, dest)

            self.update_progress(total_dates, total_dates, "Batch processing complete!")
            self.safe_log("\n=== BATCH PROCESSING COMPLETE ===")
            messagebox.showinfo("Done", "Batch Processing Complete")

        except Exception as e:
            self.safe_log(f"CRITICAL ERROR: {e}")
            import traceback
            self.safe_log(traceback.format_exc())

        finally:
            self.processing = False
            self.root.after(3000, self.reset_progress)  # Reset after 3 seconds

    def start_cog_pipeline(self):
        if self.processing: return
        self.processing = True
        self.reset_progress()  # Reset progress bar at start

        in_root = Path(self.cog_in_root.get())
        out_dir = Path(self.cog_out_dir.get())
        r2_dest = self.r2_destination.get()

        thread = threading.Thread(
            target=self.run_cog_pipeline_thread,
            args=(in_root, out_dir, r2_dest),
            daemon=True
        )
        thread.start()

    def run_cog_pipeline_thread(self, in_root, out_dir, r2_dest):
        try:
            self.safe_log("Starting COG Pipeline...")

            # 1. Scan and Convert
            out_dir.mkdir(parents=True, exist_ok=True)
            files = self.processor.scan_files(in_root)

            if not files:
                 self.safe_log("No mosaic TIFFs found.")
                 self.reset_progress()
            else:
                 self.safe_log(f"Found {len(files)} potential files.")
                 tracker = ProgressTracker(len(files))

                 for i, f in enumerate(files):
                     tracker.update(i)
                     self.update_progress(i, len(files), f"Converting to COG - {tracker.get_progress_string()}")

                     # Calculate output name
                     iso = self.processor.extract_iso_date(f)
                     if iso:
                         out_name = f"{iso}.tif"
                     else:
                         out_name = f"{f.stem}.tif"

                     out_path = out_dir / out_name

                     self.processor.convert_to_cog(f, out_path)

                 # Mark conversion complete
                 tracker.update(len(files))
                 self.update_progress(len(files), len(files), f"COG conversion complete - {tracker.get_progress_string()}")

            # 2. Sync
            self.safe_log("\nStarting Upload to R2...")
            self.update_progress(0, 0, "Uploading to R2...")
            self.processor.sync_to_r2(out_dir, r2_dest)

            self.update_progress(100, 100, "Pipeline complete!")
            self.safe_log("\nPIPELINE COMPLETE.")
            messagebox.showinfo("Done", "COG & Upload Pipeline Complete")

        except Exception as e:
             self.safe_log(f"CRITICAL ERROR: {e}")
             import traceback
             self.safe_log(traceback.format_exc())
        finally:
             self.processing = False
             self.root.after(3000, self.reset_progress)  # Reset after 3 seconds


def main():
    root = tk.Tk()
    app = UnifiedAppGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
