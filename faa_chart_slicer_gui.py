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

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL is required but not installed.")
    print("Install with: pip install gdal --break-system-packages")
    sys.exit(1)


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

            warp_options = gdal.WarpOptions(
                format=format,
                dstSRS='EPSG:3857',
                cutlineDSName=str(shapefile),
                cropToCutline=True,
                dstAlpha=True,
                resampleAlg=gdal.GRA_Lanczos, # Use high quality resampling
                creationOptions=creation_opts,
                multithread=True
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
                ds = None # Close dataset
                self.log(f"✓ Created {output_tiff.name}")
                return True
            else:
                self.log(f"✗ Failed to create {output_tiff.name}")
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
        """Convert VRT to a single large GeoTIFF (BigTIFF)."""
        try:
            self.log(f"Creating Combined GeoTIFF: {output_tiff.name}...")
            self.log("This may take a while for large areas...")
            
            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES', 'PREDICTOR=2']
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
            return False

    def generate_tiles(self, input_vrt: Path, output_dir: Path, zoom_levels: str) -> bool:
        """Generate tiles using gdal2tiles.py with real-time output capturing."""
        try:
            self.log(f"Generating Tiles (Zoom: {zoom_levels})...")
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
                '--zoom', zoom_levels,
                '--processes', str(cpu_count),
                '--webviewer=none',
                '--exclude',
                '--tiledriver=WEBP',
                '--webp-quality=90',
                '--xyz',
                str(input_vrt),
                str(output_dir)
            ]
            
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
        """Convert VRT or TIFF to COG using GDAL."""
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
            
            translate_options = gdal.TranslateOptions(
                format='COG',
                creationOptions=[
                    'COMPRESS=DEFLATE',
                    'PREDICTOR=2',
                    'BIGTIFF=IF_SAFER',
                    'NUM_THREADS=8',
                    'RESAMPLING=LANCZOS'
                ]
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



class BatchReviewWindow(tk.Toplevel):
    def __init__(self, parent, review_data, shape_dir, process_callback):
        super().__init__(parent)
        self.title("Batch Review")
        self.geometry("900x600")
        self.review_data = review_data # Dict: Date -> List of Items
        self.shape_dir = shape_dir
        self.process_callback = process_callback
        
        self.create_ui()
        
    def create_ui(self):
        # Top: Instructions
        lbl = ttk.Label(self, text="Review the charts found for each date. You can add missing files or remove incorrect ones.", wraplength=800)
        lbl.pack(pady=10)
        
        # Treeview
        columns = ("item", "files", "shapefile", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Date/Chart")
        self.tree.heading("files", text="Source Files (TIFF/ZIP)")
        self.tree.heading("shapefile", text="Shapefile")
        self.tree.heading("status", text="Status")
        
        self.tree.column("#0", width=250)
        self.tree.column("files", width=350)
        self.tree.column("shapefile", width=200)
        self.tree.column("status", width=80)
        
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
        
        # Bottom Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=10)
        ttk.Button(btn_frame, text="RUN BATCH PROCESS", command=self.run_batch).pack(side=tk.RIGHT, padx=10)

    def populate_tree(self):
        # Clear
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        # Sort dates
        dates = sorted(self.review_data.keys())
        
        for date in dates:
            date_node = self.tree.insert("", tk.END, text=date, open=True)
            items = self.review_data[date]
            
            for idx, item in enumerate(items):
                # item is dict: {loc, ed, files, shp, ...}
                loc = item['loc']
                files = item['files']
                shp = item['shp']
                
                # Display text
                files_str = f"{len(files)} files" if len(files) > 1 else (files[0].name if files else "MISSING")
                shp_str = shp.name if shp else "MISSING"
                status = "Ready" if (files and shp) else "Incomplete"
                
                # Insert Chart Node
                chart_node = self.tree.insert(date_node, tk.END, text=loc, values=("", shp_str, status))
                
                # Insert File Nodes as children of Chart? Or just listed in column? 
                # Better: Chart Node shows summary. Children show files.
                for f in files:
                    self.tree.insert(chart_node, tk.END, text="File", values=(f.name, "", ""))
                
                # Valid indicator color? (Not easy in standard Treeview without tags)
                
                # Store ref to data in tag or separate map? 
                # We need to map tree item to self.review_data
                # easy way: use tags or construct ID
                # item_id = f"{date}|{idx}" 
                # Setting item ID manually is cleaner.
                
                # Re-insert with ID
                # self.tree.delete(chart_node) 
                # Actually, let's keep a map.
                
    def get_selected_item(self):
        sel = self.tree.selection()
        if not sel: return None, None, None
        
        # Identify what was selected
        # If it's a child (File), get parent
        item_id = sel[0]
        parent_id = self.tree.parent(item_id)
        
        if parent_id:
            # Selected a file node, switch to parent Chart node
            chart_node_id = parent_id
            # But wait, parent of parent?
            # Root -> Date -> Chart -> File
            if self.tree.parent(chart_node_id): 
                 # This is correct: root->date->chart->file
                 pass
            else:
                 # Should not happen if structure matches
                 pass
        else:
            chart_node_id = item_id

        # Traverse up to find Date and Index
        # We need a robust way to link tree node to data.
        # Let's rely on text navigation for now since we didn't store IDs
        # OR better: rebuilding the Populate method to store IDs is safer but complex edit.
        # Simpler: Search review_data for the matching entry.
        
        # Let's try to parse the tree structure
        # Chart Node text is "Location"
        # Parent Node text is "Date"
        
        node_text = self.tree.item(chart_node_id, "text")
        parent_id = self.tree.parent(chart_node_id)
        if not parent_id: return None, None, None # Selected a Date node
        
        date_text = self.tree.item(parent_id, "text")
        
        # Find in data
        if date_text in self.review_data:
            items = self.review_data[date_text]
            for item in items:
                if item['loc'] == node_text:
                    return date_text, item, chart_node_id
        
        return None, None, None

    def add_file(self):
        date, item, node_id = self.get_selected_item()
        if not item: return
        
        f = filedialog.askopenfilename(title="Select Chart File", filetypes=[("Chart", "*.zip *.tif *.tiff")])
        if f:
            path = Path(f)
            item['files'].append(path)
            # Update UI
            # Simple refresh of children
            self.tree.insert(node_id, tk.END, text="File", values=(path.name, "", "Manual"))
            # Update status
            self.tree.set(node_id, "status", "Modified")

    def change_shapefile(self):
        date, item, node_id = self.get_selected_item()
        if not item: return
        
        f = filedialog.askopenfilename(title="Select Shapefile", initialdir=self.shape_dir, filetypes=[("Shapefile", "*.shp")])
        if f:
            path = Path(f)
            item['shp'] = path
            self.tree.set(node_id, "shapefile", path.name)
            self.tree.set(node_id, "status", "Modified")

    def remove_item(self):
        date, item, node_id = self.get_selected_item()
        if not item: 
            # maybe selected a date?
            return
            
        if messagebox.askyesno("Confirm", f"Remove {item['loc']} from processing?"):
            self.review_data[date].remove(item)
            self.tree.delete(node_id)


    def show_context_menu(self, event):
        try:
            self.tree.selection_set(self.tree.identify_row(event.y))
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

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
        self.zoom_levels = tk.StringVar(value="0-11")
        self.output_format = tk.StringVar(value="geotiff")  # Options: 'tiles', 'geotiff', 'cog', 'both'
        self.optimize_vrt = tk.BooleanVar(value=True) # Pipeline optimization
        self.r2_enabled = tk.BooleanVar(value=False)
        self.r2_destination = tk.StringVar(value="r2:sectionals")
        
        self.processing = False
        self.log_queue = queue.Queue()
        
        # Batch vars
        self.batch_source_dir = tk.StringVar(value="/Volumes/projects/newrawtiffs")
        self.batch_csv_file = tk.StringVar(value="master_dole.csv")
        self.batch_shape_dir = tk.StringVar(value="shapefiles")
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

        ttk.Label(frame, text="Filter Date (YYYY-MM-DD):").grid(row=3, column=0, **grid_opts)
        ttk.Entry(frame, textvariable=self.batch_date_filter, width=20).grid(row=3, column=1, **grid_opts)
        ttk.Label(frame, text="(Leave empty for all)").grid(row=3, column=2, **grid_opts)

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
        ttk.Label(options_frame, text="(e.g., 0-11)").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Output Format Options
        ttk.Label(options_frame, text="Output Format:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        format_frame = ttk.Frame(options_frame)
        format_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(format_frame, text="Generate Tiles", variable=self.output_format, value="tiles").pack(side=tk.LEFT, padx=5)
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
                if zoom:
                    self.safe_log(f"Generating Tiles ({zoom})...")
                    self.update_progress(0, 0, f"Generating tiles (zoom {zoom})...")
                    self.processor.generate_tiles(vrt_file, tiles_dir, zoom)
                else:
                    self.safe_log("Warning: No zoom levels specified for tile generation.")

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
        date_filter = self.batch_date_filter.get().strip()

        # Start a thread to PREPARE the data, then show UI
        thread = threading.Thread(
            target=self.prepare_batch_data,
            args=(source_dir, csv_file, shape_dir, out_root, date_filter),
            daemon=True
        )
        thread.start()

    def prepare_batch_data(self, source_dir, csv_file, shape_dir, out_root, date_filter):
        try:
            self.safe_log("Preparing batch data for review...")
            
            # 1. Scan Data
            self.batch_processor.scan_source_dir(source_dir)
            self.batch_processor.load_dole_data(csv_file)
            
            # 2. Build Review Data Dictionary
            # Dict: Date -> List of { 'loc': str, 'ed': str, 'files': List[Path], 'shp': Path }
            review_data = defaultdict(list)
            
            dates = sorted(self.batch_processor.dole_data.keys())
            for date in dates:
                if date_filter and date != date_filter: continue
                
                records = self.batch_processor.dole_data[date]
                for rec in records:
                    loc = rec['location']
                    ed = rec['edition']
                    ts = rec.get('wayback_ts')
                    
                    found_files = self.batch_processor.find_files(loc, ed, ts)
                    shp_path = self.batch_processor.find_shapefile(loc, shape_dir)
                    
                    item = {
                        'loc': loc,
                        'ed': ed,
                        'files': found_files,
                        'shp': shp_path,
                        'rec': rec
                    }
                    review_data[date].append(item)
            
            # Launch Review Window (on Main Thread)
            self.root.after(0, lambda: BatchReviewWindow(self.root, review_data, shape_dir, self.execute_batch_from_review))
            
        except Exception as e:
            self.safe_log(f"Error preparing batch: {e}")
            self.processing = False

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
            # Count total items to process
            total_dates = len(review_data)
            date_index = 0

            # Iterate Dates found in review data
            for date, items in review_data.items():
                date_index += 1
                self.safe_log(f"\nPROCESSING DATE: {date}")
                self.update_progress(date_index - 1, total_dates, f"Processing date {date_index}/{total_dates}: {date}")

                date_out_dir = out_root / date
                date_out_dir.mkdir(parents=True, exist_ok=True)

                processed_tiffs = [] # List of warped tiffs for this date

                # Count total files for this date
                total_files = sum(len(item['files']) for item in items if item['files'] and item['shp'])
                file_index = 0
                tracker = ProgressTracker(total_files) if total_files > 0 else None

                for item in items:
                    loc = item['loc']
                    files = item['files']
                    shp_path = item['shp']

                    if not files:
                        self.safe_log(f"  [SKIP] No files for {loc}")
                        continue
                    if not shp_path:
                        self.safe_log(f"  [SKIP] No shapefile for {loc}")
                        continue

                    # Process EACH found file for this chart (e.g. North and South)
                    for f_path in files:
                        self.safe_log(f"  Source: {f_path.name}")

                        # Get list of TIFFs to process
                        tiffs_to_process = []
                        if f_path.suffix.lower() == '.zip':
                            self.safe_log(f"    Unzipping...")
                            extracted = self.batch_processor.unzip_file(f_path, date_out_dir / "temp_extract")
                            tiffs_to_process.extend(extracted)
                        else:
                            tiffs_to_process.append(f_path)

                        for tiff_path in tiffs_to_process:
                            file_index += 1
                            if tracker:
                                tracker.update(file_index)
                                self.update_progress(
                                    date_index - 1 + (file_index / total_files),
                                    total_dates,
                                    f"Date {date} - {tracker.get_progress_string()}"
                                )

                            self.safe_log(f"    Processing: {tiff_path.name}")

                            # Warp
                            use_vrt = self.optimize_vrt.get()
                            ext = ".vrt" if use_vrt else ".tif"
                            fmt = "VRT" if use_vrt else "GTiff"

                            out_name = f"{tiff_path.stem}_clipped{ext}"
                            out_path = date_out_dir / "warped" / out_name

                            self.safe_log(f"    Warping (Format: {fmt})...")
                            if self.processor.warp_and_cut(tiff_path, shp_path, out_path, format=fmt):
                                processed_tiffs.append(out_path)
                            else:
                                self.safe_log(f"    Failed processing {tiff_path.name}")

                if not processed_tiffs:
                    self.safe_log(f"Date {date}: No files processed successfully.")
                    continue

                # VRT & Combine
                self.update_progress(date_index, total_dates, f"Date {date} - Building mosaic...")
                vrt_file = date_out_dir / f"combined_{date}.vrt"
                if self.processor.build_vrt(processed_tiffs, vrt_file):
                    output_fmt = self.output_format.get()

                    if output_fmt in ["geotiff", "both"]:
                         self.safe_log("  Creating Mosaic GeoTIFF...")
                         self.update_progress(date_index, total_dates, f"Date {date} - Creating GeoTIFF...")
                         self.processor.create_combined_tiff(vrt_file, date_out_dir / f"mosaic_{date}.tif")

                    if output_fmt in ["cog", "both"]:
                         self.safe_log("  Creating Mosaic COG...")
                         self.update_progress(date_index, total_dates, f"Date {date} - Creating COG...")
                         self.processor.convert_to_cog(vrt_file, date_out_dir / f"mosaic_{date}_cog.tif")

                    if output_fmt in ["tiles", "both"]:
                        zoom = self.zoom_levels.get()
                        self.safe_log(f"  Generating Tiles ({zoom})...")
                        self.update_progress(date_index, total_dates, f"Date {date} - Generating tiles...")
                        self.processor.generate_tiles(vrt_file, date_out_dir, zoom)

                # R2 Sync
                if self.r2_enabled.get():
                     # Sync the date folder
                     dest = self.r2_destination.get().rstrip('/') + f"/{date}"
                     self.safe_log(f"  Syncing to {dest}...")
                     self.update_progress(date_index, total_dates, f"Date {date} - Uploading...")
                     # If using COG mode, maybe we only want to sync the COG?
                     # For now sync entire date folder logic
                     self.processor.sync_to_r2(date_out_dir, dest)

            self.update_progress(total_dates, total_dates, "Batch processing complete!")
            self.safe_log("\nBATCH PROCESSING COMPLETE.")
            messagebox.showinfo("Done", "Batch Processing Complete")

        except Exception as e:
            self.safe_log(f"CRITICAL BATCH ERROR: {e}")
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
