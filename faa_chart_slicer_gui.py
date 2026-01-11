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
from typing import Optional, List, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL is required but not installed.")
    print("Install with: pip install gdal --break-system-packages")
    sys.exit(1)


class ChartProcessor:
    """Core processing logic using GDAL."""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
    
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def warp_and_cut(self, input_tiff: Path, shapefile: Path, output_tiff: Path) -> bool:
        """
        Warp and cut the TIFF using the shapefile as a cutline.
        Now converts to RGBA first to ensure consistent color tables and allow Lanczos resampling.
        """
        try:
            self.log(f"Processing: {input_tiff.name}")
            self.log(f"  Shapefile: {shapefile.name}")
            
            # Ensure output directory exists
            output_tiff.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Expand to RGBA (in memory VRT) to allow high-quality resampling and color consistency
            temp_vrt_name = f"/vsimem/{input_tiff.stem}_rgba.vrt"
            translate_options = gdal.TranslateOptions(
                format="VRT",
                rgbExpand="rgba"
            )
            
            # Create temp RGBA VRT
            ds_vrt = gdal.Translate(temp_vrt_name, str(input_tiff), options=translate_options)
            ds_vrt = None # Flush/Close
            
            # Step 2: Warp with Cutline
            warp_options = gdal.WarpOptions(
                format='GTiff',
                dstSRS='EPSG:3857',
                cutlineDSName=str(shapefile),
                cropToCutline=True,
                dstAlpha=True,
                resampleAlg=gdal.GRA_Lanczos, # Use high quality resampling
                creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_NEEDED'],
                multithread=True
            )
            
            ds = gdal.Warp(str(output_tiff), temp_vrt_name, options=warp_options)
            
            # Cleanup
            gdal.Unlink(temp_vrt_name)
            
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
            try: gdal.Unlink(f"/vsimem/{input_tiff.stem}_rgba.vrt")
            except: pass
            return False

    def build_vrt(self, input_files: List[Path], output_vrt: Path) -> bool:
        """Combine multiple TIFFs into a single VRT."""
        try:
            self.log(f"Building VRT: {output_vrt.name}...")
            
            input_strs = [str(f) for f in input_files]
            
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
                self.log(f"✗ Failed to create VRT.")
                return False
                
        except Exception as e:
            self.log(f"✗ Error building VRT: {e}")
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
                '--webp-quality=50',
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


class UnifiedAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FAA Chart Tool - Unified")
        self.root.geometry("1000x800")
        
        self.processor = ChartProcessor(log_callback=self.safe_log)
        
        # State
        self.charts: List[Dict] = [] # List of dicts: {'path': Path, 'shapefile': Path, 'status': str}
        self.output_dir = tk.StringVar()
        self.zoom_levels = tk.StringVar(value="0-11")
        self.output_format = tk.StringVar(value="tiles")  # Options: 'tiles', 'geotiff', 'both'
        self.processing = False
        self.log_queue = queue.Queue()
        
        self.setup_ui()
        self.check_log_queue()
        
    def setup_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        
        # --- Top Section: Chart Manager ---
        top_frame = ttk.LabelFrame(main, text="Chart Manager", padding="5")
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
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="Add Charts...", command=self.add_charts).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Link Shapefile...", command=self.link_shapefile).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Auto-Link (Name Match)", command=self.auto_link).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Remove Selected", command=self.remove_charts).pack(side=tk.LEFT, padx=5)
        
        # --- Middle Section: Options ---
        options_frame = ttk.LabelFrame(main, text="Processing Options", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output Directory
        ttk.Label(options_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(options_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(options_frame, text="Browse...", command=self.browse_output).grid(row=0, column=2, padx=5, pady=5)
        
        # Tiling Options
        ttk.Label(options_frame, text="Tile Zoom Levels:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.zoom_entry = ttk.Entry(options_frame, textvariable=self.zoom_levels, width=20)
        self.zoom_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(options_frame, text="(e.g., 0-11)").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Output Format Options
        ttk.Label(options_frame, text="Output Format:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        format_frame = ttk.Frame(options_frame)
        format_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(format_frame, text="Generate Tiles", variable=self.output_format, value="tiles").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Combined GeoTIFF", variable=self.output_format, value="geotiff").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Both", variable=self.output_format, value="both").pack(side=tk.LEFT, padx=5)
        
        # Process Button
        self.btn_process = ttk.Button(main, text="START PROCESSING", command=self.start_processing)
        self.btn_process.pack(fill=tk.X, pady=10)
        
        # --- Bottom Section: Log ---
        log_frame = ttk.LabelFrame(main, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def safe_log(self, message):
        self.log_queue.put(message)
    
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
            
            # 1. Warp/Cut individual files
            processed_files = []
            
            for i, chart in enumerate(self.charts):
                chart['status'] = "Processing..."
                # Update UI? (Requires thread safety for treeview, skipping dynamic updates for now or use after)
                
                out_name = f"{chart['path'].stem}_warped.tif"
                out_path = temp_dir / out_name
                
                success = self.processor.warp_and_cut(chart['path'], chart['shapefile'], out_path)
                
                if success:
                    processed_files.append(out_path)
                    chart['status'] = "Done"
                else:
                     chart['status'] = "Failed"
                     
            if not processed_files:
                self.safe_log("No files processed successfully. Stopping.")
                return

            # 2. Build VRT
            self.safe_log("Building VRT...")
            if self.processor.build_vrt(processed_files, vrt_file):
                self.safe_log(f"VRT created at {vrt_file}")
            else:
                self.safe_log("VRT creation failed. Stopping.")
                return

            # 3. Generate output based on selected format
            output_fmt = self.output_format.get()
            
            if output_fmt in ["geotiff", "both"]:
                tiff_file = out_dir / "combined.tif"
                self.safe_log("Creating Combined GeoTIFF...")
                self.processor.create_combined_tiff(vrt_file, tiff_file)

            if output_fmt in ["tiles", "both"]:
                zoom = self.zoom_levels.get()
                if zoom:
                    self.safe_log(f"Generating Tiles ({zoom})...")
                    self.processor.generate_tiles(vrt_file, tiles_dir, zoom)
                else:
                    self.safe_log("Warning: No zoom levels specified for tile generation.")
            
            self.safe_log("ALL TASKS COMPLETED.")
            messagebox.showinfo("Done", "Processing Complete!")
            
        except Exception as e:
            self.safe_log(f"CRITICAL ERROR: {e}")
            import traceback
            self.safe_log(traceback.format_exc())
            
        finally:
            self.processing = False
            self.root.after(0, lambda: self.btn_process.config(state='normal'))
            # Clean up temp? Maybe keep for debugging.

def main():
    root = tk.Tk()
    app = UnifiedAppGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
