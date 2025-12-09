"""
GeoTIFF Aligner Application

A sophisticated tool for aligning GeoTIFF images with Lambert Conformal Conic (LCC) projection.
Provides interactive controls for translating, scaling, and aligning aerial maps.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import math
import logging
from typing import Optional, Tuple, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- DISABLE SAFETY LIMITS ---
# This allows loading very large GeoTIFF files
Image.MAX_IMAGE_PIXELS = None
logger.info("Image size limits disabled for large GeoTIFF support")
# -----------------------------

# --- CONSTANTS ---
class Config:
    """Configuration constants for the application."""
    
    # GRS 1980 / NAD83 Ellipsoid Parameters
    ELLIPSOID_A = 6378137.0  # Semi-major axis in meters
    ELLIPSOID_F = 1 / 298.257222101  # Flattening
    
    # Lambert Conformal Conic Projection Parameters
    LAT_1 = 38.6666666666667  # First standard parallel
    LAT_2 = 33.3333333333333  # Second standard parallel
    LAT_0 = 34.0  # Latitude of origin
    LON_0 = -98.5  # Central meridian
    
    # Default base metadata (in meters)
    BASE_PIXEL_WIDTH = 42.334928995476666
    BASE_PIXEL_HEIGHT = -42.334316208518793
    DEFAULT_UL_X = -424886.449
    DEFAULT_UL_Y = 252841.780
    
    # UI Configuration
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 950
    WINDOW_TITLE = "GeoTIFF Aligner v4 (Enhanced)"
    
    # View Configuration
    DEFAULT_SCALE = 0.1
    MIN_SCALE = 0.0001
    MAX_SCALE = 50
    ZOOM_FACTOR = 1.1
    ZOOM_WHEEL_FACTOR = 0.9
    
    # Image Scale Configuration
    MIN_IMG_SCALE = 0.1
    MAX_IMG_SCALE = 10.0
    SCALE_FACTOR_FINE = 0.995
    SCALE_FACTOR_COARSE = 0.95
    
    # Grid Configuration
    LAT_MIN, LAT_MAX = 30, 40
    LON_MIN, LON_MAX = -105, -92
    GRID_SEGMENTS = 21
    
    # File Extensions
    SUPPORTED_EXTENSIONS = ('.tif', '.tiff')
    OUTPUT_SCRIPT_NAME = "run_realignment.sh"


# --- LCC MATH CLASS ---
class LCCProjection:
    """
    Lambert Conformal Conic Projection implementation.
    
    Converts geographic coordinates (latitude/longitude) to projected
    coordinates (x/y) using the LCC projection with GRS 1980 / NAD83 ellipsoid.
    """
    
    def __init__(self):
        """Initialize the LCC projection with predefined parameters."""
        # GRS 1980 / NAD83 ellipsoid parameters
        self.a = Config.ELLIPSOID_A
        self.f = Config.ELLIPSOID_F
        self.e2 = 2 * self.f - self.f ** 2  # First eccentricity squared
        self.e = math.sqrt(self.e2)  # First eccentricity

        # User Parameters (in radians)
        self.phi1 = math.radians(Config.LAT_1)
        self.phi2 = math.radians(Config.LAT_2)
        self.phi0 = math.radians(Config.LAT_0)
        self.lam0 = math.radians(Config.LON_0)
        
        # Precompute constants for performance
        self.m1 = self._calc_m(self.phi1)
        self.m2 = self._calc_m(self.phi2)
        self.t1 = self._calc_t(self.phi1)
        self.t2 = self._calc_t(self.phi2)
        self.t0 = self._calc_t(self.phi0)
        
        self.n = math.log(self.m1 / self.m2) / math.log(self.t1 / self.t2)
        self.F = self.m1 / (self.n * (self.t1 ** self.n))
        self.rho0 = self.a * self.F * (self.t0 ** self.n)
        
        logger.info("LCC Projection initialized with NAD83 parameters")

    def _calc_m(self, phi: float) -> float:
        """
        Calculate the m parameter for LCC projection.
        
        Args:
            phi: Latitude in radians
            
        Returns:
            The calculated m parameter
        """
        sin_phi = math.sin(phi)
        return math.cos(phi) / math.sqrt(1 - self.e2 * sin_phi ** 2)

    def _calc_t(self, phi: float) -> float:
        """
        Calculate the t parameter for LCC projection.
        
        Args:
            phi: Latitude in radians
            
        Returns:
            The calculated t parameter
        """
        sin_phi = math.sin(phi)
        return math.tan(math.pi / 4 - phi / 2) / (
            (1 - self.e * sin_phi) / (1 + self.e * sin_phi)
        ) ** (self.e / 2)

    def project(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        """
        Project geographic coordinates to LCC coordinates.
        
        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            
        Returns:
            Tuple of (x, y) coordinates in meters
        """
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        t = self._calc_t(phi)
        rho = self.a * self.F * (t ** self.n)
        theta = self.n * (lam - self.lam0)
        x = rho * math.sin(theta)
        y = self.rho0 - rho * math.cos(theta)
        return x, y

    # Keep legacy method names for backward compatibility
    calc_m = _calc_m
    calc_t = _calc_t

class SmartAlignApp:
    """
    Main application class for GeoTIFF alignment.
    
    Provides an interactive GUI for aligning and scaling GeoTIFF images
    with Lambert Conformal Conic projection coordinates.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the SmartAlign application.
        
        Args:
            root: The main Tkinter window
        """
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")

        self.lcc = LCCProjection()

        # Projection String for GDAL
        self.PROJ_STRING = (
            "+proj=lcc "
            f"+lat_1={Config.LAT_1} "
            f"+lat_2={Config.LAT_2} "
            f"+lat_0={Config.LAT_0} "
            f"+lon_0={Config.LON_0} "
            "+x_0=0 "
            "+y_0=0 "
            "+datum=NAD83 "
            "+units=m "
            "+no_defs"
        )
        
        # Base Metadata (Meters)
        self.base_pixel_w = Config.BASE_PIXEL_WIDTH
        self.base_pixel_h = Config.BASE_PIXEL_HEIGHT
        self.default_ul_x = Config.DEFAULT_UL_X
        self.default_ul_y = Config.DEFAULT_UL_Y

        # State
        self.image_files: List[str] = []
        self.current_index: int = 0
        self.current_image_path: Optional[str] = None
        self.output_script: Optional[str] = None
        
        self.original_img: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.orig_w: int = 0
        self.orig_h: int = 0
        
        # View State (viewport navigation)
        self.scale: float = Config.DEFAULT_SCALE  # View Zoom
        self.view_pan_x: float = 0    
        self.view_pan_y: float = 0    
        
        # Alignment State (image positioning)
        self.align_x: float = 0  # Translation X
        self.align_y: float = 0  # Translation Y
        self.img_scale: float = 1.0  # Map Scaling Factor
        
        # Mouse logic
        self.drag_start_x: int = 0
        self.drag_start_y: int = 0
        self.is_aligning: bool = False

        self._setup_ui()
        logger.info("SmartAlignApp initialized successfully")

    def _setup_ui(self):
        """Set up the user interface components."""
        # Top Control Bar
        control_frame = tk.Frame(self.root, height=60, bg="#333")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_style = {
            "bg": "#555",
            "fg": "white",
            "font": ("Arial", 10, "bold"),
            "relief": "flat"
        }
        
        tk.Button(
            control_frame,
            text="1. Load Folder",
            command=self.load_folder,
            **btn_style
        ).pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(
            control_frame,
            text="2. Save & Next (Enter)",
            command=self.save_and_next,
            **btn_style
        ).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.status_label = tk.Label(
            control_frame,
            text="Load a folder to start",
            bg="#333",
            fg="#00ff00",
            font=("Consolas", 12)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Instructions
        help_frame = tk.Frame(control_frame, bg="#333")
        help_frame.pack(side=tk.RIGHT, padx=20, pady=5)
        
        lbl_style = {"bg": "#333", "fg": "#aaa", "font": ("Arial", 9)}
        tk.Label(
            help_frame,
            text="Zoom: +/- or Wheel | Pan: L-Drag",
            **lbl_style
        ).pack(anchor="e")
        tk.Label(
            help_frame,
            text="Move Map: R-Drag | Scale Map: [ ]",
            **lbl_style
        ).pack(anchor="e")

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._bind_events()

    def _bind_events(self):
        """Bind all keyboard and mouse events."""
        # Mouse Bindings - Pan viewport with left click
        self.canvas.bind(
            "<ButtonPress-1>",
            lambda e: self.start_drag(e, aligning=False)
        )
        self.canvas.bind("<B1-Motion>", self.do_drag)
        
        # Mouse Bindings - Move map with right click
        self.canvas.bind(
            "<ButtonPress-3>",
            lambda e: self.start_drag(e, aligning=True)
        )
        self.canvas.bind("<B3-Motion>", self.do_drag)
        
        # Mouse Bindings - Move map with Shift+Left (Mac compatibility)
        self.canvas.bind(
            "<Shift-ButtonPress-1>",
            lambda e: self.start_drag(e, aligning=True)
        )
        self.canvas.bind("<Shift-B1-Motion>", self.do_drag)

        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self.zoom_wheel)
        self.canvas.bind("<Button-4>", self.zoom_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom_wheel)  # Linux scroll down
        
        # Keyboard Zoom (View)
        self.root.bind("<plus>", lambda e: self.zoom_key(Config.ZOOM_FACTOR))
        self.root.bind("<equal>", lambda e: self.zoom_key(Config.ZOOM_FACTOR))
        self.root.bind("<minus>", lambda e: self.zoom_key(1 / Config.ZOOM_FACTOR))
        self.root.bind("<underscore>", lambda e: self.zoom_key(1 / Config.ZOOM_FACTOR))

        # Keyboard Scale (Map Image)
        self.root.bind(
            "<bracketleft>",
            lambda e: self.change_img_scale(Config.SCALE_FACTOR_FINE)
        )  # [
        self.root.bind(
            "<bracketright>",
            lambda e: self.change_img_scale(1 / Config.SCALE_FACTOR_FINE)
        )  # ]
        self.root.bind(
            "<braceleft>",
            lambda e: self.change_img_scale(Config.SCALE_FACTOR_COARSE)
        )  # Shift+[
        self.root.bind(
            "<braceright>",
            lambda e: self.change_img_scale(1 / Config.SCALE_FACTOR_COARSE)
        )  # Shift+]

        # Navigation
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Left>", lambda e: self.nudge(-1, 0))
        self.root.bind("<Right>", lambda e: self.nudge(1, 0))
        self.root.bind("<Up>", lambda e: self.nudge(0, -1))
        self.root.bind("<Down>", lambda e: self.nudge(0, 1))

    def load_folder(self):
        """
        Load a folder containing TIFF files and prepare for alignment.
        
        Creates output script file if it doesn't exist and loads the first image.
        """
        folder = filedialog.askdirectory(title="Select Folder with TIFF Files")
        if not folder:
            logger.info("Folder selection cancelled")
            return
        
        logger.info(f"Loading folder: {folder}")
        
        try:
            # Find all TIFF files in the folder
            self.image_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(Config.SUPPORTED_EXTENSIONS)
            ]
            self.image_files.sort()
            
            if not self.image_files:
                logger.warning(f"No TIFF files found in {folder}")
                return messagebox.showerror(
                    "Error",
                    "No TIFF files found in the selected folder"
                )

            logger.info(f"Found {len(self.image_files)} TIFF files")

            # Create output script if it doesn't exist
            self.output_script = os.path.join(folder, Config.OUTPUT_SCRIPT_NAME)
            if not os.path.exists(self.output_script):
                with open(self.output_script, "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write("# GeoTIFF Alignment Script\n")
                    f.write("# Generated by GeoTIFF Aligner\n\n")
                    f.write("mkdir -p aligned\n\n")
                logger.info(f"Created output script: {self.output_script}")

            self.current_index = 0
            self.load_image()
            
        except Exception as e:
            logger.error(f"Error loading folder: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load folder: {str(e)}")

    def load_image(self):
        """
        Load the current image from the file list.
        
        Resets alignment parameters and centers the image in the viewport.
        """
        if self.current_index >= len(self.image_files):
            self.status_label.config(text="ALL DONE! ✓")
            self.canvas.delete("all")
            logger.info("All images processed")
            return

        self.current_image_path = self.image_files[self.current_index]
        filename = os.path.basename(self.current_image_path)
        status_text = f"{self.current_index + 1}/{len(self.image_files)}: {filename}"
        self.status_label.config(text=status_text)
        logger.info(f"Loading image: {filename}")

        try:
            self.original_img = Image.open(self.current_image_path)
            self.orig_w, self.orig_h = self.original_img.size
            logger.info(f"Image size: {self.orig_w}x{self.orig_h}")
            
            # Reset Alignment & Scale
            self.align_x = 0
            self.align_y = 0
            self.img_scale = 1.0

            # Calculate initial scale to fit image in viewport
            self.root.update()
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            
            # Scale to fit 90% of the viewport
            scale_w = cw / self.orig_w
            scale_h = ch / self.orig_h
            self.scale = min(scale_w, scale_h) * 0.9
            
            # Ensure scale is within valid range
            if self.scale <= 0:
                self.scale = Config.DEFAULT_SCALE
            self.scale = max(
                Config.MIN_SCALE,
                min(Config.MAX_SCALE, self.scale)
            )
            
            # Center the image in the viewport
            center_w = (self.orig_w * self.scale) / 2
            center_h = (self.orig_h * self.scale) / 2
            self.view_pan_x = (cw / 2) - center_w
            self.view_pan_y = (ch / 2) - center_h
            
            self.redraw()
            logger.info(f"Image loaded successfully with scale {self.scale:.4f}")
            
        except Exception as e:
            logger.error(f"Error loading image {filename}: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    # --- Interaction ---

    def start_drag(self, event: tk.Event, aligning: bool):
        """
        Start a drag operation (either pan or align).
        
        Args:
            event: The mouse event
            aligning: If True, move the map; if False, pan the viewport
        """
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_aligning = aligning
        logger.debug(f"Started {'align' if aligning else 'pan'} drag at ({event.x}, {event.y})")

    def do_drag(self, event: tk.Event):
        """
        Continue a drag operation.
        
        Args:
            event: The mouse event
        """
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        if self.is_aligning:
            self.align_x += dx
            self.align_y += dy
        else:
            self.view_pan_x += dx
            self.view_pan_y += dy
        
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.redraw()

    def nudge(self, dx: int, dy: int):
        """
        Nudge the image by a small amount using arrow keys.
        
        Args:
            dx: Change in x direction (pixels)
            dy: Change in y direction (pixels)
        """
        self.align_x += dx
        self.align_y += dy
        self.redraw()
        logger.debug(f"Nudged by ({dx}, {dy})")

    def change_img_scale(self, factor: float):
        """
        Scale the map image around the center of the current view.
        
        Args:
            factor: The scaling factor to apply (e.g., 1.005 for slight increase)
        """
        old_scale = self.img_scale
        new_scale = self.img_scale * factor
        
        # Enforce scale limits
        if new_scale < Config.MIN_IMG_SCALE or new_scale > Config.MAX_IMG_SCALE:
            logger.debug(f"Scale limit reached: {new_scale:.3f}")
            return
        
        # Adjust align_x/y so the visual center stays fixed relative to the screen
        # This prevents the image from shifting when scaling
        w_factor = self.orig_w * self.scale
        h_factor = self.orig_h * self.scale
        
        delta_w = w_factor * (new_scale - old_scale)
        delta_h = h_factor * (new_scale - old_scale)
        
        # Shift top-left position to keep center fixed
        self.align_x -= delta_w / 2
        self.align_y -= delta_h / 2
        
        self.img_scale = new_scale
        self.redraw()
        logger.info(f"Image scale changed to {self.img_scale:.3f}x")
        
        # Show visual feedback
        self._show_feedback(f"Scale: {self.img_scale:.3f}x")
    
    def _show_feedback(self, text: str, duration: int = 1000):
        """
        Display temporary feedback text on the canvas.
        
        Args:
            text: The text to display
            duration: How long to show the text in milliseconds
        """
        try:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                50,
                text=text,
                fill="yellow",
                font=("Arial", 16, "bold"),
                tags="feedback"
            )
            self.root.after(duration, lambda: self.canvas.delete("feedback"))
        except Exception as e:
            logger.error(f"Error showing feedback: {e}")

    def zoom_wheel(self, event: tk.Event):
        """
        Handle mouse wheel zoom events.
        
        Args:
            event: The mouse wheel event
        """
        if event.num == 5 or event.delta < 0:
            factor = Config.ZOOM_WHEEL_FACTOR
        else:
            factor = 1 / Config.ZOOM_WHEEL_FACTOR
            
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)
        self._apply_zoom(factor, mouse_x, mouse_y)

    def zoom_key(self, factor: float):
        """
        Handle keyboard zoom events.
        
        Args:
            factor: The zoom factor to apply
        """
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self._apply_zoom(factor, cw / 2, ch / 2)

    def _apply_zoom(self, factor: float, center_x: float, center_y: float):
        """
        Apply zoom transformation around a specific point.
        
        Args:
            factor: The zoom factor to apply
            center_x: X coordinate of zoom center
            center_y: Y coordinate of zoom center
        """
        new_scale = self.scale * factor
        
        # Enforce scale limits
        if new_scale < Config.MIN_SCALE or new_scale > Config.MAX_SCALE:
            logger.debug(f"Zoom limit reached: {new_scale:.4f}")
            return

        # Calculate relative position to zoom center
        rel_x = (center_x - self.view_pan_x) / self.scale
        rel_y = (center_y - self.view_pan_y) / self.scale
        
        # Apply new scale and adjust pan to keep zoom point fixed
        self.scale = new_scale
        self.view_pan_x = center_x - (rel_x * self.scale)
        self.view_pan_y = center_y - (rel_y * self.scale)
        
        self.redraw()
        logger.debug(f"Zoom applied: scale={self.scale:.4f}")

    # --- Rendering ---

    def redraw(self):
        """Redraw the canvas with the current image and grid overlay."""
        if not self.original_img:
            return
            
        try:
            self.canvas.delete("all")
            
            # 1. Draw Image
            disp_w = int(self.orig_w * self.scale * self.img_scale)
            disp_h = int(self.orig_h * self.scale * self.img_scale)
            
            img_x = self.view_pan_x + self.align_x
            img_y = self.view_pan_y + self.align_y
            
            if disp_w > 1 and disp_h > 1:
                resized = self.original_img.resize(
                    (disp_w, disp_h),
                    Image.Resampling.NEAREST
                )
                self.tk_img = ImageTk.PhotoImage(resized)
                self.canvas.create_image(img_x, img_y, anchor=tk.NW, image=self.tk_img)
            
            # 2. Draw Grid
            self._draw_grid()
            
        except Exception as e:
            logger.error(f"Error redrawing canvas: {e}", exc_info=True)
    
    def _draw_grid(self):
        """Draw the geographic grid overlay with lat/lon lines."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # Helper function to convert lat/lon to screen coordinates
        def to_screen(lat: float, lon: float) -> Tuple[float, float]:
            """Convert geographic coordinates to screen coordinates."""
            mx, my = self.lcc.project(lat, lon)
            dx_m = mx - self.default_ul_x
            dy_m = self.default_ul_y - my 
            
            dx_px = (dx_m / self.base_pixel_w) * self.scale
            dy_px = (dy_m / abs(self.base_pixel_h)) * self.scale
            
            sx = self.view_pan_x + dx_px
            sy = self.view_pan_y + dy_px
            return sx, sy

        # Draw Longitude Lines (vertical)
        for lon in range(Config.LON_MIN, Config.LON_MAX + 1):
            coords = []
            visible = False
            
            for i in range(Config.GRID_SEGMENTS):
                lat = Config.LAT_MIN + (Config.LAT_MAX - Config.LAT_MIN) * (i / (Config.GRID_SEGMENTS - 1))
                sx, sy = to_screen(lat, lon)
                coords.extend([sx, sy])
                if 0 <= sx <= cw and 0 <= sy <= ch:
                    visible = True
            
            if visible and len(coords) >= 4:
                self.canvas.create_line(
                    coords,
                    fill="cyan",
                    smooth=True,
                    dash=(2, 4),
                    tags="grid"
                )
                # Add longitude label
                mx, my = to_screen(38, lon)
                if 0 <= mx <= cw:
                    self.canvas.create_text(
                        mx, 20,
                        text=f"{lon}°",
                        fill="cyan",
                        font=("Arial", 9, "bold"),
                        tags="grid_label"
                    )

        # Draw Latitude Lines (horizontal)
        for lat in range(Config.LAT_MIN, Config.LAT_MAX + 1):
            coords = []
            visible = False
            
            for i in range(Config.GRID_SEGMENTS):
                lon = Config.LON_MIN + (Config.LON_MAX - Config.LON_MIN) * (i / (Config.GRID_SEGMENTS - 1))
                sx, sy = to_screen(lat, lon)
                coords.extend([sx, sy])
                if 0 <= sx <= cw and 0 <= sy <= ch:
                    visible = True
            
            if visible and len(coords) >= 4:
                self.canvas.create_line(
                    coords,
                    fill="#00ff00",
                    smooth=True,
                    dash=(2, 4),
                    tags="grid"
                )
                # Add latitude label
                mx, my = to_screen(lat, -104)
                if 0 <= my <= ch:
                    self.canvas.create_text(
                        30, my,
                        text=f"{lat}N",
                        fill="#00ff00",
                        font=("Arial", 9, "bold"),
                        tags="grid_label"
                    )

        # Draw Origin Marker
        ox, oy = to_screen(Config.LAT_0, Config.LON_0)
        self.canvas.create_line(
            ox - 20, oy, ox + 20, oy,
            fill="red",
            width=3,
            tags="origin"
        )
        self.canvas.create_line(
            ox, oy - 20, ox, oy + 20,
            fill="red",
            width=3,
            tags="origin"
        )
        self.canvas.create_text(
            ox + 25, oy + 25,
            text="ORIGIN",
            fill="red",
            anchor="nw",
            tags="origin_label"
        )

    def save_and_next(self):
        """
        Save the current alignment to the output script and move to the next image.
        
        Calculates the final georeferencing parameters based on the current
        alignment and scaling, then writes a gdal_translate command to the script.
        """
        if not self.current_image_path or not self.output_script:
            logger.warning("Cannot save: no image or output script")
            return
        
        try:
            # Convert pixel shifts to meter shifts
            # Visual shift / view scale * base pixel size = meter shift
            shift_meters_x = (self.align_x / self.scale) * self.base_pixel_w
            shift_meters_y = (self.align_y / self.scale) * abs(self.base_pixel_h)
            
            # Calculate final upper-left corner in meters
            final_ul_x = self.default_ul_x - shift_meters_x
            final_ul_y = self.default_ul_y + shift_meters_y
            
            # Calculate image dimensions in meters (accounting for scaling)
            final_width_m = self.orig_w * self.base_pixel_w * self.img_scale
            final_height_m = self.orig_h * self.base_pixel_h * self.img_scale  # Negative
            
            # Calculate lower-right corner
            final_lr_x = final_ul_x + final_width_m
            final_lr_y = final_ul_y + final_height_m
            
            filename = os.path.basename(self.current_image_path)
            
            # Build gdal_translate command
            cmd = (
                f"gdal_translate "
                f"-a_ullr {final_ul_x:.3f} {final_ul_y:.3f} "
                f"{final_lr_x:.3f} {final_lr_y:.3f} "
                f"-a_srs '{self.PROJ_STRING}' "
                f'"{filename}" "aligned/{filename}"'
            )
            
            # Write to script
            with open(self.output_script, "a") as f:
                f.write(f"echo 'Aligning {filename}...'\n")
                f.write(cmd + "\n\n")
            
            logger.info(f"Saved alignment for {filename}")
            logger.info(f"  UL: ({final_ul_x:.3f}, {final_ul_y:.3f})")
            logger.info(f"  LR: ({final_lr_x:.3f}, {final_lr_y:.3f})")
            logger.info(f"  Scale: {self.img_scale:.3f}x")
            
            print(f"✓ Saved {filename}")
            
            # Move to next image
            self.current_index += 1
            self.load_image()
            
        except Exception as e:
            logger.error(f"Error saving alignment: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to save alignment: {str(e)}")

def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting GeoTIFF Aligner application")
        root = tk.Tk()
        app = SmartAlignApp(root)
        root.mainloop()
        logger.info("Application closed")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
        raise


if __name__ == "__main__":
    main()
