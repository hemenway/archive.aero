import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import math

# --- DISABLE SAFETY LIMITS ---
Image.MAX_IMAGE_PIXELS = None
# -----------------------------

# --- LCC MATH CLASS ---
class LCCProjection:
    def __init__(self):
        # GRS 1980 / NAD83
        self.a = 6378137.0
        self.f = 1 / 298.257222101
        self.e2 = 2*self.f - self.f**2
        self.e = math.sqrt(self.e2)

        # User Parameters
        self.phi1 = math.radians(38.6666666666667)
        self.phi2 = math.radians(33.3333333333333)
        self.phi0 = math.radians(34.0)
        self.lam0 = math.radians(-98.5)
        
        # Constants
        self.m1 = self.calc_m(self.phi1)
        self.m2 = self.calc_m(self.phi2)
        self.t1 = self.calc_t(self.phi1)
        self.t2 = self.calc_t(self.phi2)
        self.t0 = self.calc_t(self.phi0)
        
        self.n = math.log(self.m1 / self.m2) / math.log(self.t1 / self.t2)
        self.F = self.m1 / (self.n * (self.t1 ** self.n))
        self.rho0 = self.a * self.F * (self.t0 ** self.n)

    def calc_m(self, phi):
        sin_phi = math.sin(phi)
        return math.cos(phi) / math.sqrt(1 - self.e2 * sin_phi**2)

    def calc_t(self, phi):
        sin_phi = math.sin(phi)
        return math.tan(math.pi/4 - phi/2) / ((1 - self.e * sin_phi) / (1 + self.e * sin_phi)) ** (self.e / 2)

    def project(self, lat_deg, lon_deg):
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        t = self.calc_t(phi)
        rho = self.a * self.F * (t ** self.n)
        theta = self.n * (lam - self.lam0)
        x = rho * math.sin(theta)
        y = self.rho0 - rho * math.cos(theta)
        return x, y

class SmartAlignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GeoTIFF Aligner v4 (Scaling Added)")
        self.root.geometry("1400x950")

        self.lcc = LCCProjection()

        # Projection String
        self.PROJ_STRING = (
            "+proj=lcc "
            "+lat_1=38.6666666666667 "
            "+lat_2=33.3333333333333 "
            "+lat_0=34 "
            "+lon_0=-98.5 "
            "+x_0=0 "
            "+y_0=0 "
            "+datum=NAD83 "
            "+units=m "
            "+no_defs"
        )
        
        # Base Metadata (Meters)
        self.base_pixel_w = 42.334928995476666
        self.base_pixel_h = -42.334316208518793
        self.default_ul_x = -424886.449
        self.default_ul_y = 252841.780

        # State
        self.image_files = []
        self.current_index = 0
        self.current_image_path = None
        
        self.original_img = None
        self.tk_img = None
        
        # View State
        self.scale = 0.1       # View Zoom
        self.view_pan_x = 0    
        self.view_pan_y = 0    
        
        # Alignment State
        self.align_x = 0       # Translation
        self.align_y = 0       
        self.img_scale = 1.0   # New: Map Scaling Factor
        
        # Mouse logic
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_aligning = False

        self._setup_ui()

    def _setup_ui(self):
        # Top Control Bar
        control_frame = tk.Frame(self.root, height=60, bg="#333")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"bg": "#555", "fg": "white", "font": ("Arial", 10, "bold"), "relief": "flat"}
        
        tk.Button(control_frame, text="1. Load Folder", command=self.load_folder, **btn_style).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(control_frame, text="2. Save & Next (Enter)", command=self.save_and_next, **btn_style).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.status_label = tk.Label(control_frame, text="Load a folder to start", bg="#333", fg="#00ff00", font=("Consolas", 12))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Instructions
        help_frame = tk.Frame(control_frame, bg="#333")
        help_frame.pack(side=tk.RIGHT, padx=20, pady=5)
        
        lbl_style = {"bg": "#333", "fg": "#aaa", "font": ("Arial", 9)}
        tk.Label(help_frame, text="Zoom: +/- or Wheel | Pan: L-Drag", **lbl_style).pack(anchor="e")
        tk.Label(help_frame, text="Move Map: R-Drag | Scale Map: [ ]", **lbl_style).pack(anchor="e")

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse Bindings
        self.canvas.bind("<ButtonPress-1>", lambda e: self.start_drag(e, aligning=False))
        self.canvas.bind("<B1-Motion>", self.do_drag)
        
        self.canvas.bind("<ButtonPress-3>", lambda e: self.start_drag(e, aligning=True)) # Right Click
        self.canvas.bind("<B3-Motion>", self.do_drag)
        
        self.canvas.bind("<Shift-ButtonPress-1>", lambda e: self.start_drag(e, aligning=True)) # Mac
        self.canvas.bind("<Shift-B1-Motion>", self.do_drag)

        self.canvas.bind("<MouseWheel>", self.zoom_wheel)
        self.canvas.bind("<Button-4>", self.zoom_wheel)
        self.canvas.bind("<Button-5>", self.zoom_wheel)
        
        # Keyboard Zoom (View)
        self.root.bind("<plus>", lambda e: self.zoom_key(1.2))
        self.root.bind("<equal>", lambda e: self.zoom_key(1.2))
        self.root.bind("<minus>", lambda e: self.zoom_key(0.8))
        self.root.bind("<underscore>", lambda e: self.zoom_key(0.8))

        # Keyboard Scale (Map Image)
        self.root.bind("<bracketleft>", lambda e: self.change_img_scale(0.995))  # [
        self.root.bind("<bracketright>", lambda e: self.change_img_scale(1.005)) # ]
        self.root.bind("<braceleft>", lambda e: self.change_img_scale(0.95))     # Shift+[
        self.root.bind("<braceright>", lambda e: self.change_img_scale(1.05))    # Shift+]

        # Navigation
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Left>", lambda e: self.nudge(-1, 0))
        self.root.bind("<Right>", lambda e: self.nudge(1, 0))
        self.root.bind("<Up>", lambda e: self.nudge(0, -1))
        self.root.bind("<Down>", lambda e: self.nudge(0, 1))

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
        self.image_files.sort()
        
        if not self.image_files:
            return messagebox.showerror("Error", "No Tiff files found")

        self.output_script = os.path.join(folder, "run_realignment.sh")
        if not os.path.exists(self.output_script):
            with open(self.output_script, "w") as f:
                f.write("#!/bin/bash\nmkdir -p aligned\n\n")

        self.current_index = 0
        self.load_image()

    def load_image(self):
        if self.current_index >= len(self.image_files):
            self.status_label.config(text="ALL DONE!")
            self.canvas.delete("all")
            return

        self.current_image_path = self.image_files[self.current_index]
        self.status_label.config(text=f"{self.current_index+1}/{len(self.image_files)}: {os.path.basename(self.current_image_path)}")

        self.original_img = Image.open(self.current_image_path)
        self.orig_w, self.orig_h = self.original_img.size
        
        # Reset Alignment & Scale
        self.align_x = 0
        self.align_y = 0
        self.img_scale = 1.0

        self.root.update()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        self.scale = min(cw/self.orig_w, ch/self.orig_h) * 0.9
        if self.scale <= 0: self.scale = 0.1
        
        center_w = (self.orig_w * self.scale) / 2
        center_h = (self.orig_h * self.scale) / 2
        self.view_pan_x = (cw / 2) - center_w
        self.view_pan_y = (ch / 2) - center_h
        
        self.redraw()

    # --- Interaction ---

    def start_drag(self, event, aligning):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_aligning = aligning

    def do_drag(self, event):
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

    def nudge(self, dx, dy):
        self.align_x += dx
        self.align_y += dy
        self.redraw()

    def change_img_scale(self, factor):
        """
        Scales the map image.
        We want to scale around the CENTER of the current view to keep alignment intuitive.
        """
        old_scale = self.img_scale
        new_scale = self.img_scale * factor
        
        # Don't go too crazy
        if new_scale < 0.1 or new_scale > 10.0: return
        
        # We need to adjust align_x/y so the visual center of the map stays fixed relative to the screen
        # Image Width on Screen:
        # W = orig_w * view_scale * img_scale
        
        # Difference in size (pixels on screen)
        # delta_w = (orig_w * view_scale * new_scale) - (orig_w * view_scale * old_scale)
        # delta_h = (orig_h * view_scale * new_scale) - (orig_h * view_scale * old_scale)
        
        # To keep center fixed, we shift Top-Left by half the delta
        # Since we are essentially expanding the image, the top-left moves UP and LEFT (negative)
        
        w_factor = self.orig_w * self.scale
        h_factor = self.orig_h * self.scale
        
        delta_w = w_factor * (new_scale - old_scale)
        delta_h = h_factor * (new_scale - old_scale)
        
        self.align_x -= delta_w / 2
        self.align_y -= delta_h / 2
        
        self.img_scale = new_scale
        self.redraw()
        
        # Show Feedback
        self.canvas.create_text(
            self.canvas.winfo_width()/2, 50, 
            text=f"Scale: {self.img_scale:.3f}x", 
            fill="yellow", font=("Arial", 16, "bold"), tags="feedback"
        )
        self.root.after(1000, lambda: self.canvas.delete("feedback"))

    def zoom_wheel(self, event):
        if event.num == 5 or event.delta < 0:
            factor = 0.9
        else:
            factor = 1.1
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)
        self._apply_zoom(factor, mouse_x, mouse_y)

    def zoom_key(self, factor):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self._apply_zoom(factor, cw/2, ch/2)

    def _apply_zoom(self, factor, center_x, center_y):
        new_scale = self.scale * factor
        if new_scale < 0.0001 or new_scale > 50: return

        rel_x = (center_x - self.view_pan_x) / self.scale
        rel_y = (center_y - self.view_pan_y) / self.scale
        
        self.scale = new_scale
        self.view_pan_x = center_x - (rel_x * self.scale)
        self.view_pan_y = center_y - (rel_y * self.scale)
        self.redraw()

    # --- Rendering ---

    def redraw(self):
        if not self.original_img: return
        self.canvas.delete("all")
        
        # 1. Draw Image
        # Size depends on img_scale now
        disp_w = int(self.orig_w * self.scale * self.img_scale)
        disp_h = int(self.orig_h * self.scale * self.img_scale)
        
        img_x = self.view_pan_x + self.align_x
        img_y = self.view_pan_y + self.align_y
        
        if disp_w > 1 and disp_h > 1:
            resized = self.original_img.resize((disp_w, disp_h), Image.Resampling.NEAREST)
            self.tk_img = ImageTk.PhotoImage(resized)
            self.canvas.create_image(img_x, img_y, anchor=tk.NW, image=self.tk_img)
            
        # 2. Draw Grid
        def to_screen(lat, lon):
            mx, my = self.lcc.project(lat, lon)
            dx_m = mx - self.default_ul_x
            dy_m = self.default_ul_y - my 
            
            dx_px = (dx_m / self.base_pixel_w) * self.scale
            dy_px = (dy_m / abs(self.base_pixel_h)) * self.scale
            
            sx = self.view_pan_x + dx_px
            sy = self.view_pan_y + dy_px
            return sx, sy

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        LAT_MIN, LAT_MAX = 30, 40
        LON_MIN, LON_MAX = -105, -92

        # Draw Longitudes
        for lon in range(LON_MIN, LON_MAX + 1):
            coords = []
            visible = False
            for i in range(21):
                lat = LAT_MIN + (LAT_MAX - LAT_MIN) * (i/20)
                sx, sy = to_screen(lat, lon)
                coords.extend([sx, sy])
                if 0 <= sx <= cw and 0 <= sy <= ch: visible = True
            
            if visible:
                self.canvas.create_line(coords, fill="cyan", smooth=True, dash=(2,4))
                mx, my = to_screen(38, lon) 
                if 0 <= mx <= cw:
                     self.canvas.create_text(mx, 20, text=f"{lon}°", fill="cyan", font=("Arial", 9, "bold"))

        # Draw Latitudes
        for lat in range(LAT_MIN, LAT_MAX + 1):
            coords = []
            visible = False
            for i in range(21):
                lon = LON_MIN + (LON_MAX - LON_MIN) * (i/20)
                sx, sy = to_screen(lat, lon)
                coords.extend([sx, sy])
                if 0 <= sx <= cw and 0 <= sy <= ch: visible = True
            
            if visible:
                self.canvas.create_line(coords, fill="#00ff00", smooth=True, dash=(2,4))
                mx, my = to_screen(lat, -104) 
                if 0 <= my <= ch:
                     self.canvas.create_text(30, my, text=f"{lat}N", fill="#00ff00", font=("Arial", 9, "bold"))

        # Origin
        ox, oy = to_screen(34, -98.5)
        self.canvas.create_line(ox-20, oy, ox+20, oy, fill="red", width=3)
        self.canvas.create_line(ox, oy-20, ox, oy+20, fill="red", width=3)
        self.canvas.create_text(ox+25, oy+25, text="ORIGIN", fill="red", anchor="nw")

    def save_and_next(self):
        # We need to account for self.img_scale in meters now.
        # The visual shift (align_x) is based on the *current scaled size* on screen.
        
        # Convert pixels to meters
        # Meter_Shift = (Visual_Shift / View_Scale) * Base_Pixel_Size
        # Note: Visual_Shift (align_x) is pixels. View_Scale relates pixels to base_pixels.
        
        shift_meters_x = (self.align_x / self.scale) * self.base_pixel_w
        shift_meters_y = (self.align_y / self.scale) * abs(self.base_pixel_h)
        
        final_ul_x = self.default_ul_x - shift_meters_x
        final_ul_y = self.default_ul_y + shift_meters_y
        
        # Calculate Lower Right
        # The image width in meters has effectively changed because we scaled the image.
        # Width_Meters = Orig_Width_Pixels * Base_Pixel_Size * Img_Scale
        
        final_width_m = self.orig_w * self.base_pixel_w * self.img_scale
        final_height_m = self.orig_h * self.base_pixel_h * self.img_scale # base_pixel_h is negative
        
        final_lr_x = final_ul_x + final_width_m
        final_lr_y = final_ul_y + final_height_m
        
        filename = os.path.basename(self.current_image_path)
        
        cmd = (
            f"gdal_translate "
            f"-a_ullr {final_ul_x:.3f} {final_ul_y:.3f} {final_lr_x:.3f} {final_lr_y:.3f} "
            f"-a_srs '{self.PROJ_STRING}' "
            f"\"{filename}\" \"aligned/{filename}\""
        )
        
        with open(self.output_script, "a") as f:
            f.write(f"echo 'Aligning {filename}...'\n")
            f.write(cmd + "\n\n")
            
        print(f"Saved {filename}")
        self.current_index += 1
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartAlignApp(root)
    root.mainloop()
