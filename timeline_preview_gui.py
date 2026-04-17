#!/usr/bin/env python3
"""
Interactive local timeline viewer for sectional chart TIFFs.

- Reads master_dole-style CSV via analyze_dole.py helpers
- Resolves files from /Volumes/projects/rawtiffs recursively
- Lets you click a timeline segment to open the matched TIFF in Quick Look
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import date, timedelta
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from analyze_dole import analyze_csv, build_file_index, resolve_tif_file


SCANNED_MASTER_CSV = "/Users/ryanhemenway/archive.aero/master_dole_scanned.csv"
COMBINED_MASTER_CSV = "/Users/ryanhemenway/archive.aero/master_dole_combined.csv"
DEFAULT_CSV = SCANNED_MASTER_CSV
LEGACY_MASTER_CSV = "/Users/ryanhemenway/archive.aero/master_dole.csv"
EARLY_CSV = "/Users/ryanhemenway/archive.aero/early_master_dole.csv"
CSV_PRESETS = [SCANNED_MASTER_CSV, COMBINED_MASTER_CSV, LEGACY_MASTER_CSV, EARLY_CSV]
DEFAULT_TIF_DIR = "/Volumes/projects/rawtiffs"
EARLY_TIF_DIR = "/Volumes/projects/rawtiffs"
STATE_SUFFIX_RE = re.compile(r",\s*[A-Z]{2}$")
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
Image.MAX_IMAGE_PIXELS = None
MAX_PREVIEW_WIDTH = 4096
CSV_FIELD_ALIASES = {
    "filename": ["filename", "tif_filename"],
    "date": ["date", "Date"],
    "edition": ["edition", "Edition"],
    "end_date": ["end_date", "End Date"],
    "left_bounds": ["Left bounds", "Left cut bounds", "left_bounds"],
    "right_bounds": ["Right bounds", "Right cut bounds", "right_bounds"],
    "top_bounds": ["Top bounds", "Top cut bounds", "top_bounds"],
    "bottom_bounds": ["Bottom bounds", "Bottom cut bounds", "bottom_bounds"],
    "proj_lat1": ["proj_lat1"],
    "proj_lat2": ["proj_lat2"],
    "proj_lat0": ["proj_lat0"],
    "proj_lon0": ["proj_lon0"],
    "crs": ["crs"],
    "tl_x": ["tl_x"],
    "tl_y": ["tl_y"],
    "tr_x": ["tr_x"],
    "tr_y": ["tr_y"],
    "br_x": ["br_x"],
    "br_y": ["br_y"],
    "bl_x": ["bl_x"],
    "bl_y": ["bl_y"],
    "gcp_tl_lat": ["gcp_tl_lat"],
    "gcp_tl_lon": ["gcp_tl_lon"],
    "gcp_tr_lat": ["gcp_tr_lat"],
    "gcp_tr_lon": ["gcp_tr_lon"],
    "gcp_br_lat": ["gcp_br_lat"],
    "gcp_br_lon": ["gcp_br_lon"],
    "gcp_bl_lat": ["gcp_bl_lat"],
    "gcp_bl_lon": ["gcp_bl_lon"],
}
CSV_EDITOR_FIELDS = [
    "filename",
    "date",
    "edition",
    "end_date",
    "left_bounds",
    "right_bounds",
    "top_bounds",
    "bottom_bounds",
    "proj_lat1",
    "proj_lat2",
    "proj_lat0",
    "proj_lon0",
    "crs",
    "tl_x",
    "tl_y",
    "tr_x",
    "tr_y",
    "br_x",
    "br_y",
    "bl_x",
    "bl_y",
    "gcp_tl_lat",
    "gcp_tl_lon",
    "gcp_tr_lat",
    "gcp_tr_lon",
    "gcp_br_lat",
    "gcp_br_lon",
    "gcp_bl_lat",
    "gcp_bl_lon",
]
CORNER_KEYS = ("tl", "tr", "br", "bl")


def _location_has_state_suffix(name):
    return bool(STATE_SUFFIX_RE.search((name or "").strip()))


def _is_early_csv_path(csv_path):
    return os.path.basename((csv_path or "").strip()).lower() == os.path.basename(EARLY_CSV).lower()


def _default_tif_dir_for_csv(csv_path, primary_tif_dir=DEFAULT_TIF_DIR):
    if _is_early_csv_path(csv_path):
        return EARLY_TIF_DIR
    return primary_tif_dir


def _clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def _first_value(row, keys):
    for key in keys:
        value = _clean_value(row.get(key))
        if value:
            return value
    return ""


def _to_float(value):
    text = _clean_value(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_longitude(value):
    numeric = _to_float(value)
    if numeric is None:
        return ""
    if numeric > 0:
        numeric = -numeric
    return str(numeric)


def _normalize_date_value(value):
    text = _clean_value(value)
    if not text:
        return ""
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return text


def build_lcc_crs(lat1, lat2, lat0, lon0):
    lat1 = _clean_value(lat1)
    lat2 = _clean_value(lat2)
    lat0 = _clean_value(lat0)
    lon0 = _clean_value(lon0)
    if not (lat1 and lat2 and lat0 and lon0):
        return ""
    return (
        f"+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} "
        f"+lon_0={lon0} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
    )


def parse_lcc_params_from_crs(crs_str):
    crs_text = _clean_value(crs_str)
    if not crs_text:
        return {}

    def _match(pattern):
        match = re.search(pattern, crs_text)
        return match.group(1) if match else ""

    return {
        "proj_lat1": _match(r"\+lat_1=([-\d.]+)"),
        "proj_lat2": _match(r"\+lat_2=([-\d.]+)"),
        "proj_lat0": _match(r"\+lat_0=([-\d.]+)"),
        "proj_lon0": _match(r"\+lon_0=([-\d.]+)"),
    }


def infer_lon0_from_bounds(row):
    left = _to_float(row.get("left_bounds"))
    right = _to_float(row.get("right_bounds"))
    if left is None or right is None:
        return ""
    if left > 0:
        left = -left
    if right > 0:
        right = -right
    return str((left + right) / 2.0)


def normalize_editor_row(row):
    normalized = {key: "" for key in CSV_EDITOR_FIELDS}
    for canonical, aliases in CSV_FIELD_ALIASES.items():
        normalized[canonical] = _first_value(row, aliases)

    normalized["date"] = _normalize_date_value(normalized["date"])
    normalized["end_date"] = _normalize_date_value(normalized["end_date"])
    normalized["left_bounds"] = _normalize_longitude(normalized["left_bounds"])
    normalized["right_bounds"] = _normalize_longitude(normalized["right_bounds"])

    parsed = parse_lcc_params_from_crs(normalized["crs"])
    for key, value in parsed.items():
        if value and not normalized.get(key):
            normalized[key] = value

    if not normalized["proj_lon0"]:
        normalized["proj_lon0"] = infer_lon0_from_bounds(normalized)

    if not normalized["crs"]:
        normalized["crs"] = build_lcc_crs(
            normalized["proj_lat1"],
            normalized["proj_lat2"],
            normalized["proj_lat0"],
            normalized["proj_lon0"],
        )

    if not normalized["filename"]:
        normalized["filename"] = _first_value(row, ["tif_filename", "filename"])

    return normalized


def _canonical_field_to_csv_name(canonical):
    aliases = CSV_FIELD_ALIASES.get(canonical, [])
    return aliases[0] if aliases else canonical


def load_csv_row_by_line(csv_path, line_num):
    if not line_num or line_num < 2:
        raise ValueError(f"Invalid CSV line number: {line_num}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        for row_line_num, row in enumerate(reader, start=2):
            if row_line_num == line_num:
                return normalize_editor_row(row)

    raise RuntimeError(f"CSV row not found at line {line_num}: {csv_path}")


def update_csv_row_by_line(csv_path, line_num, updates):
    if not line_num or line_num < 2:
        raise ValueError(f"Invalid CSV line number: {line_num}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        rows = list(reader)

    field_map = {}
    for canonical, aliases in CSV_FIELD_ALIASES.items():
        existing_name = next((name for name in aliases if name in fieldnames), None)
        if existing_name is None:
            existing_name = _canonical_field_to_csv_name(canonical)
            fieldnames.append(existing_name)
        field_map[canonical] = existing_name

    target_index = line_num - 2
    if target_index < 0 or target_index >= len(rows):
        raise RuntimeError(f"CSV row not found at line {line_num}: {csv_path}")

    row = rows[target_index]
    normalized = normalize_editor_row(row)
    for key, value in updates.items():
        normalized[key] = _clean_value(value)

    normalized["date"] = _normalize_date_value(normalized.get("date"))
    normalized["end_date"] = _normalize_date_value(normalized.get("end_date"))
    normalized["left_bounds"] = _normalize_longitude(normalized.get("left_bounds"))
    normalized["right_bounds"] = _normalize_longitude(normalized.get("right_bounds"))
    if not normalized.get("proj_lon0"):
        normalized["proj_lon0"] = infer_lon0_from_bounds(normalized)
    normalized["crs"] = build_lcc_crs(
        normalized.get("proj_lat1"),
        normalized.get("proj_lat2"),
        normalized.get("proj_lat0"),
        normalized.get("proj_lon0"),
    ) or _clean_value(normalized.get("crs"))

    for canonical, csv_name in field_map.items():
        row[csv_name] = normalized.get(canonical, "")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

@dataclass
class Segment:
    location: str
    edition: str
    start_date: date
    end_date: date
    tif_filename: str
    file_path: str
    line_num: int
    has_georef_data: bool


def _record_has_gcp_data(record):
    pixel_corner_fields = [
        "tl_x",
        "tl_y",
        "tr_x",
        "tr_y",
        "br_x",
        "br_y",
        "bl_x",
        "bl_y",
    ]
    if all(_clean_value(record.get(field)) for field in pixel_corner_fields):
        return True

    geo_corner_fields = [
        "gcp_tl_lat",
        "gcp_tl_lon",
        "gcp_tr_lat",
        "gcp_tr_lon",
        "gcp_br_lat",
        "gcp_br_lon",
        "gcp_bl_lat",
        "gcp_bl_lon",
    ]
    return any(_clean_value(record.get(field)) for field in geo_corner_fields)


class TimelinePreviewApp(tk.Tk):
    LABEL_WIDTH = 190
    HEADER_HEIGHT = 32
    ROW_HEIGHT = 24
    GROUP_HEADER_HEIGHT = 22
    MIN_SEGMENT_PX = 3
    TIMELINE_PAD_RIGHT = 40
    DAY_PX = 0.38  # ~2100px across ~15 years

    def __init__(self, rows, min_date, max_date, csv_path, tif_dir, file_index):
        super().__init__()
        self.title("Sectional Timeline Preview")
        self.geometry("1400x850")

        self.rows_all = rows
        self.rows_filtered = rows
        self.min_date = min_date
        self.max_date = max_date
        self.total_days = max(1, (max_date - min_date).days)
        self.timeline_width = max(1800, int(self.total_days * self.DAY_PX))
        self.csv_path = csv_path
        self.tif_dir = tif_dir
        self.file_index = file_index
        self.primary_tif_dir = tif_dir if not _is_early_csv_path(csv_path) else DEFAULT_TIF_DIR
        self.file_index_cache = {tif_dir: file_index}
        self.dataset_cache = {
            csv_path: {
                "rows": rows,
                "min_date": min_date,
                "max_date": max_date,
                "errors": [],
                "tif_dir": tif_dir,
                "file_index": file_index,
            }
        }
        self.cache_lock = threading.Lock()

        self.filter_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.meta_var = tk.StringVar()

        self.csv_options = []
        for candidate in [*CSV_PRESETS, csv_path]:
            if candidate and candidate not in self.csv_options:
                self.csv_options.append(candidate)
        self.csv_choice_var = tk.StringVar(value=csv_path)

        self.item_to_segment = {}
        self.selected_item = None
        self.selected_segment = None
        self.sticky_text_items = {}
        self.editor_windows = {}

        self._build_ui()
        self._bind_events()
        self._refresh_summary()
        self.redraw()
        self._set_status(self._ready_status())
        self.after(150, self._start_csv_cache_warmup)

    def _build_ui(self):
        top = ttk.Frame(self, padding=(10, 10, 10, 6))
        top.pack(fill="x")

        ttk.Label(top, text="Filter locations:").pack(side="left")
        filter_entry = ttk.Entry(top, textvariable=self.filter_var, width=28)
        filter_entry.pack(side="left", padx=(6, 6))
        self.filter_entry = filter_entry

        ttk.Button(top, text="Clear", command=self.clear_filter).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Open Selected", command=self.open_selected).pack(side="left", padx=(0, 10))
        ttk.Button(top, text="Redraw", command=self.redraw).pack(side="left", padx=(0, 10))
        ttk.Label(top, text="CSV:").pack(side="left")
        csv_combo = ttk.Combobox(
            top,
            textvariable=self.csv_choice_var,
            values=self.csv_options,
            width=52,
            state="readonly",
        )
        csv_combo.pack(side="left", padx=(6, 6))
        csv_combo.bind("<<ComboboxSelected>>", self._on_csv_choice_change)
        self.csv_combo = csv_combo

        ttk.Button(top, text="Reload CSV", command=lambda: self.reload_selected_csv(force=True)).pack(side="left", padx=(0, 10))

        ttk.Label(top, textvariable=self.summary_var).pack(side="left", padx=(8, 0))

        meta = ttk.Frame(self, padding=(10, 0, 10, 6))
        meta.pack(fill="x")
        ttk.Label(meta, textvariable=self.meta_var).pack(side="left")
        self._refresh_meta()

        legend = ttk.Frame(self, padding=(10, 0, 10, 6))
        legend.pack(fill="x")
        self._legend_chip(legend, "#2f80ed", "Has saved georef points").pack(side="left")
        self._legend_chip(legend, "#f39c12", "No saved georef points").pack(side="left", padx=(10, 0))

        main = ttk.Frame(self, padding=(10, 0, 10, 0))
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg="white", highlightthickness=0)
        y_scroll = ttk.Scrollbar(main, orient="vertical", command=self.canvas.yview)
        x_scroll = ttk.Scrollbar(self, orient="horizontal", command=self._on_xscroll)
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(fill="x", padx=10, pady=(0, 4))

        status = ttk.Frame(self, padding=(10, 0, 10, 10))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(side="left", fill="x", expand=True)

    def _legend_chip(self, parent, color, label):
        frame = ttk.Frame(parent)
        swatch = tk.Canvas(frame, width=14, height=14, highlightthickness=0, bd=0)
        swatch.create_rectangle(1, 1, 13, 13, fill=color, outline="#666")
        swatch.pack(side="left")
        ttk.Label(frame, text=label).pack(side="left", padx=(4, 0))
        return frame

    def _bind_events(self):
        self.filter_entry.bind("<KeyRelease>", self._on_filter_change)
        self.filter_entry.bind("<Escape>", lambda _e: self.clear_filter())
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-2>", self.on_canvas_right_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Control-Button-1>", self.on_canvas_right_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind(
            "<Leave>",
            lambda _e: self._set_status(self._ready_status()),
        )
        self.bind("<Command-f>", lambda _e: self.focus_filter())
        self.bind("<Control-f>", lambda _e: self.focus_filter())
        self.bind("<Command-r>", self._on_reload_key)
        self.bind("<Control-r>", self._on_reload_key)
        self.bind("<Return>", self._on_return_key)
        self.bind("<Escape>", self._on_escape)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _on_filter_change(self, _event=None):
        self.apply_filter()

    def _on_csv_choice_change(self, _event=None):
        self.reload_selected_csv()

    def _on_return_key(self, _event=None):
        if self.selected_segment:
            self.open_selected()
        return "break"

    def _on_reload_key(self, _event=None):
        self.reload_selected_csv(force=True)
        return "break"

    def _on_escape(self, _event=None):
        if self.focus_get() == self.filter_entry and self.filter_var.get():
            self.clear_filter()
            return "break"
        return None

    def _on_mousewheel(self, event):
        # macOS delivers small deltas; normalize for vertical scrolling.
        if event.widget is self.canvas or str(event.widget).startswith(str(self.canvas)):
            delta = -1 * int(event.delta)
            if delta != 0:
                units = 1 if delta > 0 else -1
                self.canvas.yview_scroll(units, "units")
            return "break"
        return None

    def _on_shift_mousewheel(self, event):
        delta = -1 * int(event.delta)
        if delta != 0:
            units = 1 if delta > 0 else -1
            self.canvas.xview_scroll(units, "units")
            self._refresh_sticky_text_items()
        return "break"

    def _on_xscroll(self, *args):
        self.canvas.xview(*args)
        self._refresh_sticky_text_items()

    def _register_sticky_text(self, item_id, base_x):
        self.sticky_text_items[item_id] = float(base_x)
        return item_id

    def _refresh_sticky_text_items(self):
        if not self.sticky_text_items:
            return
        left_x = self.canvas.canvasx(0)
        for item_id, base_x in list(self.sticky_text_items.items()):
            if item_id not in self.canvas.find_all():
                self.sticky_text_items.pop(item_id, None)
                continue
            coords = self.canvas.coords(item_id)
            if len(coords) < 2:
                continue
            self.canvas.coords(item_id, left_x + base_x, coords[1])
            self.canvas.tag_raise(item_id)

    def focus_filter(self):
        self.filter_entry.focus_set()
        self.filter_entry.selection_range(0, "end")
        return "break"

    def clear_filter(self):
        if self.filter_var.get():
            self.filter_var.set("")
            self.apply_filter()
        self.focus_filter()

    def apply_filter(self):
        q = self.filter_var.get().strip().lower()
        if not q:
            self.rows_filtered = self.rows_all
        else:
            self.rows_filtered = [row for row in self.rows_all if q in row["location"].lower()]
        self._refresh_summary()
        self.redraw()

    def _refresh_summary(self):
        total_locations = len(self.rows_all)
        shown_locations = len(self.rows_filtered)
        shown_segments = sum(len(row["segments"]) for row in self.rows_filtered)
        self.summary_var.set(
            f"Showing {shown_locations}/{total_locations} locations, {shown_segments} segments"
        )

    def _refresh_meta(self):
        self.meta_var.set(f"CSV: {self.csv_path}    TIFF dir: {self.tif_dir}")

    def _preferred_tif_dir_for_csv(self, csv_path):
        return _default_tif_dir_for_csv(csv_path, primary_tif_dir=self.primary_tif_dir)

    def _ready_status(self):
        return "Ready. Left-click opens in Quick Look. Right-click opens the segment editor."

    def _get_or_build_file_index(self, tif_dir, announce=True):
        with self.cache_lock:
            cached = self.file_index_cache.get(tif_dir)
        if cached is not None:
            return cached

        if announce:
            self._set_status(f"Indexing local TIFFs recursively: {tif_dir}")
            self.update_idletasks()
        built_index = build_file_index(tif_dir)
        with self.cache_lock:
            return self.file_index_cache.setdefault(tif_dir, built_index)

    def _build_dataset(self, csv_path, force=False, announce=True):
        target_csv = (csv_path or "").strip()
        if not target_csv:
            raise RuntimeError("No CSV path selected")

        if not force:
            with self.cache_lock:
                cached = self.dataset_cache.get(target_csv)
            if cached is not None:
                return cached

        target_tif_dir = self._preferred_tif_dir_for_csv(target_csv)
        file_index = self._get_or_build_file_index(target_tif_dir, announce=announce)
        errors, locations_data = analyze_csv(target_csv)
        rows, min_date, max_date = build_rows(locations_data, file_index)
        dataset = {
            "rows": rows,
            "min_date": min_date,
            "max_date": max_date,
            "errors": errors,
            "tif_dir": target_tif_dir,
            "file_index": file_index,
        }
        with self.cache_lock:
            self.dataset_cache[target_csv] = dataset
        return dataset

    def _apply_dataset(self, csv_path, dataset, scroll_pos=None):
        self.rows_all = dataset["rows"]
        self.min_date = dataset["min_date"]
        self.max_date = dataset["max_date"]
        self.total_days = max(1, (self.max_date - self.min_date).days)
        self.timeline_width = max(1800, int(self.total_days * self.DAY_PX))
        self.csv_path = csv_path
        self.tif_dir = dataset["tif_dir"]
        self.file_index = dataset["file_index"]
        if csv_path not in self.csv_options:
            self.csv_options.append(csv_path)
            self.csv_combo.configure(values=self.csv_options)
        self.csv_choice_var.set(csv_path)
        self._refresh_meta()

        q = self.filter_var.get().strip().lower()
        if not q:
            self.rows_filtered = self.rows_all
        else:
            self.rows_filtered = [row for row in self.rows_all if q in row["location"].lower()]

        self._refresh_summary()
        self.redraw(scroll_pos=scroll_pos)

        errors = dataset.get("errors") or []
        segment_count = sum(len(row["segments"]) for row in self.rows_all)
        if errors:
            self._set_status(
                f"Reloaded CSV ({os.path.basename(self.csv_path)}) with {len(errors)} issue(s): "
                f"{len(self.rows_all)} locations, {segment_count} segments"
            )
        else:
            self._set_status(
                f"Reloaded CSV ({os.path.basename(self.csv_path)}): "
                f"{len(self.rows_all)} locations, {segment_count} segments"
            )

    def invalidate_csv_cache(self, csv_path):
        target_csv = (csv_path or "").strip()
        if not target_csv:
            return
        with self.cache_lock:
            self.dataset_cache.pop(target_csv, None)

    def _start_csv_cache_warmup(self):
        targets = [csv for csv in self.csv_options if csv and csv != self.csv_path]
        if not targets:
            return

        def worker():
            for target_csv in targets:
                try:
                    self._build_dataset(target_csv, announce=False)
                except Exception:
                    continue

        threading.Thread(target=worker, daemon=True).start()

    def redraw(self, scroll_pos=None):
        self.canvas.delete("all")
        self.item_to_segment.clear()
        self.selected_item = None
        self.selected_segment = None
        self.sticky_text_items.clear()

        width = self.LABEL_WIDTH + self.timeline_width + self.TIMELINE_PAD_RIGHT
        content_height = self._content_height()
        height = self.HEADER_HEIGHT + content_height + 10

        self._draw_background(width, height)
        self._draw_header(self.HEADER_HEIGHT + content_height)
        self._draw_rows()

        self.canvas.configure(scrollregion=(0, 0, width, height))
        if scroll_pos:
            self._restore_scroll_position(scroll_pos)
        self._refresh_sticky_text_items()

    def _capture_scroll_position(self):
        if not hasattr(self, "canvas"):
            return None
        return {
            "x": float(self.canvas.canvasx(0)),
            "y": float(self.canvas.canvasy(0)),
        }

    def _restore_scroll_position(self, scroll_pos):
        if not scroll_pos:
            return

        self.update_idletasks()

        try:
            left = max(0.0, float(scroll_pos.get("x", 0.0)))
            top = max(0.0, float(scroll_pos.get("y", 0.0)))
        except (TypeError, ValueError):
            return

        scrollregion = self.canvas.cget("scrollregion")
        if not scrollregion:
            return

        try:
            x1, y1, x2, y2 = [float(value) for value in str(scrollregion).split()]
        except ValueError:
            return

        content_width = max(1.0, x2 - x1)
        content_height = max(1.0, y2 - y1)
        viewport_width = max(1.0, float(self.canvas.winfo_width()))
        viewport_height = max(1.0, float(self.canvas.winfo_height()))
        max_x = max(0.0, content_width - viewport_width)
        max_y = max(0.0, content_height - viewport_height)

        left = min(left, max_x)
        top = min(top, max_y)

        self.canvas.xview_moveto(0.0 if max_x <= 0 else left / content_width)
        self.canvas.yview_moveto(0.0 if max_y <= 0 else top / content_height)
        self._refresh_sticky_text_items()

    def reload_selected_csv(self, force=False):
        target_csv = (self.csv_choice_var.get() or self.csv_path).strip()
        return self.reload_csv(csv_path=target_csv, force=force)

    def reload_csv(self, csv_path=None, force=False):
        target_csv = (csv_path or self.csv_path).strip()
        if not target_csv:
            self._set_status("Reload failed: no CSV path selected")
            return False
        if not self._prepare_reload_for_csv(target_csv):
            return False
        scroll_pos = self._capture_scroll_position()

        try:
            if force:
                self._set_status(f"Reloading CSV: {target_csv}")
                self.update_idletasks()
            dataset = self._build_dataset(target_csv, force=force, announce=force)
        except Exception as exc:
            messagebox.showerror("Reload failed", f"Failed to reload CSV:\n\n{target_csv}\n\n{exc}")
            self._set_status(f"Reload failed: {exc}")
            return False

        self._apply_dataset(target_csv, dataset, scroll_pos=scroll_pos)
        return True

    def _draw_background(self, width, height):
        self.canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="")
        self.canvas.create_rectangle(0, 0, width, self.HEADER_HEIGHT, fill="#f5f7fa", outline="")
        self.canvas.create_line(self.LABEL_WIDTH, 0, self.LABEL_WIDTH, height, fill="#b0bec5")

    def _grouped_rows(self):
        with_state = [row for row in self.rows_filtered if row.get("has_state_name")]
        without_state = [row for row in self.rows_filtered if not row.get("has_state_name")]
        sections = []
        if with_state:
            sections.append(("Locations with state in name", with_state))
        if without_state:
            sections.append(("Locations without state in name", without_state))
        return sections

    def _content_height(self):
        sections = self._grouped_rows()
        if not sections:
            return self.ROW_HEIGHT + 16
        return sum(self.GROUP_HEADER_HEIGHT + (len(section_rows) * self.ROW_HEIGHT) for _name, section_rows in sections)

    def _draw_header(self, content_bottom):
        # Year ticks
        for y in range(self.min_date.year, self.max_date.year + 1):
            year_date = date(y, 1, 1)
            if year_date < self.min_date:
                year_date = self.min_date
            offset = (year_date - self.min_date).days
            x = self.LABEL_WIDTH + int(offset * self.DAY_PX)
            self.canvas.create_line(x, 0, x, content_bottom,
                                    fill="#eceff1")
            self.canvas.create_text(x + 4, self.HEADER_HEIGHT // 2, anchor="w",
                                    text=str(y), fill="#455a64", font=("Helvetica", 10, "bold"))

        self._register_sticky_text(
            self.canvas.create_text(
                10,
                self.HEADER_HEIGHT // 2,
                anchor="w",
                text="Location",
                fill="#263238",
                font=("Helvetica", 10, "bold"),
            ),
            10,
        )

    def _draw_rows(self):
        sections = self._grouped_rows()
        full_width = self.LABEL_WIDTH + self.timeline_width + self.TIMELINE_PAD_RIGHT

        if not sections:
            self.canvas.create_text(
                self.LABEL_WIDTH + 20,
                self.HEADER_HEIGHT + 30,
                anchor="w",
                text="No matching locations",
                fill="#607d8b",
                font=("Helvetica", 12),
            )
            return

        y_cursor = self.HEADER_HEIGHT
        draw_idx = 0
        is_early_csv = _is_early_csv_path(self.csv_path)

        for section_name, section_rows in sections:
            header_top = y_cursor
            header_bottom = header_top + self.GROUP_HEADER_HEIGHT
            self.canvas.create_rectangle(0, header_top, full_width, header_bottom, fill="#eef3f8", outline="")
            self._register_sticky_text(
                self.canvas.create_text(
                    8,
                    header_top + (self.GROUP_HEADER_HEIGHT // 2),
                    anchor="w",
                    text=f"{section_name} ({len(section_rows)})",
                    fill="#34495e",
                    font=("Helvetica", 9, "bold"),
                ),
                8,
            )
            self.canvas.create_line(0, header_bottom, full_width, header_bottom, fill="#d7e0e8")
            y_cursor = header_bottom

            for row in section_rows:
                y = y_cursor
                y_mid = y + self.ROW_HEIGHT // 2
                if draw_idx % 2 == 0:
                    self.canvas.create_rectangle(0, y, full_width, y + self.ROW_HEIGHT, fill="#fcfdff", outline="")
                self.canvas.create_line(0, y + self.ROW_HEIGHT, full_width, y + self.ROW_HEIGHT, fill="#f0f3f6")
                self._register_sticky_text(
                    self.canvas.create_text(
                        8,
                        y_mid,
                        anchor="w",
                        text=row["location"],
                        fill="#263238",
                        font=("Helvetica", 10),
                    ),
                    8,
                )

                for seg in row["segments"]:
                    x1, x2 = self._segment_x(seg)
                    y1 = y + 4
                    y2 = y + self.ROW_HEIGHT - 4

                    fill = "#2f80ed" if seg.has_georef_data else "#f39c12"
                    outline = "#1f2d3d" if seg.file_path and os.path.exists(seg.file_path) else "#90a4ae"

                    item = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
                    self.item_to_segment[item] = seg

                    label_text = "-" if seg.edition == "Unknown" else str(seg.edition or "")
                    label_width = x2 - x1
                    min_label_px = 16 if is_early_csv else 24
                    label_font = ("Helvetica", 7, "bold") if is_early_csv else ("Helvetica", 8, "bold")

                    # Early-master segments are much shorter (fixed 56-day spans), so use a
                    # smaller font and lower threshold to show known edition numbers on-block.
                    if is_early_csv and label_text == "-":
                        label_text = ""

                    if label_width >= min_label_px and label_text:
                        self.canvas.create_text(
                            (x1 + x2) / 2,
                            y_mid,
                            text=label_text,
                            fill="white",
                            font=label_font,
                        )

                y_cursor += self.ROW_HEIGHT
                draw_idx += 1

    def _segment_x(self, seg):
        start_offset = (seg.start_date - self.min_date).days
        duration = max(1, (seg.end_date - seg.start_date).days)
        x1 = self.LABEL_WIDTH + int(start_offset * self.DAY_PX)
        x2 = self.LABEL_WIDTH + int((start_offset + duration) * self.DAY_PX)
        if x2 - x1 < self.MIN_SEGMENT_PX:
            x2 = x1 + self.MIN_SEGMENT_PX
        return x1, x2

    def _segment_from_event(self, event):
        # Find topmost rectangle tied to a segment.
        items = self.canvas.find_overlapping(
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
        )
        for item_id in reversed(items):
            seg = self.item_to_segment.get(item_id)
            if seg:
                return item_id, seg
        return None, None

    def on_canvas_motion(self, event):
        _item, seg = self._segment_from_event(event)
        if seg:
            exists = bool(seg.file_path and os.path.exists(seg.file_path))
            status = (
                f"{seg.location} | Ed {seg.edition} | "
                f"{seg.start_date.isoformat()} -> {seg.end_date.isoformat()} | "
                f"{'FOUND' if exists else 'MISSING'} | {seg.file_path or seg.tif_filename}"
            )
            self._set_status(status)
        else:
            self._set_status(self._ready_status())

    def on_canvas_click(self, event):
        # On macOS, Control-click can map to secondary click; avoid opening Quick Look
        # when the edit handler is intended.
        if getattr(event, "state", 0) & 0x0004:
            return "break"
        item_id, seg = self._segment_from_event(event)
        if not seg:
            return
        self._select_item(item_id, seg)
        self.open_segment(seg)

    def on_canvas_right_click(self, event):
        item_id, seg = self._segment_from_event(event)
        if not seg:
            return "break"
        self._select_item(item_id, seg)
        self.open_segment_editor(seg)
        return "break"

    def _select_item(self, item_id, seg):
        if self.selected_item and self.selected_item in self.item_to_segment:
            old_seg = self.item_to_segment[self.selected_item]
            default_outline = "#1f2d3d" if old_seg.file_path and os.path.exists(old_seg.file_path) else "#90a4ae"
            self.canvas.itemconfigure(self.selected_item, width=1, outline=default_outline)

        self.selected_item = item_id
        self.selected_segment = seg
        self.canvas.itemconfigure(item_id, width=2, outline="#e53935")

    def open_selected(self):
        if not self.selected_segment:
            messagebox.showinfo("No selection", "Click a timeline segment first.")
            return
        self.open_segment(self.selected_segment)

    def _segment_editor_key(self, seg, csv_path=None):
        return ((csv_path or self.csv_path), int(seg.line_num or 0))

    def _normalize_segment_metadata(self, seg):
        return {
            "edition": "" if str(seg.edition) == "Unknown" else str(seg.edition or ""),
            "date": seg.start_date.isoformat() if seg.start_date else "",
            "end_date": seg.end_date.isoformat() if seg.end_date else "",
        }

    def _validate_segment_metadata_values(self, date_value, end_date_value):
        def _parse_iso(value, label):
            if not value:
                return None
            try:
                return date.fromisoformat(value)
            except ValueError:
                messagebox.showerror(
                    "Invalid date",
                    f"{label} must be in YYYY-MM-DD format.\n\nReceived: {value}",
                )
                return False

        parsed_start = _parse_iso(date_value, "Start date")
        if parsed_start is False:
            return None

        parsed_end = _parse_iso(end_date_value, "End date")
        if parsed_end is False:
            return None

        if parsed_start and parsed_end and parsed_end < parsed_start:
            messagebox.showerror(
                "Invalid range",
                f"End date {end_date_value} is before start date {date_value}.",
            )
            return None

        return {
            "start_date": parsed_start,
            "end_date": parsed_end,
        }

    def _save_segment_editor_state(self, csv_path, seg, updates):
        validated = self._validate_segment_metadata_values(
            _clean_value(updates.get("date")),
            _clean_value(updates.get("end_date")),
        )
        if validated is None:
            return False

        try:
            update_csv_row_by_line(csv_path, seg.line_num, updates)
        except Exception as exc:
            messagebox.showerror("Save failed", f"Failed to save metadata to CSV:\n\n{csv_path}\n\n{exc}")
            self._set_status(f"Metadata save failed: {exc}")
            return False

        self.invalidate_csv_cache(csv_path)
        self._set_status(f"Saved metadata for {seg.location} (line {seg.line_num}).")
        return validated

    def _open_editors_for_csv(self, csv_path):
        target_csv = (csv_path or "").strip()
        if not target_csv:
            return []

        editors = []
        for editor_key, window in list(self.editor_windows.items()):
            if not window or not window.winfo_exists():
                self.editor_windows.pop(editor_key, None)
                continue
            if (window.csv_path or "").strip() == target_csv:
                editors.append(window)
        return editors

    def _prepare_reload_for_csv(self, csv_path):
        dirty_editors = [window for window in self._open_editors_for_csv(csv_path) if window.has_unsaved_changes()]
        if not dirty_editors:
            return True

        editor_count = len(dirty_editors)
        editor_label = "editor" if editor_count == 1 else "editors"
        decision = messagebox.askyesnocancel(
            "Unsaved changes",
            (
                f"There {'is' if editor_count == 1 else 'are'} {editor_count} open segment {editor_label} "
                f"with unsaved changes.\n\n"
                "Yes: save changes, then reload.\n"
                "No: reload without saving.\n"
                "Cancel: keep editing."
            ),
        )
        if decision is None:
            self._set_status("Reload cancelled.")
            return False
        if decision is False:
            return True

        for window in dirty_editors:
            if not window.save_changes(reload_after_save=False):
                self._set_status("Reload cancelled.")
                return False
        return True

    def open_segment_editor(self, seg):
        source_csv_path = self.csv_path
        editor_key = self._segment_editor_key(seg, csv_path=source_csv_path)
        existing = self.editor_windows.get(editor_key)
        if existing and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        window = SegmentEditorWindow(
            app=self,
            seg=seg,
            csv_path=source_csv_path,
            editor_key=editor_key,
        )
        self.editor_windows[editor_key] = window

    def edit_segment_metadata(self, seg):
        self.open_segment_editor(seg)

    def open_segment(self, seg):
        path = seg.file_path
        if not path:
            messagebox.showerror("No file", f"No file is associated with this segment.\n\n{seg.tif_filename}")
            return

        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Resolved file does not exist:\n\n{path}")
            self._set_status(f"Missing file: {path}")
            return

        try:
            # Launch macOS Quick Look for fast browsing.
            subprocess.Popen(
                ["qlmanage", "-p", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            # Fallback to the default app if Quick Look fails.
            subprocess.Popen(["open", path])

        self._set_status(
            f"Opened in Quick Look: {seg.location} | Ed {seg.edition} | {os.path.basename(path)}"
        )

    def _set_status(self, text):
        self.status_var.set(text)


class SegmentEditorWindow(tk.Toplevel):
    SIDEBAR_WIDTH = 240
    MIN_ZOOM = 0.05
    MAX_ZOOM = 8.0
    POINT_RADIUS = 5
    CROSSHAIR_SIZE = 20
    LOUPE_SIZE = 220
    LOUPE_ZOOM = 4
    FAST_RENDER_DELAY_MS = 1
    HIGH_QUALITY_RENDER_DELAY_MS = 90
    CORNER_LABELS = ("Top Left", "Top Right", "Bottom Right", "Bottom Left")
    CORNER_SHORT = ("TL", "TR", "BR", "BL")

    def __init__(self, app, seg, csv_path, editor_key):
        super().__init__(app)
        self.app = app
        self.seg = seg
        self.csv_path = csv_path
        self.editor_key = editor_key
        self.row_data = load_csv_row_by_line(csv_path, seg.line_num)
        self.original_metadata = {
            "date": self.row_data.get("date", ""),
            "end_date": self.row_data.get("end_date", ""),
            "edition": self.row_data.get("edition", ""),
        }
        self.preview_image = None
        self.tk_image = None
        self.image_item = None
        self.original_width = 0
        self.original_height = 0
        self.zoom = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_origin = None
        self.is_dragging = False
        self.render_after_id = None
        self.high_quality_after_id = None
        self.image_load_token = 0
        self.image_loading = False
        self.has_local_file = bool(seg.file_path and os.path.exists(seg.file_path))
        self.preview_scale_x = 1.0
        self.preview_scale_y = 1.0
        self.overlay_lines = []
        self.active_point = None
        self.locked_points = []
        self.original_points_count = 0
        self.original_locked_points = []
        self.has_saved_locked_points = self._row_has_locked_points()
        self._syncing_lon0 = False
        self.last_render_size = None
        self.last_render_resample = None
        self.loupe_image = None
        self.loupe_image_item = None

        self.date_var = tk.StringVar(value=self.original_metadata["date"])
        self.end_date_var = tk.StringVar(value=self.original_metadata["end_date"])
        self.edition_var = tk.StringVar(value=self.original_metadata["edition"])
        self.bounds_vars = {
            "top": tk.StringVar(value=self.row_data.get("top_bounds", "")),
            "bottom": tk.StringVar(value=self.row_data.get("bottom_bounds", "")),
            "left": tk.StringVar(value=self.row_data.get("left_bounds", "")),
            "right": tk.StringVar(value=self.row_data.get("right_bounds", "")),
        }
        self.proj_vars = {
            "lat1": tk.StringVar(value=self.row_data.get("proj_lat1", "") or "45.0"),
            "lat2": tk.StringVar(value=self.row_data.get("proj_lat2", "") or "33.0"),
            "lat0": tk.StringVar(value=self.row_data.get("proj_lat0", "") or "39.0"),
            "lon0": tk.StringVar(value=self.row_data.get("proj_lon0", "")),
        }
        self.gcp_vars = {
            corner: {
                "lat": tk.StringVar(value=self.row_data.get(f"gcp_{corner}_lat", "")),
                "lon": tk.StringVar(value=self.row_data.get(f"gcp_{corner}_lon", "")),
            }
            for corner in CORNER_KEYS
        }
        self.status_var = tk.StringVar(value="Loaded. Edit metadata or place four points.")

        self.title(f"Segment Editor - {seg.location}")
        self.geometry("1600x950")
        self.minsize(1000, 700)
        self.configure(bg="#111111")
        self.transient(app)

        self._build_ui()
        self._bind_events()
        self._sync_lon0_from_bounds()
        self._update_dirty_state()
        self._begin_async_image_load()

    def _build_ui(self):
        root = tk.Frame(self, bg="#111111")
        root.pack(fill="both", expand=True)

        workspace = tk.Frame(root, bg="#111111")
        workspace.pack(side="top", fill="both", expand=True)

        sidebar = tk.Frame(
            workspace,
            bg="#1e1e1e",
            width=self.SIDEBAR_WIDTH,
            highlightbackground="#444444",
            highlightthickness=1,
        )
        sidebar.pack(side="left", fill="y", padx=10, pady=10)
        sidebar.pack_propagate(False)

        viewer = tk.Frame(workspace, bg="#000000")
        viewer.pack(side="left", fill="both", expand=True)

        toolbar = tk.Frame(root, bg="#222222", height=50, highlightbackground="#333333", highlightthickness=1)
        toolbar.pack(side="bottom", fill="x")
        toolbar.pack_propagate(False)

        self._build_sidebar(sidebar)
        self._build_viewer(viewer)
        self._build_toolbar(toolbar)

    def _build_sidebar(self, parent):
        status = tk.Label(
            parent,
            textvariable=self.status_var,
            bg="#1e1e1e",
            fg="#7CFC7C",
            font=("Helvetica", 12, "bold"),
            pady=8,
            wraplength=self.SIDEBAR_WIDTH - 28,
            justify="center",
        )
        status.pack(fill="x")

        actions = self._panel(parent)
        actions.pack(fill="x", pady=(0, 10))

        self.lock_button = self._button(actions, "LOCK POINT (Enter)", self.confirm_point, bg="#007bff", state="disabled")
        self.lock_button.pack(fill="x")
        action_row = tk.Frame(actions, bg="#2a2a2a")
        action_row.pack(fill="x", pady=(8, 0))
        self._button(action_row, "Undo", self.undo_point, bg="#dc3545").pack(side="left", fill="x", expand=True)
        self._button(action_row, "Reset View", self.fit_image, bg="#6c757d").pack(side="left", fill="x", expand=True, padx=(6, 0))

        bounds_panel = self._panel(parent)
        bounds_panel.pack(fill="x", pady=(0, 10))
        self._panel_title(bounds_panel, "Map Extents (From CSV)")
        self._sidebar_field(bounds_panel, "Top Latitude", self.bounds_vars["top"]).pack(fill="x", pady=(0, 6))
        row = tk.Frame(bounds_panel, bg="#2a2a2a")
        row.pack(fill="x", pady=(0, 6))
        self._sidebar_field(row, "Left Longitude", self.bounds_vars["left"]).pack(side="left", fill="x", expand=True)
        self._sidebar_field(row, "Right Longitude", self.bounds_vars["right"]).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._sidebar_field(bounds_panel, "Bottom Latitude", self.bounds_vars["bottom"]).pack(fill="x")

        gcp_panel = self._panel(parent)
        gcp_panel.pack(fill="x", pady=(0, 10))
        self._panel_title(gcp_panel, "GCP Lat/Lon (Optional)")
        for corner, label in zip(CORNER_KEYS, self.CORNER_SHORT):
            row = tk.Frame(gcp_panel, bg="#2a2a2a")
            row.pack(fill="x", pady=(0, 6))
            self._sidebar_field(row, f"{label} Lat", self.gcp_vars[corner]["lat"]).pack(side="left", fill="x", expand=True)
            self._sidebar_field(row, f"{label} Lon", self.gcp_vars[corner]["lon"]).pack(side="left", fill="x", expand=True, padx=(6, 0))

        lcc_panel = self._panel(parent)
        lcc_panel.pack(fill="x", pady=(0, 10))
        self._panel_title(lcc_panel, "LCC Params")
        row = tk.Frame(lcc_panel, bg="#2a2a2a")
        row.pack(fill="x", pady=(0, 6))
        self._sidebar_field(row, "Lat 1", self.proj_vars["lat1"]).pack(side="left", fill="x", expand=True)
        self._sidebar_field(row, "Lat 2", self.proj_vars["lat2"]).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._sidebar_field(lcc_panel, "Lat Origin", self.proj_vars["lat0"]).pack(fill="x", pady=(0, 6))
        self._sidebar_field(lcc_panel, "Lon Origin", self.proj_vars["lon0"], readonly=True).pack(fill="x")
        self._button(parent, "Check Overlay", self.check_overlay, bg="#6c757d").pack(fill="x")

    def _build_viewer(self, parent):
        header = tk.Frame(parent, bg="#000000")
        header.pack(side="top", fill="x")

        self.title_badge = tk.Label(
            header,
            text=self.seg.tif_filename or os.path.basename(self.seg.file_path or "") or self.seg.location,
            bg="#111111",
            fg="#ffffff",
            font=("Helvetica", 12, "bold"),
            padx=14,
            pady=6,
            relief="solid",
            borderwidth=1,
        )
        self.title_badge.pack(side="top", pady=10)

        badge_row = tk.Frame(header, bg="#000000")
        badge_row.pack(side="top", pady=(0, 8))
        self._badge(badge_row, "CSV Found" if self.has_local_file else "Missing", "#28a745" if self.has_local_file else "#dc3545").pack(side="left")
        self._badge(badge_row, f"{self.seg.location} | Line {self.seg.line_num}", "#444444").pack(side="left", padx=(8, 0))

        self.canvas_shell = tk.Frame(parent, bg="#000000")
        self.canvas_shell.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_shell, bg="#000000", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.loupe = tk.Canvas(
            self.canvas_shell,
            width=self.LOUPE_SIZE,
            height=self.LOUPE_SIZE,
            bg="#000000",
            highlightthickness=2,
            highlightbackground="#00ffff",
            bd=0,
        )
        self.loupe.create_line(
            self.LOUPE_SIZE // 2,
            0,
            self.LOUPE_SIZE // 2,
            self.LOUPE_SIZE,
            fill="#ff4d4d",
            width=1,
            tags="crosshair",
        )
        self.loupe.create_line(
            0,
            self.LOUPE_SIZE // 2,
            self.LOUPE_SIZE,
            self.LOUPE_SIZE // 2,
            fill="#ff4d4d",
            width=1,
            tags="crosshair",
        )
        self.loupe.place_forget()

    def _build_toolbar(self, parent):
        left = tk.Frame(parent, bg="#222222")
        left.pack(side="left", fill="y", padx=10)

        self._toolbar_field(left, "DATE", self.date_var, width=12).pack(side="left", padx=(0, 12), pady=9)
        self._toolbar_field(left, "END DATE", self.end_date_var, width=12).pack(side="left", padx=(0, 12), pady=9)
        self._toolbar_field(left, "EDITION", self.edition_var, width=8).pack(side="left", pady=9)

        spacer = tk.Frame(parent, bg="#222222")
        spacer.pack(side="left", fill="both", expand=True)

        self.toolbar_status = tk.Label(
            spacer,
            text="No unsaved changes.",
            bg="#222222",
            fg="#cccccc",
            font=("Helvetica", 11),
        )
        self.toolbar_status.pack(side="right", padx=(0, 14))

        buttons = tk.Frame(parent, bg="#222222")
        buttons.pack(side="right", padx=10)
        self._button(buttons, "Close", self._close, bg="#6c757d").pack(side="right", pady=8)
        self.save_button = self._button(buttons, "Save (Space)", self._save, bg="#28a745", state="disabled")
        self.save_button.pack(side="right", padx=(0, 8), pady=8)

    def _bind_events(self):
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _e: self._close())
        self.bind("<space>", self._save)
        self.bind("<Command-s>", self._save)
        self.bind("<Control-s>", self._save)
        self.bind("<Return>", lambda _e: self.confirm_point())
        self.bind("<Up>", lambda e: self._nudge_active_point(0, -1, 10 if (e.state & 0x1) else 1))
        self.bind("<Down>", lambda e: self._nudge_active_point(0, 1, 10 if (e.state & 0x1) else 1))
        self.bind("<Left>", lambda e: self._nudge_active_point(-1, 0, 10 if (e.state & 0x1) else 1))
        self.bind("<Right>", lambda e: self._nudge_active_point(1, 0, 10 if (e.state & 0x1) else 1))

        self.canvas.bind("<Configure>", lambda _e: self._schedule_render())
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._drag_pan)
        self.canvas.bind("<ButtonRelease-1>", self._finish_pointer_action)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.fit_image())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        variables = [
            self.date_var,
            self.end_date_var,
            self.edition_var,
            *self.bounds_vars.values(),
            *self.proj_vars.values(),
        ]
        for corner in CORNER_KEYS:
            variables.extend(self.gcp_vars[corner].values())
        for variable in variables:
            variable.trace_add("write", self._update_dirty_state)

    def _panel(self, parent):
        return tk.Frame(parent, bg="#2a2a2a", padx=10, pady=10)

    def _panel_title(self, parent, text):
        tk.Label(
            parent,
            text=text.upper(),
            bg="#2a2a2a",
            fg="#8f8f8f",
            font=("Helvetica", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

    def _meta_pair(self, parent, label, value):
        tk.Label(parent, text=label, bg="#2a2a2a", fg="#aaaaaa", font=("Helvetica", 9), anchor="w").pack(fill="x")
        tk.Label(
            parent,
            text=value,
            bg="#2a2a2a",
            fg="#ffffff",
            font=("Helvetica", 11),
            anchor="w",
            justify="left",
            wraplength=self.SIDEBAR_WIDTH - 40,
        ).pack(fill="x", pady=(0, 8))

    def _sidebar_field(self, parent, label, variable, readonly=False):
        frame = tk.Frame(parent, bg="#2a2a2a")
        tk.Label(frame, text=label, bg="#2a2a2a", fg="#aaaaaa", font=("Helvetica", 9), anchor="w").pack(fill="x")
        entry = tk.Entry(
            frame,
            textvariable=variable,
            bg="#222222",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            font=("Helvetica", 11),
            readonlybackground="#222222",
        )
        if readonly:
            entry.configure(state="readonly")
        entry.pack(fill="x", pady=(2, 0), ipady=3)
        return frame

    def _toolbar_field(self, parent, label, variable, width):
        wrapper = tk.Frame(parent, bg="#222222")
        tk.Label(wrapper, text=label, bg="#222222", fg="#aaaaaa", font=("Helvetica", 10, "bold")).pack(side="left", padx=(0, 6))
        entry = tk.Entry(
            wrapper,
            textvariable=variable,
            width=width,
            bg="#333333",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            justify="center",
            font=("Helvetica", 12),
        )
        entry.pack(side="left")
        return wrapper

    def _button(self, parent, text, command, bg, state="normal"):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=bg,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=("Helvetica", 11, "bold"),
            state=state,
        )

    def _badge(self, parent, text, bg):
        return tk.Label(
            parent,
            text=text,
            bg=bg,
            fg="#ffffff",
            padx=8,
            pady=3,
            font=("Helvetica", 9, "bold"),
        )

    def _close(self):
        if self.has_unsaved_changes():
            decision = messagebox.askyesnocancel(
                "Unsaved changes",
                "Save changes to this segment before closing?\n\nYes saves and closes.\nNo closes without saving.\nCancel keeps the editor open.",
            )
            if decision is None:
                return
            if decision and not self.save_changes(reload_after_save=True):
                return
        self.app.editor_windows.pop(self.editor_key, None)
        self.image_load_token += 1
        if self.render_after_id is not None:
            try:
                self.after_cancel(self.render_after_id)
            except Exception:
                pass
            self.render_after_id = None
        if self.high_quality_after_id is not None:
            try:
                self.after_cancel(self.high_quality_after_id)
            except Exception:
                pass
            self.high_quality_after_id = None
        self.destroy()

    def _open_in_quick_look(self):
        self.app.open_segment(self.seg)

    def _current_values(self):
        return {
            "date": self.date_var.get().strip(),
            "end_date": self.end_date_var.get().strip(),
            "edition": self.edition_var.get().strip(),
        }

    def _update_dirty_state(self, *_args):
        self._sync_lon0_from_bounds()
        is_dirty = self.has_unsaved_changes()
        self.save_button.configure(state="normal" if is_dirty else "disabled")
        if is_dirty:
            self.status_var.set("Ready. Space to Save.")
            self.toolbar_status.configure(text="Unsaved changes.", fg="#f0c674")
        else:
            self.status_var.set("Loaded. Edit metadata or place four points.")
            self.toolbar_status.configure(text="No unsaved changes.", fg="#cccccc")
        self.lock_button.configure(
            state="normal" if self.active_point is not None and len(self.locked_points) < 4 else "disabled"
        )

    def has_unsaved_changes(self):
        current = self._collect_updates(include_points=False)
        original = self._original_updates(include_points=False)
        points_dirty = self.locked_points != self.original_locked_points
        return current != original or points_dirty

    def save_changes(self, reload_after_save=True):
        if len(self.locked_points) not in {0, 4}:
            messagebox.showerror("Incomplete points", "Lock all four points before saving, or clear them all.")
            return False
        updates = self._collect_updates(include_points=True)
        validated = self.app._save_segment_editor_state(self.csv_path, self.seg, updates)
        if not validated:
            return False

        self.seg.start_date = validated["start_date"]
        self.seg.end_date = validated["end_date"]
        self.seg.edition = updates.get("edition") or "Unknown"
        self.row_data = normalize_editor_row(updates)
        self.original_metadata = {
            "date": updates.get("date", ""),
            "end_date": updates.get("end_date", ""),
            "edition": updates.get("edition", ""),
        }
        self.original_points_count = len(self.locked_points)
        self.original_locked_points = [dict(point) for point in self.locked_points]
        self._update_dirty_state()
        if reload_after_save:
            self.app._set_status(
                f"Saved metadata for {self.seg.location} (line {self.seg.line_num}); reloading..."
            )
            self.app.update_idletasks()
            reloaded = self.app.reload_csv(csv_path=self.csv_path)
            if reloaded:
                self.toolbar_status.configure(text="Saved. Timeline reloaded.", fg="#7CFC7C")
            else:
                self.toolbar_status.configure(text="Saved. Reload skipped.", fg="#7CFC7C")
        else:
            self.toolbar_status.configure(text="Saved. Reload pending.", fg="#7CFC7C")
        self.status_var.set("Saved.")
        self.save_button.configure(state="disabled")
        return True

    def _save(self, _event=None):
        self.save_changes(reload_after_save=True)
        return "break"

    def _row_has_locked_points(self):
        for corner in CORNER_KEYS:
            if _to_float(self.row_data.get(f"{corner}_x")) is None:
                return False
            if _to_float(self.row_data.get(f"{corner}_y")) is None:
                return False
        return True

    def _begin_async_image_load(self):
        if not self.has_local_file:
            self.image_loading = False
            self.status_var.set("Missing local file.")
            self._render_placeholder("Missing local file")
            return

        self.image_load_token += 1
        load_token = self.image_load_token
        path = self.seg.file_path

        self.preview_image = None
        self.tk_image = None
        self.zoom = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.image_loading = True
        self.overlay_lines = []
        self.last_render_size = None
        self.last_render_resample = None
        self.status_var.set("Loading image preview...")
        self._render_placeholder("Loading image preview...")

        def worker():
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    original_width, original_height = rgb.size
                    if rgb.width > MAX_PREVIEW_WIDTH:
                        ratio = MAX_PREVIEW_WIDTH / float(rgb.width)
                        preview_h = max(1, int(rgb.height * ratio))
                        rgb = rgb.resize((MAX_PREVIEW_WIDTH, preview_h), RESAMPLE_LANCZOS)
                    result = {
                        "preview_image": rgb,
                        "original_width": original_width,
                        "original_height": original_height,
                    }
            except Exception as exc:
                result = {"error": str(exc)}

            try:
                self.after(0, lambda: self._finish_async_image_load(load_token, result))
            except RuntimeError:
                return

        threading.Thread(target=worker, daemon=True).start()

    def _finish_async_image_load(self, load_token, result):
        if load_token != self.image_load_token or not self.winfo_exists():
            return

        self.image_loading = False
        error = result.get("error")
        if error:
            self.preview_image = None
            self.status_var.set(f"Image load failed: {error}")
            self._render_placeholder(f"Image load failed\n{error}")
            return

        self.preview_image = result["preview_image"]
        self.original_width = result["original_width"]
        self.original_height = result["original_height"]
        self.preview_scale_x = self.original_width / float(self.preview_image.width)
        self.preview_scale_y = self.original_height / float(self.preview_image.height)
        if self.has_saved_locked_points:
            self.locked_points = self._load_locked_points()
            self.original_points_count = len(self.locked_points)
            self.original_locked_points = [dict(point) for point in self.locked_points]

        self._update_dirty_state()
        self.after(0, self.fit_image)
        if len(self.locked_points) == 4:
            self.after(50, self.check_overlay)

    def _render_placeholder(self, message):
        self.canvas.delete("all")
        self.image_item = None
        self.loupe.place_forget()
        self.canvas.create_text(
            max(100, self.canvas.winfo_width() // 2),
            max(100, self.canvas.winfo_height() // 2),
            text=message,
            fill="#cccccc",
            font=("Helvetica", 16),
            justify="center",
        )

    def _schedule_render(self, fast=False):
        if self.render_after_id is not None:
            self.after_cancel(self.render_after_id)
        delay = self.FAST_RENDER_DELAY_MS if fast else 15
        resample = getattr(Image, "BILINEAR", RESAMPLE_LANCZOS) if fast else RESAMPLE_LANCZOS
        self.render_after_id = self.after(delay, lambda: self._render_image(resample))

    def _schedule_high_quality_render(self):
        if self.high_quality_after_id is not None:
            self.after_cancel(self.high_quality_after_id)
        self.high_quality_after_id = self.after(
            self.HIGH_QUALITY_RENDER_DELAY_MS,
            lambda: self._render_image(RESAMPLE_LANCZOS),
        )

    def _render_image(self, resample):
        self.render_after_id = None
        if resample == RESAMPLE_LANCZOS:
            self.high_quality_after_id = None
        if not self.preview_image:
            self._render_placeholder("Loading image preview..." if self.image_loading else "No image loaded")
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if canvas_w <= 1 or canvas_h <= 1:
            return

        if self.zoom is None:
            self._set_fit_zoom(canvas_w, canvas_h)

        resized_w = max(1, int(self.preview_image.width * self.zoom))
        resized_h = max(1, int(self.preview_image.height * self.zoom))
        render_size = (resized_w, resized_h)
        if self.last_render_size != render_size or self.last_render_resample != resample or self.tk_image is None:
            resized = self.preview_image.resize(render_size, resample)
            self.tk_image = ImageTk.PhotoImage(resized)
            self.last_render_size = render_size
            self.last_render_resample = resample

        if self.image_item is None:
            self.canvas.delete("all")
            self.image_item = self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfigure(self.image_item, image=self.tk_image)
            self.canvas.coords(self.image_item, self.pan_x, self.pan_y)
        self._render_overlay()

    def _set_fit_zoom(self, canvas_w=None, canvas_h=None):
        if not self.preview_image:
            return

        canvas_w = max(1, canvas_w or self.canvas.winfo_width())
        canvas_h = max(1, canvas_h or self.canvas.winfo_height())
        zoom = min(canvas_w / self.preview_image.width, canvas_h / self.preview_image.height)
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        self._center_image(canvas_w, canvas_h)

    def _center_image(self, canvas_w=None, canvas_h=None):
        if not self.preview_image or self.zoom is None:
            return
        canvas_w = max(1, canvas_w or self.canvas.winfo_width())
        canvas_h = max(1, canvas_h or self.canvas.winfo_height())
        img_w = self.preview_image.width * self.zoom
        img_h = self.preview_image.height * self.zoom
        self.pan_x = (canvas_w - img_w) / 2
        self.pan_y = (canvas_h - img_h) / 2

    def fit_image(self):
        if not self.preview_image:
            return
        self._set_fit_zoom()
        self._schedule_render()

    def actual_size(self):
        if not self.preview_image:
            return
        self.zoom = 1.0
        self._center_image()
        self._clamp_pan()
        self._schedule_render()

    def _start_pan(self, event):
        self.drag_origin = (event.x, event.y, self.pan_x, self.pan_y)
        self.is_dragging = False

    def _drag_pan(self, event):
        if not self.drag_origin or not self.preview_image:
            return
        start_x, start_y, pan_x, pan_y = self.drag_origin
        if abs(event.x - start_x) > 4 or abs(event.y - start_y) > 4:
            self.is_dragging = True
        self.pan_x = pan_x + (event.x - start_x)
        self.pan_y = pan_y + (event.y - start_y)
        self._clamp_pan()
        self._update_canvas_position()
        self._render_overlay()

    def _finish_pointer_action(self, event):
        if not self.drag_origin:
            return
        if not self.is_dragging:
            point = self._canvas_to_image_point(event.x, event.y)
            if point:
                self.active_point = point
                if len(self.locked_points) < 4:
                    self.status_var.set(f"Adjusting {self.CORNER_LABELS[len(self.locked_points)]}. Press Enter to lock.")
                self._update_dirty_state()
                self._render_overlay()
        self.drag_origin = None
        self.is_dragging = False

    def _on_mousewheel(self, event):
        if not self.preview_image:
            return "break"
        factor = 1.1 if event.delta > 0 else 0.9
        self._zoom_to(self.zoom * factor if self.zoom else factor, event.x, event.y)
        return "break"

    def _zoom_to(self, new_zoom, center_x, center_y):
        if not self.preview_image:
            return
        old_zoom = self.zoom or 1.0
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        image_x = (center_x - self.pan_x) / old_zoom
        image_y = (center_y - self.pan_y) / old_zoom
        self.zoom = new_zoom
        self.pan_x = center_x - (image_x * new_zoom)
        self.pan_y = center_y - (image_y * new_zoom)
        self._clamp_pan()
        self._schedule_render(fast=True)
        self._schedule_high_quality_render()

    def _clamp_pan(self):
        if not self.preview_image or self.zoom is None:
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        img_w = self.preview_image.width * self.zoom
        img_h = self.preview_image.height * self.zoom

        if img_w <= canvas_w:
            self.pan_x = (canvas_w - img_w) / 2
        else:
            self.pan_x = min(0, max(canvas_w - img_w, self.pan_x))

        if img_h <= canvas_h:
            self.pan_y = (canvas_h - img_h) / 2
        else:
            self.pan_y = min(0, max(canvas_h - img_h, self.pan_y))

    def _collect_updates(self, include_points):
        updates = {
            "filename": self.row_data.get("filename") or self.seg.tif_filename,
            "date": self.date_var.get().strip(),
            "end_date": self.end_date_var.get().strip(),
            "edition": self.edition_var.get().strip(),
            "top_bounds": self.bounds_vars["top"].get().strip(),
            "bottom_bounds": self.bounds_vars["bottom"].get().strip(),
            "left_bounds": self.bounds_vars["left"].get().strip(),
            "right_bounds": self.bounds_vars["right"].get().strip(),
            "proj_lat1": self.proj_vars["lat1"].get().strip(),
            "proj_lat2": self.proj_vars["lat2"].get().strip(),
            "proj_lat0": self.proj_vars["lat0"].get().strip(),
            "proj_lon0": self.proj_vars["lon0"].get().strip(),
        }
        for corner in CORNER_KEYS:
            updates[f"gcp_{corner}_lat"] = self.gcp_vars[corner]["lat"].get().strip()
            updates[f"gcp_{corner}_lon"] = self.gcp_vars[corner]["lon"].get().strip()

        updates["crs"] = build_lcc_crs(
            updates["proj_lat1"],
            updates["proj_lat2"],
            updates["proj_lat0"],
            updates["proj_lon0"],
        )

        if include_points:
            if len(self.locked_points) == 4:
                original_points = [self._preview_point_to_original(point) for point in self.locked_points]
                for corner, point in zip(CORNER_KEYS, original_points):
                    updates[f"{corner}_x"] = self._format_num(point["x"])
                    updates[f"{corner}_y"] = self._format_num(point["y"])
            else:
                for corner in CORNER_KEYS:
                    updates[f"{corner}_x"] = ""
                    updates[f"{corner}_y"] = ""
        return updates

    def _original_updates(self, include_points):
        updates = dict(self.row_data)
        if include_points:
            return updates
        filtered = dict(updates)
        for corner in CORNER_KEYS:
            filtered.pop(f"{corner}_x", None)
            filtered.pop(f"{corner}_y", None)
        return filtered

    def _load_locked_points(self):
        points = []
        for corner in CORNER_KEYS:
            x = _to_float(self.row_data.get(f"{corner}_x"))
            y = _to_float(self.row_data.get(f"{corner}_y"))
            if x is None or y is None:
                return []
            points.append(
                {
                    "x": x / self.preview_scale_x,
                    "y": y / self.preview_scale_y,
                }
            )
        return points

    def _preview_point_to_original(self, point):
        return {
            "x": float(point["x"]) * self.preview_scale_x,
            "y": float(point["y"]) * self.preview_scale_y,
        }

    def _format_num(self, value):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")

    def _current_bounds(self):
        top = _to_float(self.bounds_vars["top"].get())
        bottom = _to_float(self.bounds_vars["bottom"].get())
        left = _to_float(self.bounds_vars["left"].get())
        right = _to_float(self.bounds_vars["right"].get())
        if None in {top, bottom, left, right}:
            return None
        if left > 0:
            left = -left
        if right > 0:
            right = -right
        return {"top": top, "bottom": bottom, "left": left, "right": right}

    def _sync_lon0_from_bounds(self):
        if self._syncing_lon0:
            return
        bounds = self._current_bounds()
        lon0 = ""
        if bounds:
            lon0 = str((bounds["left"] + bounds["right"]) / 2.0)
        if self.proj_vars["lon0"].get() == lon0:
            return
        self._syncing_lon0 = True
        try:
            self.proj_vars["lon0"].set(lon0)
        finally:
            self._syncing_lon0 = False

    def _current_gcp_geo(self):
        points = []
        for corner in CORNER_KEYS:
            lat = _to_float(self.gcp_vars[corner]["lat"].get())
            lon = _to_float(self.gcp_vars[corner]["lon"].get())
            if lat is None or lon is None:
                return None
            if lon > 0:
                lon = -lon
            points.append((lon, lat))
        return points

    def _current_crs(self):
        return build_lcc_crs(
            self.proj_vars["lat1"].get(),
            self.proj_vars["lat2"].get(),
            self.proj_vars["lat0"].get(),
            self.proj_vars["lon0"].get(),
        )

    def _canvas_to_image_point(self, canvas_x, canvas_y):
        if not self.preview_image or self.zoom is None:
            return None
        image_x = (canvas_x - self.pan_x) / self.zoom
        image_y = (canvas_y - self.pan_y) / self.zoom
        if image_x < 0 or image_x > self.preview_image.width or image_y < 0 or image_y > self.preview_image.height:
            return None
        return {"x": image_x, "y": image_y}

    def _image_to_canvas_point(self, point):
        return (
            self.pan_x + (float(point["x"]) * self.zoom),
            self.pan_y + (float(point["y"]) * self.zoom),
        )

    def _draw_overlay(self):
        self._render_overlay()

    def _update_canvas_position(self):
        if self.image_item is not None:
            self.canvas.coords(self.image_item, self.pan_x, self.pan_y)

    def _render_overlay(self):
        self.canvas.delete("overlay")
        if self.zoom is None:
            self.loupe.place_forget()
            return

        for line in self.overlay_lines:
            coords = []
            for point in line:
                x, y = self._image_to_canvas_point(point)
                coords.extend([x, y])
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#ff4d4d", width=2, smooth=True, tags="overlay")

        if len(self.locked_points) > 1:
            poly = []
            for point in self.locked_points:
                x, y = self._image_to_canvas_point(point)
                poly.extend([x, y])
            self.canvas.create_line(*poly, fill="#ff4d4d", width=2, tags="overlay")

        for idx, point in enumerate(self.locked_points):
            self._draw_point_marker(point, idx + 1, fill="#ffd400")

        if self.active_point:
            self._draw_point_marker(self.active_point, len(self.locked_points) + 1, fill="#00e5ff")
            self._draw_active_crosshair(self.active_point)
            self._update_loupe()
        else:
            self.loupe.place_forget()

    def _draw_point_marker(self, point, index, fill):
        x, y = self._image_to_canvas_point(point)
        radius = max(4, self.POINT_RADIUS)
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline="#ff2d2d",
            width=2,
            tags="overlay",
        )
        self.canvas.create_text(
            x + 12,
            y - 12,
            text=str(index),
            fill="#ffff00",
            font=("Helvetica", 12, "bold"),
            tags="overlay",
        )

    def _draw_active_crosshair(self, point):
        x, y = self._image_to_canvas_point(point)
        size = self.CROSSHAIR_SIZE
        self.canvas.create_line(x - size, y, x + size, y, fill="#00e5ff", width=2, tags="overlay")
        self.canvas.create_line(x, y - size, x, y + size, fill="#00e5ff", width=2, tags="overlay")

    def _update_loupe(self):
        if not self.active_point or not self.preview_image:
            self.loupe.place_forget()
            return

        sample_size = max(8, int(self.LOUPE_SIZE / self.LOUPE_ZOOM))
        half = sample_size // 2
        left = int(round(self.active_point["x"])) - half
        top = int(round(self.active_point["y"])) - half
        max_left = max(0, self.preview_image.width - sample_size)
        max_top = max(0, self.preview_image.height - sample_size)
        left = max(0, min(left, max_left))
        top = max(0, min(top, max_top))
        right = min(self.preview_image.width, left + sample_size)
        bottom = min(self.preview_image.height, top + sample_size)

        crop = self.preview_image.crop((left, top, right, bottom))
        crop = crop.resize((self.LOUPE_SIZE, self.LOUPE_SIZE), getattr(Image, "NEAREST", RESAMPLE_LANCZOS))
        self.loupe_image = ImageTk.PhotoImage(crop)
        if self.loupe_image_item is None:
            self.loupe_image_item = self.loupe.create_image(0, 0, anchor="nw", image=self.loupe_image)
            self.loupe.tag_lower(self.loupe_image_item)
        else:
            self.loupe.itemconfigure(self.loupe_image_item, image=self.loupe_image)

        screen_x, screen_y = self._image_to_canvas_point(self.active_point)
        offset = 24
        margin = 10
        left_pos = int(screen_x + offset)
        top_pos = int(screen_y + offset)
        shell_w = max(1, self.canvas_shell.winfo_width())
        shell_h = max(1, self.canvas_shell.winfo_height())

        if left_pos + self.LOUPE_SIZE + margin > shell_w:
            left_pos = int(screen_x - self.LOUPE_SIZE - offset)
        if top_pos + self.LOUPE_SIZE + margin > shell_h:
            top_pos = int(screen_y - self.LOUPE_SIZE - offset)

        left_pos = max(margin, min(left_pos, shell_w - self.LOUPE_SIZE - margin))
        top_pos = max(margin, min(top_pos, shell_h - self.LOUPE_SIZE - margin))
        self.loupe.place(x=left_pos, y=top_pos)

    def confirm_point(self):
        if self.active_point is None or len(self.locked_points) >= 4:
            return "break"
        self.locked_points.append(dict(self.active_point))
        self.active_point = None
        if len(self.locked_points) == 4:
            self.status_var.set("4 points locked. Check overlay or save.")
            self.check_overlay()
        else:
            self.status_var.set(f"Tap near {self.CORNER_LABELS[len(self.locked_points)]}.")
        self._update_dirty_state()
        self._render_overlay()
        return "break"

    def undo_point(self):
        if self.active_point is not None:
            self.active_point = None
        elif self.locked_points:
            self.locked_points.pop()
        self.overlay_lines = []
        self.status_var.set("Point removed.")
        self._update_dirty_state()
        self._render_overlay()

    def _nudge_active_point(self, dx, dy, step):
        if self.active_point is None or not self.preview_image:
            return "break"
        self.active_point["x"] += dx * step
        self.active_point["y"] += dy * step
        self.active_point["x"] = max(0.0, min(self.preview_image.width, self.active_point["x"]))
        self.active_point["y"] = max(0.0, min(self.preview_image.height, self.active_point["y"]))
        self._render_overlay()
        return "break"

    def check_overlay(self):
        if len(self.locked_points) < 4:
            self.status_var.set("Need four locked points for overlay preview.")
            self.overlay_lines = []
            self._render_overlay()
            return
        bounds = self._current_bounds()
        if not bounds:
            self.status_var.set("Map extents are incomplete.")
            self.overlay_lines = []
            self._render_overlay()
            return
        crs = self._current_crs()
        if not crs:
            self.status_var.set("LCC params are incomplete.")
            self.overlay_lines = []
            self._render_overlay()
            return
        try:
            self.overlay_lines = self._compute_projection_lines(
                self.locked_points,
                self._current_gcp_geo(),
                bounds,
                crs,
            )
            self.status_var.set("Overlay preview updated.")
        except Exception as exc:
            self.overlay_lines = []
            self.status_var.set(f"Overlay failed: {exc}")
        self._render_overlay()

    def _compute_projection_lines(self, clicks, gcp_geo, bounds, crs):
        corners = [
            (bounds["left"], bounds["top"]),
            (bounds["right"], bounds["top"]),
            (bounds["right"], bounds["bottom"]),
            (bounds["left"], bounds["bottom"]),
        ]
        geo_anchors = gcp_geo or corners
        if len(geo_anchors) != 4:
            raise RuntimeError("Need four GCP geo points or complete bounds.")

        meter_anchors = np.array(self._project_lonlat_points(crs, geo_anchors), dtype=float)
        pixel_anchors = np.array([[float(point["x"]), float(point["y"])] for point in clicks], dtype=float)
        A = np.hstack([meter_anchors, np.ones((4, 1))])
        coeff_x, _, _, _ = np.linalg.lstsq(A, pixel_anchors[:, 0], rcond=None)
        coeff_y, _, _, _ = np.linalg.lstsq(A, pixel_anchors[:, 1], rcond=None)

        lines = []
        indices = [0, 1, 2, 3, 0]
        for idx in range(4):
            start = corners[indices[idx]]
            end = corners[indices[idx + 1]]
            lons = np.linspace(start[0], end[0], 50)
            lats = np.linspace(start[1], end[1], 50)
            projected = np.array(self._project_lonlat_points(crs, list(zip(lons, lats))), dtype=float)
            input_m = np.hstack([projected, np.ones((50, 1))])
            px = input_m @ coeff_x
            py = input_m @ coeff_y
            lines.append([{"x": float(x), "y": float(y)} for x, y in zip(px, py)])
        return lines

    def _project_lonlat_points(self, crs, lonlat_points):
        if not lonlat_points:
            return []
        input_text = "".join(f"{lon} {lat}\n" for lon, lat in lonlat_points)
        result = subprocess.run(
            ["gdaltransform", "-s_srs", "EPSG:4269", "-t_srs", crs],
            input=input_text,
            capture_output=True,
            text=True,
            check=True,
        )
        points = []
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                points.append((float(parts[0]), float(parts[1])))
        if len(points) != len(lonlat_points):
            raise RuntimeError("gdaltransform returned incomplete output")
        return points


def build_rows(locations_data, file_index):
    all_dates = [r["date"] for records in locations_data.values() for r in records if r.get("date")]
    if not all_dates:
        raise RuntimeError("No valid dates found in CSV.")

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()
    if min_date == max_date:
        # Avoid zero-width timelines.
        max_date = max_date + timedelta(days=1)

    rows = []
    for location in sorted(locations_data.keys(), key=lambda name: (name.lower(), name)):
        records = [
            r for r in locations_data[location]
            if r.get("date") and r.get("end_date")
        ]
        if not records:
            continue
        records.sort(key=lambda r: r["date"])

        segments = []
        for rec in records:
            file_path = resolve_tif_file(rec.get("tif_filename", ""), file_index)
            segments.append(
                Segment(
                    location=location,
                    edition=str(rec.get("edition", "")),
                    start_date=rec["date"].date(),
                    end_date=rec["end_date"].date(),
                    tif_filename=rec.get("tif_filename", ""),
                    file_path=file_path,
                    line_num=int(rec.get("line") or 0),
                    has_georef_data=_record_has_gcp_data(rec),
                )
            )

        rows.append(
            {
                "location": location,
                "segments": segments,
                "has_state_name": _location_has_state_suffix(location),
            }
        )

    return rows, min_date, max_date


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive timeline viewer that opens chart TIFFs in Quick Look on click.",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"Input CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--tif-dir", default=DEFAULT_TIF_DIR, help=f"TIFF root directory (default: {DEFAULT_TIF_DIR})")
    return parser.parse_args()


def main():
    args = parse_args()
    tif_dir = _default_tif_dir_for_csv(args.csv, primary_tif_dir=args.tif_dir)

    print(f"Loading CSV: {args.csv}")
    errors, locations_data = analyze_csv(args.csv)
    if errors:
        print(f"Found {len(errors)} CSV issue(s); continuing with timeline data.", file=sys.stderr)
        for err in errors[:10]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)

    print(f"Indexing local TIFFs recursively: {tif_dir}")
    file_index = build_file_index(tif_dir)
    print(f"Indexed {len(file_index)} lookup keys")

    rows, min_date, max_date = build_rows(locations_data, file_index)
    print(f"Loaded {len(rows)} locations, {sum(len(r['segments']) for r in rows)} timeline segments")

    app = TimelinePreviewApp(
        rows=rows,
        min_date=min_date,
        max_date=max_date,
        csv_path=args.csv,
        tif_dir=tif_dir,
        file_index=file_index,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
