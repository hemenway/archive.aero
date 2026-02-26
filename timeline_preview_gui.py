#!/usr/bin/env python3
"""
Interactive local timeline viewer for sectional chart TIFFs.

- Reads master_dole-style CSV via analyze_dole.py helpers
- Resolves files from /Volumes/drive/newrawtiffs recursively
- Lets you click a timeline segment to open the matched TIFF in Quick Look
"""

import argparse
import csv
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import date, timedelta
from tkinter import messagebox, simpledialog, ttk

from analyze_dole import analyze_csv, build_file_index, resolve_tif_file


DEFAULT_CSV = "/Users/ryanhemenway/archive.aero/master_dole.csv"
EARLY_CSV = "/Users/ryanhemenway/archive.aero/early_master_dole.csv"
CSV_PRESETS = [DEFAULT_CSV, EARLY_CSV]
DEFAULT_TIF_DIR = "/Volumes/drive/newrawtiffs"
EARLY_TIF_DIR = "/Volumes/projects/rawtiffs"


def _is_early_csv_path(csv_path):
    return os.path.basename((csv_path or "").strip()).lower() == os.path.basename(EARLY_CSV).lower()


def _default_tif_dir_for_csv(csv_path, primary_tif_dir=DEFAULT_TIF_DIR):
    if _is_early_csv_path(csv_path):
        return EARLY_TIF_DIR
    return primary_tif_dir


def update_csv_edition_by_line(csv_path, line_num, new_edition):
    """
    Update the edition value for a CSV row identified by the GUI's stored line number.

    Uses the same row numbering convention as analyze_csv(): header is line 1,
    first data row is line 2.
    """
    if not line_num or line_num < 2:
        raise ValueError(f"Invalid CSV line number: {line_num}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        preferred_key = None
        if "edition" in fieldnames:
            preferred_key = "edition"
        elif "Edition" in fieldnames:
            preferred_key = "Edition"
        else:
            raise RuntimeError(f"No editable edition column found in CSV: {csv_path}")

        rows = []
        found = False
        for row_line_num, row in enumerate(reader, start=2):
            if row_line_num == line_num:
                row[preferred_key] = new_edition
                found = True
            rows.append(row)

    if not found:
        raise RuntimeError(f"CSV row not found at line {line_num}: {csv_path}")

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


class TimelinePreviewApp(tk.Tk):
    LABEL_WIDTH = 190
    HEADER_HEIGHT = 32
    ROW_HEIGHT = 24
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

        self._build_ui()
        self._bind_events()
        self._refresh_summary()
        self.redraw()
        self._set_status("Ready. Left-click opens in Quick Look. Right-click edits edition.")

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

        ttk.Button(top, text="Reload CSV", command=self.reload_selected_csv).pack(side="left", padx=(0, 10))

        ttk.Label(top, textvariable=self.summary_var).pack(side="left", padx=(8, 0))

        meta = ttk.Frame(self, padding=(10, 0, 10, 6))
        meta.pack(fill="x")
        ttk.Label(meta, textvariable=self.meta_var).pack(side="left")
        self._refresh_meta()

        legend = ttk.Frame(self, padding=(10, 0, 10, 6))
        legend.pack(fill="x")
        self._legend_chip(legend, "#2f80ed", "Known edition").pack(side="left")
        self._legend_chip(legend, "#f39c12", "Unknown edition").pack(side="left", padx=(10, 0))
        self._legend_chip(legend, "#cfd8dc", "Missing local file").pack(side="left", padx=(10, 0))

        main = ttk.Frame(self, padding=(10, 0, 10, 0))
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg="white", highlightthickness=0)
        y_scroll = ttk.Scrollbar(main, orient="vertical", command=self.canvas.yview)
        x_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
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
            lambda _e: self._set_status("Ready. Left-click opens in Quick Look. Right-click edits edition."),
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
        self.reload_selected_csv()
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
        return "break"

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

    def _get_or_build_file_index(self, tif_dir):
        if tif_dir not in self.file_index_cache:
            self._set_status(f"Indexing local TIFFs recursively: {tif_dir}")
            self.update_idletasks()
            self.file_index_cache[tif_dir] = build_file_index(tif_dir)
        return self.file_index_cache[tif_dir]

    def redraw(self):
        self.canvas.delete("all")
        self.item_to_segment.clear()
        self.selected_item = None
        self.selected_segment = None

        width = self.LABEL_WIDTH + self.timeline_width + self.TIMELINE_PAD_RIGHT
        height = self.HEADER_HEIGHT + max(1, len(self.rows_filtered)) * self.ROW_HEIGHT + 10

        self._draw_background(width, height)
        self._draw_header()
        self._draw_rows()

        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.canvas.xview_moveto(0)

    def reload_selected_csv(self):
        target_csv = (self.csv_choice_var.get() or self.csv_path).strip()
        self.reload_csv(csv_path=target_csv)

    def reload_csv(self, csv_path=None):
        target_csv = (csv_path or self.csv_path).strip()
        if not target_csv:
            self._set_status("Reload failed: no CSV path selected")
            return
        target_tif_dir = self._preferred_tif_dir_for_csv(target_csv)

        self._set_status(f"Reloading CSV: {target_csv}")
        self.update_idletasks()

        try:
            file_index = self._get_or_build_file_index(target_tif_dir)
            errors, locations_data = analyze_csv(target_csv)
            rows, min_date, max_date = build_rows(locations_data, file_index)
        except Exception as exc:
            messagebox.showerror("Reload failed", f"Failed to reload CSV:\n\n{target_csv}\n\n{exc}")
            self._set_status(f"Reload failed: {exc}")
            return

        self.rows_all = rows
        self.min_date = min_date
        self.max_date = max_date
        self.total_days = max(1, (max_date - min_date).days)
        self.timeline_width = max(1800, int(self.total_days * self.DAY_PX))
        self.csv_path = target_csv
        self.tif_dir = target_tif_dir
        self.file_index = file_index
        if target_csv not in self.csv_options:
            self.csv_options.append(target_csv)
            self.csv_combo.configure(values=self.csv_options)
        self.csv_choice_var.set(target_csv)
        self._refresh_meta()

        q = self.filter_var.get().strip().lower()
        if not q:
            self.rows_filtered = self.rows_all
        else:
            self.rows_filtered = [row for row in self.rows_all if q in row["location"].lower()]

        self._refresh_summary()
        self.redraw()

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

    def _draw_background(self, width, height):
        self.canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="")
        self.canvas.create_rectangle(0, 0, width, self.HEADER_HEIGHT, fill="#f5f7fa", outline="")
        self.canvas.create_line(self.LABEL_WIDTH, 0, self.LABEL_WIDTH, height, fill="#b0bec5")

    def _draw_header(self):
        # Year ticks
        for y in range(self.min_date.year, self.max_date.year + 1):
            year_date = date(y, 1, 1)
            if year_date < self.min_date:
                year_date = self.min_date
            offset = (year_date - self.min_date).days
            x = self.LABEL_WIDTH + int(offset * self.DAY_PX)
            self.canvas.create_line(x, 0, x, self.HEADER_HEIGHT + len(self.rows_filtered) * self.ROW_HEIGHT,
                                    fill="#eceff1")
            self.canvas.create_text(x + 4, self.HEADER_HEIGHT // 2, anchor="w",
                                    text=str(y), fill="#455a64", font=("Helvetica", 10, "bold"))

        self.canvas.create_text(10, self.HEADER_HEIGHT // 2, anchor="w",
                                text="Location", fill="#263238", font=("Helvetica", 10, "bold"))

    def _draw_rows(self):
        if not self.rows_filtered:
            self.canvas.create_text(
                self.LABEL_WIDTH + 20,
                self.HEADER_HEIGHT + 30,
                anchor="w",
                text="No matching locations",
                fill="#607d8b",
                font=("Helvetica", 12),
            )
            return

        for i, row in enumerate(self.rows_filtered):
            y = self.HEADER_HEIGHT + i * self.ROW_HEIGHT
            y_mid = y + self.ROW_HEIGHT // 2
            if i % 2 == 0:
                self.canvas.create_rectangle(0, y, self.LABEL_WIDTH + self.timeline_width + self.TIMELINE_PAD_RIGHT,
                                             y + self.ROW_HEIGHT, fill="#fcfdff", outline="")
            self.canvas.create_line(0, y + self.ROW_HEIGHT, self.LABEL_WIDTH + self.timeline_width + self.TIMELINE_PAD_RIGHT,
                                    y + self.ROW_HEIGHT, fill="#f0f3f6")
            self.canvas.create_text(8, y_mid, anchor="w", text=row["location"], fill="#263238",
                                    font=("Helvetica", 10))

            is_early_csv = _is_early_csv_path(self.csv_path)
            for seg in row["segments"]:
                x1, x2 = self._segment_x(seg)
                y1 = y + 4
                y2 = y + self.ROW_HEIGHT - 4

                if seg.file_path and os.path.exists(seg.file_path):
                    fill = "#f39c12" if seg.edition == "Unknown" else "#2f80ed"
                    outline = "#1f2d3d"
                else:
                    fill = "#cfd8dc"
                    outline = "#90a4ae"

                item = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=1)
                self.item_to_segment[item] = seg

                label_text = str(seg.edition or "")
                label_width = x2 - x1
                min_label_px = 16 if is_early_csv else 24
                label_font = ("Helvetica", 7, "bold") if is_early_csv else ("Helvetica", 8, "bold")

                # Early-master segments are much shorter (fixed 56-day spans), so use a
                # smaller font and lower threshold to show known edition numbers on-block.
                if is_early_csv and label_text == "Unknown":
                    label_text = ""

                if label_width >= min_label_px and label_text:
                    self.canvas.create_text((x1 + x2) / 2, y_mid, text=label_text,
                                            fill="white" if fill in {"#2f80ed", "#f39c12"} else "#263238",
                                            font=label_font)

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
            self._set_status("Ready. Left-click opens in Quick Look. Right-click edits edition.")

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
        self.edit_segment_edition(seg)
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

    def edit_segment_edition(self, seg):
        current_edition = "" if str(seg.edition) == "Unknown" else str(seg.edition or "")
        prompt = (
            f"Edit edition for {seg.location}\n"
            f"{seg.start_date.isoformat()} -> {seg.end_date.isoformat()}\n"
            f"CSV line: {seg.line_num}\n\n"
            "Enter edition number (blank clears it):"
        )
        value = simpledialog.askstring(
            "Edit Edition",
            prompt,
            initialvalue=current_edition,
            parent=self,
        )
        if value is None:
            return

        new_edition = value.strip()
        try:
            update_csv_edition_by_line(self.csv_path, seg.line_num, new_edition)
        except Exception as exc:
            messagebox.showerror("Save failed", f"Failed to save edition to CSV:\n\n{self.csv_path}\n\n{exc}")
            self._set_status(f"Edition save failed: {exc}")
            return

        self._set_status(
            f"Saved edition '{new_edition or 'Unknown'}' for {seg.location} (line {seg.line_num}); reloading..."
        )
        self.update_idletasks()
        self.reload_csv(csv_path=self.csv_path)

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
                )
            )

        rows.append({"location": location, "segments": segments})

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
