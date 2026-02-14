#!/usr/bin/env python3
"""
Quick verification tool for master_dole.csv
Displays chart images full-screen with editable metadata overlay.
Run: python3 verify_dole.py
"""

import csv
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CSV_FILE = SCRIPT_DIR / 'master_dole.csv'
SOURCE_DIR = Path('/Volumes/drive/newrawtiffs')
CACHE_DIR = SCRIPT_DIR / '.verify_cache'
PORT = 8765
PREFETCH_COUNT = 3
MAX_DIM = 4096
JPEG_QUALITY = 85

# --- Global state ---
records = []
records_lock = threading.Lock()
prefetch_executor = None
converting = set()
converting_lock = threading.Lock()


def load_csv():
    """Load CSV records into memory."""
    global records
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    return records


def save_csv():
    """Write all records back to CSV."""
    with records_lock:
        if not records:
            return
        fieldnames = list(records[0].keys())
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def cache_path_for(tif_filename):
    """Get the JPEG cache path for a given tif_filename."""
    h = hashlib.md5(tif_filename.encode()).hexdigest()
    return CACHE_DIR / f"{h}.jpg"


def resolve_source_file(record):
    """Find the actual source file on disk for a record.
    Returns (path, filetype) or (None, None)."""
    tif_filename = record.get('tif_filename', '')
    download_link = record.get('download_link', '')

    if not tif_filename:
        return None, None

    # Direct match
    direct = SOURCE_DIR / tif_filename
    if direct.exists():
        ext = direct.suffix.lower()
        return direct, ext

    # Determine the real extension from download_link
    parsed = urlparse(download_link)
    url_ext = Path(parsed.path).suffix.lower()

    # Try with the URL's extension
    if url_ext and url_ext != direct.suffix.lower():
        alt = direct.with_suffix(url_ext)
        if alt.exists():
            return alt, url_ext

    # For ZIPs: check extracted directory
    if url_ext == '.zip' or tif_filename.endswith('.zip'):
        zip_path = SOURCE_DIR / tif_filename
        if not zip_path.exists():
            zip_path = direct.with_suffix('.zip')
        stem = zip_path.stem
        extract_dir = SOURCE_DIR / stem
        if extract_dir.is_dir():
            tifs = list(extract_dir.glob('**/*.tif')) + list(extract_dir.glob('**/*.tiff'))
            if tifs:
                return tifs[0], '.tif'

    # For PDFs: the tif_filename may have .tif but actual file is .pdf
    if url_ext == '.pdf':
        pdf_name = tif_filename
        if pdf_name.endswith('.tif'):
            pdf_name = pdf_name[:-4] + '.pdf'
        pdf_path = SOURCE_DIR / pdf_name
        if pdf_path.exists():
            return pdf_path, '.pdf'

    return None, None


def convert_image(record_index):
    """Convert a source image to a cached JPEG preview. Returns cache path or None."""
    if record_index < 0 or record_index >= len(records):
        return None

    record = records[record_index]
    tif_filename = record.get('tif_filename', '')
    if not tif_filename:
        return None

    cached = cache_path_for(tif_filename)
    if cached.exists():
        return cached

    # Prevent duplicate conversions
    with converting_lock:
        if tif_filename in converting:
            return None
        converting.add(tif_filename)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        source_path, filetype = resolve_source_file(record)

        if source_path is None:
            return None

        if filetype in ('.tif', '.tiff'):
            return _convert_tif(source_path, cached)
        elif filetype == '.pdf':
            return _convert_pdf(source_path, cached)
        elif filetype == '.zip':
            return _convert_zip(source_path, cached)
        else:
            return None
    except Exception as e:
        print(f"  [convert] Error for record {record_index}: {e}")
        return None
    finally:
        with converting_lock:
            converting.discard(tif_filename)


def _convert_tif(source, dest):
    """Convert a GeoTIFF to JPEG using gdal_translate.
    Handles paletted (1-band + color table) and RGB images."""
    # Try with -expand rgb first (required for paletted images)
    try:
        result = subprocess.run([
            'gdal_translate', '-of', 'JPEG',
            '-expand', 'rgb',
            '-outsize', str(MAX_DIM), '0',
            '-co', f'QUALITY={JPEG_QUALITY}',
            '-q',
            str(source), str(dest)
        ], capture_output=True, timeout=120)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception:
        pass

    # Fallback: without -expand (for already-RGB images)
    if dest.exists():
        dest.unlink()
    try:
        result = subprocess.run([
            'gdal_translate', '-of', 'JPEG',
            '-outsize', str(MAX_DIM), '0',
            '-co', f'QUALITY={JPEG_QUALITY}',
            '-q',
            str(source), str(dest)
        ], capture_output=True, timeout=120)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception:
        pass

    # Last resort: sips
    if dest.exists():
        dest.unlink()
    try:
        subprocess.run([
            'sips', '-s', 'format', 'jpeg',
            '-Z', str(MAX_DIM),
            str(source), '--out', str(dest)
        ], capture_output=True, timeout=120)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception:
        pass
    return None


def _convert_pdf(source, dest):
    """Convert a PDF to JPEG using sips or qlmanage."""
    try:
        subprocess.run([
            'sips', '-s', 'format', 'jpeg',
            '-Z', str(MAX_DIM),
            str(source), '--out', str(dest)
        ], capture_output=True, timeout=120)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
    except Exception:
        pass

    # Fallback: qlmanage
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run([
                'qlmanage', '-t', '-s', str(MAX_DIM),
                '-o', tmpdir, str(source)
            ], capture_output=True, timeout=60)
            for f in Path(tmpdir).iterdir():
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    shutil.copy2(f, dest)
                    if dest.exists():
                        return dest
    except Exception:
        pass
    return None


def _convert_zip(source, dest):
    """Extract image from ZIP and convert. Handles TIF and PDF contents."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(source) as zf:
                names = zf.namelist()
                tif_names = [n for n in names
                             if n.lower().endswith(('.tif', '.tiff'))]
                pdf_names = [n for n in names
                             if n.lower().endswith('.pdf')]
                if tif_names:
                    extracted = zf.extract(tif_names[0], tmpdir)
                    return _convert_tif(Path(extracted), dest)
                elif pdf_names:
                    extracted = zf.extract(pdf_names[0], tmpdir)
                    return _convert_pdf(Path(extracted), dest)
    except Exception:
        pass
    return None


def prefetch_images(current_index, filtered_indices):
    """Pre-convert upcoming images in background."""
    for offset in range(1, PREFETCH_COUNT + 1):
        pos = None
        # Find the next filtered index
        try:
            cur_pos = filtered_indices.index(current_index)
            if cur_pos + offset < len(filtered_indices):
                pos = filtered_indices[cur_pos + offset]
        except ValueError:
            pos = current_index + offset

        if pos is not None and 0 <= pos < len(records):
            cached = cache_path_for(records[pos].get('tif_filename', ''))
            if not cached.exists():
                t = threading.Thread(target=convert_image, args=(pos,), daemon=True)
                t.start()


# --- HTML Template ---
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chart Verification Tool</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background:#111; color:#eee;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    overflow:hidden; height:100vh; width:100vw;
    user-select:none;
}

/* Image viewer */
#viewer {
    position:absolute; inset:0;
    display:flex; align-items:center; justify-content:center;
    cursor:grab; overflow:hidden;
}
#viewer.dragging { cursor:grabbing; }
#chart-img {
    transform-origin:0 0;
    max-width:none; max-height:none;
    image-rendering:auto;
    transition: none;
}
#loading {
    position:absolute; inset:0;
    display:flex; align-items:center; justify-content:center;
    font-size:18px; color:#888;
    pointer-events:none; z-index:5;
}
#loading.hidden { display:none; }
#error-msg {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    background:rgba(200,50,50,0.9); padding:16px 24px; border-radius:8px;
    font-size:14px; z-index:6; display:none;
}

/* Bottom bar */
#bar {
    position:fixed; bottom:0; left:0; right:0;
    background:rgba(20,20,30,0.92);
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    border-top:1px solid rgba(255,255,255,0.1);
    padding:10px 16px;
    z-index:100;
    display:flex; flex-wrap:wrap; align-items:center; gap:8px;
    font-size:13px;
}
#bar .field-group {
    display:flex; align-items:center; gap:4px;
}
#bar label {
    color:#999; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;
    white-space:nowrap;
}
#bar input, #bar select {
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
    color:#fff; padding:4px 8px; border-radius:4px; font-size:13px;
    font-family:inherit; outline:none;
}
#bar input:focus, #bar select:focus {
    border-color:#4a9eff; box-shadow:0 0 0 2px rgba(74,158,255,0.3);
}
#bar input.changed {
    border-color:#ff9800; background:rgba(255,152,0,0.15);
}
.field-edition { width:70px; }
.field-date { width:110px; }
.field-interval { width:55px; }
.field-location { width:140px; }

/* Navigation */
#nav {
    display:flex; align-items:center; gap:6px; margin-left:auto;
}
#nav button, .action-btn {
    background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2);
    color:#eee; padding:5px 12px; border-radius:4px; cursor:pointer;
    font-size:12px; font-family:inherit; white-space:nowrap;
}
#nav button:hover, .action-btn:hover {
    background:rgba(255,255,255,0.2);
}
.action-btn.save-btn {
    background:rgba(74,158,255,0.3); border-color:rgba(74,158,255,0.5);
}
.action-btn.save-btn:hover {
    background:rgba(74,158,255,0.5);
}
#counter {
    color:#888; font-size:12px; white-space:nowrap; min-width:90px; text-align:center;
}

/* Top filter bar */
#filters {
    position:fixed; top:0; left:0; right:0;
    background:rgba(20,20,30,0.88);
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    border-bottom:1px solid rgba(255,255,255,0.08);
    padding:8px 16px;
    z-index:100;
    display:flex; align-items:center; gap:12px;
    font-size:12px;
}
#filters select {
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
    color:#fff; padding:4px 8px; border-radius:4px; font-size:12px;
    font-family:inherit; outline:none;
}
#filters select:focus {
    border-color:#4a9eff;
}
#filters label {
    color:#999; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;
}
#status-text {
    margin-left:auto; color:#666; font-size:11px;
}
#dup-badge {
    background:rgba(255,152,0,0.3); color:#ffb74d;
    padding:2px 8px; border-radius:10px; font-size:11px;
    display:none;
}

/* Help overlay */
#help {
    position:fixed; inset:0; z-index:200;
    background:rgba(0,0,0,0.85);
    display:none; align-items:center; justify-content:center;
}
#help.visible { display:flex; }
#help-content {
    background:#1a1a2e; border:1px solid rgba(255,255,255,0.1);
    border-radius:12px; padding:30px 40px; max-width:500px; width:90%;
}
#help-content h2 { margin-bottom:16px; color:#4a9eff; font-size:18px; }
#help-content table { width:100%; }
#help-content td { padding:4px 0; }
#help-content td:first-child {
    font-family:monospace; color:#ffb74d; padding-right:20px; white-space:nowrap;
}
#help-content td:last-child { color:#ccc; }

/* Link display */
#link-display {
    position:fixed; bottom:52px; left:0; right:0;
    background:rgba(20,20,30,0.85); padding:4px 16px;
    font-size:10px; color:#666; z-index:99;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    border-top:1px solid rgba(255,255,255,0.05);
    display:none;
}
#link-display a { color:#4a9eff; text-decoration:none; }
#link-display a:hover { text-decoration:underline; }
</style>
</head>
<body>

<!-- Top filter bar -->
<div id="filters">
    <label>Location</label>
    <select id="filter-location"><option value="">All</option></select>
    <label>Edition</label>
    <select id="filter-edition">
        <option value="">All</option>
        <option value="unknown">Unknown</option>
        <option value="known">Known</option>
    </select>
    <span id="dup-badge"></span>
    <span id="status-text"></span>
</div>

<!-- Image viewer -->
<div id="viewer">
    <img id="chart-img" draggable="false">
    <div id="loading">Loading image...</div>
    <div id="error-msg"></div>
</div>

<!-- Link display -->
<div id="link-display"><a id="source-link" href="#" target="_blank"></a></div>

<!-- Bottom metadata bar -->
<div id="bar">
    <div class="field-group">
        <label>Loc</label>
        <input id="f-location" class="field-location" tabindex="1">
    </div>
    <div class="field-group">
        <label>Date</label>
        <input id="f-date" class="field-date" tabindex="2" placeholder="YYYY-MM-DD">
    </div>
    <div class="field-group">
        <label>End</label>
        <input id="f-end-date" class="field-date" tabindex="3" placeholder="YYYY-MM-DD">
    </div>
    <div class="field-group">
        <label>Ed</label>
        <input id="f-edition" class="field-edition" tabindex="4">
    </div>
    <div class="field-group">
        <label>Int</label>
        <input id="f-interval" class="field-interval" tabindex="5">
    </div>
    <button class="action-btn save-btn" onclick="saveRecord()" tabindex="6">Save</button>
    <div id="nav">
        <button onclick="navigate(-1)">&larr;</button>
        <span id="counter">0 / 0</span>
        <button onclick="navigate(1)">&rarr;</button>
    </div>
</div>

<!-- Help overlay -->
<div id="help">
    <div id="help-content">
        <h2>Keyboard Shortcuts</h2>
        <table>
            <tr><td>&larr; K / &rarr; J</td><td>Previous / Next record</td></tr>
            <tr><td>Tab / Shift+Tab</td><td>Next / Previous field</td></tr>
            <tr><td>Enter</td><td>Save changes</td></tr>
            <tr><td>Escape</td><td>Cancel edits / Close help</td></tr>
            <tr><td>H</td><td>Toggle this help</td></tr>
            <tr><td>L</td><td>Toggle source link</td></tr>
            <tr><td>R</td><td>Reset zoom to fit</td></tr>
            <tr><td>Scroll wheel</td><td>Zoom in/out</td></tr>
            <tr><td>Click + Drag</td><td>Pan image</td></tr>
        </table>
    </div>
</div>

<script>
// --- State ---
let allRecords = [];
let filteredIndices = [];
let currentFilteredPos = 0;
let originalValues = {};
let scale = 1, panX = 0, panY = 0;
let isDragging = false, dragStartX = 0, dragStartY = 0, panStartX = 0, panStartY = 0;

const img = document.getElementById('chart-img');
const viewer = document.getElementById('viewer');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('error-msg');

const fields = {
    location:  document.getElementById('f-location'),
    date:      document.getElementById('f-date'),
    end_date:  document.getElementById('f-end-date'),
    edition:   document.getElementById('f-edition'),
    interval:  document.getElementById('f-interval'),
};

// --- Init ---
async function init() {
    const resp = await fetch('/api/records');
    allRecords = await resp.json();
    populateFilters();
    applyFilter();
    loadRecord();
}

function populateFilters() {
    const locs = [...new Set(allRecords.map(r => r.location))].sort();
    const sel = document.getElementById('filter-location');
    locs.forEach(loc => {
        const opt = document.createElement('option');
        opt.value = loc; opt.textContent = loc;
        sel.appendChild(opt);
    });
}

function applyFilter() {
    const locFilter = document.getElementById('filter-location').value;
    const edFilter = document.getElementById('filter-edition').value;

    filteredIndices = [];
    allRecords.forEach((r, i) => {
        if (locFilter && r.location !== locFilter) return;
        if (edFilter === 'unknown' && r.edition !== 'Unknown') return;
        if (edFilter === 'known' && r.edition === 'Unknown') return;
        filteredIndices.push(i);
    });

    // Clamp position
    if (currentFilteredPos >= filteredIndices.length) {
        currentFilteredPos = Math.max(0, filteredIndices.length - 1);
    }
    updateCounter();
}

document.getElementById('filter-location').addEventListener('change', () => {
    currentFilteredPos = 0;
    applyFilter();
    loadRecord();
});
document.getElementById('filter-edition').addEventListener('change', () => {
    currentFilteredPos = 0;
    applyFilter();
    loadRecord();
});

// --- Record loading ---
function currentRecordIndex() {
    if (filteredIndices.length === 0) return -1;
    return filteredIndices[currentFilteredPos];
}

function loadRecord() {
    const idx = currentRecordIndex();
    if (idx < 0) {
        loading.textContent = 'No records match filter';
        loading.classList.remove('hidden');
        img.style.display = 'none';
        return;
    }

    const record = allRecords[idx];

    // Fill fields
    fields.location.value = record.location || '';
    fields.date.value = record.date || '';
    fields.end_date.value = record.end_date || '';
    fields.edition.value = record.edition || '';
    fields.interval.value = record.interval_days || '';

    // Store originals for change detection
    originalValues = {
        location: record.location || '',
        date: record.date || '',
        end_date: record.end_date || '',
        edition: record.edition || '',
        interval_days: record.interval_days || '',
    };
    clearChangedHighlights();

    // Update link
    const linkEl = document.getElementById('source-link');
    linkEl.href = record.download_link || '#';
    linkEl.textContent = record.download_link || '';

    // Check for duplicates (same location + date, different sources)
    const dupes = allRecords.filter((r, i) =>
        i !== idx && r.location === record.location && r.date === record.date
    );
    const dupBadge = document.getElementById('dup-badge');
    if (dupes.length > 0) {
        dupBadge.style.display = 'inline';
        dupBadge.textContent = `${dupes.length + 1} sources for this date`;
    } else {
        dupBadge.style.display = 'none';
    }

    updateCounter();
    loadImage(idx);
    triggerPrefetch(idx);
}

function loadImage(idx) {
    loading.textContent = 'Loading image...';
    loading.classList.remove('hidden');
    errorMsg.style.display = 'none';
    img.style.display = 'none';

    const newImg = new Image();
    newImg.onload = () => {
        img.src = newImg.src;
        img.style.display = 'block';
        loading.classList.add('hidden');
        resetZoom();
    };
    newImg.onerror = () => {
        loading.classList.add('hidden');
        errorMsg.textContent = 'Image not available - source file may not exist locally';
        errorMsg.style.display = 'block';
    };
    newImg.src = `/api/image/${idx}?t=${Date.now()}`;
}

function triggerPrefetch(idx) {
    // Tell server to prefetch upcoming images
    fetch(`/api/prefetch/${idx}`, { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filtered: filteredIndices.slice(
            currentFilteredPos + 1, currentFilteredPos + 4
        )})
    }).catch(() => {});
}

function updateCounter() {
    const counter = document.getElementById('counter');
    if (filteredIndices.length === 0) {
        counter.textContent = '0 / 0';
    } else {
        counter.textContent = `${currentFilteredPos + 1} / ${filteredIndices.length}`;
    }
    const statusText = document.getElementById('status-text');
    statusText.textContent = `${allRecords.length} total records`;
}

// --- Navigation ---
function navigate(delta) {
    if (filteredIndices.length === 0) return;
    currentFilteredPos = Math.max(0, Math.min(filteredIndices.length - 1, currentFilteredPos + delta));
    loadRecord();
}

// --- Saving ---
async function saveRecord() {
    const idx = currentRecordIndex();
    if (idx < 0) return;

    const updates = {
        location: fields.location.value,
        date: fields.date.value,
        end_date: fields.end_date.value,
        edition: fields.edition.value,
        interval_days: fields.interval.value,
    };

    try {
        const resp = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: idx, updates }),
        });
        const result = await resp.json();
        if (result.ok) {
            // Update local state
            Object.assign(allRecords[idx], updates);
            originalValues = { ...updates };
            clearChangedHighlights();
            // Brief flash to confirm save
            document.getElementById('bar').style.borderTopColor = '#4caf50';
            setTimeout(() => {
                document.getElementById('bar').style.borderTopColor = '';
            }, 300);
        }
    } catch (e) {
        console.error('Save failed:', e);
    }
}

// --- Change detection ---
Object.entries(fields).forEach(([key, el]) => {
    el.addEventListener('input', () => {
        const mapKey = key === 'interval' ? 'interval_days' : key;
        if (el.value !== originalValues[mapKey]) {
            el.classList.add('changed');
        } else {
            el.classList.remove('changed');
        }
    });
});

function clearChangedHighlights() {
    Object.values(fields).forEach(el => el.classList.remove('changed'));
}

// --- Zoom & Pan ---
function resetZoom() {
    const vw = viewer.clientWidth;
    const vh = viewer.clientHeight - 52; // account for bars
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    if (!iw || !ih) return;

    scale = Math.min(vw / iw, vh / ih, 1);
    panX = (vw - iw * scale) / 2;
    panY = (vh - ih * scale) / 2 + 36; // offset for top bar
    applyTransform();
}

function applyTransform() {
    img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}

viewer.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = viewer.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const oldScale = scale;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    scale = Math.max(0.05, Math.min(20, scale * factor));

    // Zoom toward cursor
    panX = mx - (mx - panX) * (scale / oldScale);
    panY = my - (my - panY) * (scale / oldScale);
    applyTransform();
}, { passive: false });

viewer.addEventListener('mousedown', (e) => {
    if (e.target.closest('#bar') || e.target.closest('#filters')) return;
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    panStartX = panX;
    panStartY = panY;
    viewer.classList.add('dragging');
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    panX = panStartX + (e.clientX - dragStartX);
    panY = panStartY + (e.clientY - dragStartY);
    applyTransform();
});

window.addEventListener('mouseup', () => {
    isDragging = false;
    viewer.classList.remove('dragging');
});

// Double-click to toggle between fit and 100%
viewer.addEventListener('dblclick', (e) => {
    if (e.target.closest('#bar') || e.target.closest('#filters')) return;
    const rect = viewer.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (Math.abs(scale - 1) > 0.01) {
        // Zoom to 100% centered on click point
        const oldScale = scale;
        scale = 1;
        panX = mx - (mx - panX) * (scale / oldScale);
        panY = my - (my - panY) * (scale / oldScale);
        applyTransform();
    } else {
        resetZoom();
    }
});

// --- Keyboard ---
document.addEventListener('keydown', (e) => {
    // Skip if typing in an input
    const inInput = document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT';

    if (e.key === 'Escape') {
        if (document.getElementById('help').classList.contains('visible')) {
            document.getElementById('help').classList.remove('visible');
            return;
        }
        if (inInput) {
            document.activeElement.blur();
            // Revert changes
            const idx = currentRecordIndex();
            if (idx >= 0) {
                fields.location.value = originalValues.location;
                fields.date.value = originalValues.date;
                fields.end_date.value = originalValues.end_date;
                fields.edition.value = originalValues.edition;
                fields.interval.value = originalValues.interval_days;
                clearChangedHighlights();
            }
            return;
        }
    }

    if (e.key === 'Enter' && inInput) {
        e.preventDefault();
        saveRecord();
        document.activeElement.blur();
        return;
    }

    if (inInput) return; // Don't capture other keys while editing

    if (e.key === 'ArrowLeft' || e.key === 'k') {
        e.preventDefault(); navigate(-1);
    } else if (e.key === 'ArrowRight' || e.key === 'j') {
        e.preventDefault(); navigate(1);
    } else if (e.key === 'h') {
        document.getElementById('help').classList.toggle('visible');
    } else if (e.key === 'l') {
        const linkDiv = document.getElementById('link-display');
        linkDiv.style.display = linkDiv.style.display === 'none' ? 'block' : 'none';
    } else if (e.key === 'r') {
        resetZoom();
    } else if (e.key === 'Tab') {
        e.preventDefault();
        fields.location.focus();
    }
});

// --- Start ---
init();
</script>
</body>
</html>"""


class VerifyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the verification tool."""

    def log_message(self, format, *args):
        # Suppress default logging except errors
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/api/records':
            self._serve_records()
        elif self.path.startswith('/api/image/'):
            self._serve_image()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/save':
            self._handle_save()
        elif self.path.startswith('/api/prefetch/'):
            self._handle_prefetch()
        else:
            self.send_error(404)

    def _serve_html(self):
        content = HTML_PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_records(self):
        data = json.dumps(records).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_image(self):
        try:
            idx_str = self.path.split('/api/image/')[1].split('?')[0]
            idx = int(idx_str)
        except (ValueError, IndexError):
            self.send_error(400, 'Invalid index')
            return

        if idx < 0 or idx >= len(records):
            self.send_error(404, 'Record not found')
            return

        # Check cache first
        tif_filename = records[idx].get('tif_filename', '')
        cached = cache_path_for(tif_filename)
        if not cached.exists():
            # Convert on demand
            result = convert_image(idx)
            if not result:
                self.send_error(404, 'Image not available')
                return

        if cached.exists():
            with open(cached, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, 'Image conversion failed')

    def _handle_save(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        idx = data.get('index')
        updates = data.get('updates', {})

        if idx is None or idx < 0 or idx >= len(records):
            self._json_response({'ok': False, 'error': 'Invalid index'})
            return

        with records_lock:
            for key, value in updates.items():
                if key in records[idx]:
                    records[idx][key] = value

        # Save CSV in background
        threading.Thread(target=save_csv, daemon=True).start()
        self._json_response({'ok': True})

    def _handle_prefetch(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        indices = data.get('filtered', [])
        for idx in indices[:PREFETCH_COUNT]:
            if 0 <= idx < len(records):
                cached = cache_path_for(records[idx].get('tif_filename', ''))
                if not cached.exists():
                    t = threading.Thread(target=convert_image, args=(idx,), daemon=True)
                    t.start()

        self._json_response({'ok': True})

    def _json_response(self, data):
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    print(f"Loading {CSV_FILE}...")
    load_csv()
    print(f"  Loaded {len(records)} records")

    # Check source directory
    if SOURCE_DIR.exists():
        count = sum(1 for _ in SOURCE_DIR.iterdir())
        print(f"  Source images: {SOURCE_DIR} ({count} files)")
    else:
        print(f"  WARNING: Source directory not found: {SOURCE_DIR}")
        print(f"  Images will not be available.")

    # Cache stats
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_count = sum(1 for _ in CACHE_DIR.glob('*.jpg'))
    print(f"  Cached previews: {cached_count}")

    print(f"\nStarting server on http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop\n")

    server = HTTPServer(('localhost', PORT), VerifyHandler)
    server.daemon_threads = True

    # Open browser after a short delay
    def open_browser():
        time.sleep(0.5)
        webbrowser.open(f'http://localhost:{PORT}')
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
