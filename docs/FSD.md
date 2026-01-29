# File Specification Document (FSD)
## Archive.aero - Historical Aeronautical Chart Viewer System

**Version:** 3.0.0
**Last Updated:** 2026-01-28
**Maintainer:** Ryan Hemenway

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Frontend Specification (index.html)](#3-frontend-specification-indexhtml)
4. [Backend Specification (newslicer.py)](#4-backend-specification-newslicerpy)
5. [PMTiles Conversion Pipeline](#5-pmtiles-conversion-pipeline)
6. [Data Specifications](#6-data-specifications)
7. [Deployment](#7-deployment)

---

## 1. System Overview

### 1.1 Purpose

Archive.aero provides interactive visualization of historical FAA aeronautical sectional charts spanning from 2011 to present. The system functions as a "ForeFlight Time Machine" allowing users to see how aviation charts and airspace have evolved over time.

### 1.2 Components

| Component | File | Purpose |
|-----------|------|---------|
| **Frontend Viewer** | `index.html` | Web-based interactive map with timeline |
| **Backend Processor** | `newslicer.py` | CLI tool for chart processing and output generation |
| **PMTiles Converter** | `pmandupload.sh` / `pmandupload-rs` | MBTiles to PMTiles conversion and R2 upload |

### 1.3 Technology Stack

**Frontend:**
- HTML5/CSS3/JavaScript (ES6+)
- Leaflet.js 1.9.4 (mapping library)
- PMTiles library (tile fetching from single-file archives)
- PapaParse 5.4.1 (CSV parsing)

**Backend:**
- Python 3.7+
- GDAL/OGR (geospatial processing)
- gdal2tiles.py (tile generation)
- Requests (HTTP downloads)
- psutil (optional, for RAM detection)

### 1.4 Data Flow

```
Raw Chart TIFFs/ZIPs (from Internet Archive)
         │
         ▼
    newslicer.py ──────────────────────────────┐
         │                                      │
         ├─► VRT (virtual raster mosaic)       │
         │                                      │
         ├─► GeoTIFF (compressed, optional)    │
         │                                      │
         └─► MBTiles (tiled, optional)         │
                      │                         │
                      ▼                         │
             pmandupload.sh/rs                  │
                      │                         │
                      ▼                         │
                 PMTiles ◄──────────────────────┘
                      │
                      ▼
            Cloudflare R2 (CDN)
                      │
                      ▼
              index.html viewer
```

---

## 2. Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       User's Web Browser                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  index.html (Frontend)                     │  │
│  │  - Leaflet Map with PMTiles tile source                   │  │
│  │  - Timeline Controller with playback                      │  │
│  │  - Location search (Nominatim)                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CDN / Static File Server                        │
│                                                                  │
│  /dates.csv                  - Timeline metadata                 │
│  /sectionals/pmtiles/        - PMTiles directory                │
│    └── YYYY-MM-DD.pmtiles    - Single-file tile archives        │
│                                                                  │
│  Origin: https://data.archive.aero/sectionals/pmtiles/          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                        Uploaded via rclone
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                           │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ newslicer.py │──►│  MBTiles     │──►│ pmandupload  │        │
│  │              │   │  (per date)  │   │  (convert +  │        │
│  │  - Warp      │   │              │   │   upload)    │        │
│  │  - Mosaic    │   └──────────────┘   └──────────────┘        │
│  │  - Output    │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
archive.aero/
├── index.html                    # Frontend viewer (single-page app)
├── dates.csv                     # Timeline metadata for frontend
├── master_dole.csv               # Chart source metadata (locations/editions/URLs)
├── FSD.md                        # This document
├── newslicer.py                  # Backend processor CLI
├── pmandupload.sh                # PMTiles conversion shell script
├── pmandupload-rs/               # Rust PMTiles converter (faster)
│   └── src/main.rs
├── shapefiles/                   # Chart boundary definitions
│   └── sectional/                # 56 FAA sectional boundaries
│       ├── SEC_ALBUQUERQUE.shp
│       ├── SEC_ANCHORAGE.shp
│       └── ...
├── docs/
│   └── map.txt                   # newslicer.py logic flow diagram
└── output/                       # Generated outputs (example)
    ├── .temp/                    # Temporary processing directory
    │   └── YYYY-MM-DD/
    │       ├── location_file.tif # Warped individual charts
    │       └── mosaic_YYYY-MM-DD.vrt
    ├── YYYY-MM-DD.tif            # GeoTIFF output (if format=geotiff)
    └── YYYY-MM-DD.mbtiles        # MBTiles output (if format=mbtiles)
```

---

## 3. Frontend Specification (index.html)

### 3.1 Overview

Single-page application providing an interactive timeline-based map viewer for historical aeronautical charts using PMTiles format.

### 3.2 Global Configuration

```javascript
const CONFIG = {
  baseUrl: 'https://data.archive.aero/sectionals/pmtiles/',
  csvUrl: 'dates.csv',
  initialView: { center: [32.7767, -96.7970], zoom: 10 },
  frames: []  // Populated dynamically from CSV
};
```

### 3.3 Core Classes

#### 3.3.1 MapController

**Purpose:** Manages Leaflet map instance and PMTiles-backed tile layers.

**Constructor:**
```javascript
constructor(mapId: String)
```

**Properties:**
- `map` - Leaflet map instance
- `cache` - Object storing layer data by date
- `activeLayers` - Array of currently displayed layers
- `pmtilesInstances` - Cache for PMTiles instances
- `requestCounter` - Tracks most recent frame request (prevents stale updates)

**Key Methods:**

| Method | Description |
|--------|-------------|
| `getLayerData(date, customUrl)` | Returns cached layer or creates new PMTiles-backed tile layer |
| `showFrame(index)` | Displays chart for specified frame index with smooth transitions |
| `waitForLayer(layer)` | Returns Promise that resolves when layer tiles are loaded |
| `setOpacity(value)` | Sets opacity for all active layers |

**PMTiles Tile Layer:**
```javascript
// Custom createTile method fetches tiles from PMTiles archive
layer.createTile = function(coords, done) {
  const tile = document.createElement('img');
  p.getZxy(coords.z, coords.x, coords.y).then(result => {
    if (!layer._map) return; // Guard against stale callbacks
    if (result && result.data) {
      const blob = new Blob([result.data], { type: 'image/webp' });
      tile.src = URL.createObjectURL(blob);
      // ... handle load/error
    }
  });
  return tile;
};
```

#### 3.3.2 TimelineApp

**Purpose:** Controls timeline UI and user interactions.

**Properties:**
- `mapCtrl` - Reference to MapController
- `frames` - Array of timeline frames with date and percentage position
- `currentIndex` - Current frame index
- `isPlaying` - Boolean playback state
- `playbackSpeed` - Playback speed multiplier

**Key Methods:**

| Method | Description |
|--------|-------------|
| `update(index, lazy)` | Updates timeline to specified index |
| `step(direction)` | Steps timeline forward (+1) or backward (-1) |
| `togglePlay()` | Toggles playback mode (2000ms per frame) |
| `updateShareUrl()` | Generates shareable URL with current state |
| `hideAllPanels(except)` | Closes all overlay panels except specified |

### 3.4 Data Loading

**CSV Format (dates.csv):**
```csv
date_iso
2011-10-15
2012-09-08
2013-02-15
...
```

**Loading Process:**
```javascript
Papa.parse(CONFIG.csvUrl, {
  download: true,
  header: true,
  skipEmptyLines: true,
  complete: (results) => {
    CONFIG.frames = results.data
      .filter(row => row.date_iso && row.date_iso !== '?')
      .map(row => ({
        id: Utils.formatDateId(row.date_iso),
        date: row.date_iso
      }))
      .sort((a, b) => new Date(a.date) - new Date(b.date));
  }
});
```

### 3.5 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next frame |
| `←` | Previous frame |
| `Space` | Play/Pause |
| `F` | Toggle fullscreen |
| `S` | Open share panel |
| `?` | Show keyboard shortcuts |
| `Esc` | Close overlays |

### 3.6 URL Parameters

**Shareable Link Format:**
```
https://archive.aero/?date=2015-06-25&lat=32.7767&lng=-96.7970&zoom=10
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `date` | String (ISO) | Initial date (YYYY-MM-DD) |
| `lat` | Float | Initial latitude |
| `lng` | Float | Initial longitude |
| `zoom` | Integer | Initial zoom level (0-18) |

---

## 4. Backend Specification (newslicer.py)

### 4.1 Overview

Python CLI tool for processing FAA aeronautical charts into GeoTIFFs or MBTiles with sophisticated fallback logic ensuring all 56 chart locations are represented in each date's output.

### 4.2 System Requirements

**Required:**
- Python 3.7+
- GDAL 3.0+ with Python bindings (`osgeo`)
- gdal2tiles.py (included with GDAL)
- requests library

**Optional:**
- psutil (for dynamic RAM cache sizing)
- pmtiles CLI tool (for PMTiles conversion)

### 4.3 CLI Arguments

```bash
python newslicer.py [OPTIONS]
```

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--source` | `-s` | Path | `/Volumes/drive/newrawtiffs` | Source directory containing TIFF/ZIP files |
| `--output` | `-o` | Path | `/Volumes/projects/sync` | Output directory for processed files |
| `--csv` | `-c` | Path | `~/archive.aero/master_dole.csv` | Master CSV with chart metadata |
| `--shapefiles` | `-b` | Path | `~/archive.aero/shapefiles` | Directory containing shapefiles |
| `--temp-dir` | `-t` | Path | `/Volumes/projects/.temp` | Temporary directory for intermediates |
| `--format` | `-f` | String | `geotiff` | Output: `geotiff`, `mbtiles`, `tiles`, or `both` |
| `--zoom` | `-z` | String | `0-11` | Zoom levels for tiles (e.g., `0-11`) |
| `--resample` | `-r` | String | `nearest` | Resampling: `nearest`, `bilinear`, `cubic`, `cubicspline` |
| `--warp-output` | | String | `tif` | Intermediate format: `tif` or `vrt` |
| `--clip-projwin` | | Float[4] | None | Optional projWin clip (ULX ULY LRX LRY) |
| `--compression` | | String | `AUTO` | GeoTIFF compression: `AUTO`, `ZSTD`, `LZW`, `DEFLATE`, `NONE` |
| `--num-threads` | | String | `4` | Threads for GeoTIFF compression |
| `--preview` | | Flag | false | Dry-run mode (no downloads) |
| `--download-delay` | | Float | `8.0` | Seconds between downloads |
| `--parallel-mbtiles` | | Integer | `2` | Parallel MBTiles creation jobs |
| `--mbtiles-only` | | Flag | false | Resume mode: only process MBTiles from existing VRTs |
| `--mbtiles-output` | | Path | None | Separate output directory for MBTiles |
| `--tile-format` | | String | `PNG` | MBTiles tile format: `PNG`, `JPEG`, `WEBP` |

### 4.4 Core Class: ChartSlicer

#### 4.4.1 Constructor

```python
def __init__(self, source_dir: Path, output_dir: Path, csv_file: Path,
             shape_dir: Path, preview_mode: bool = False,
             download_delay: float = 8.0, temp_dir: Path = None)
```

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `source_dir` | Path | Directory containing raw TIFFs/ZIPs |
| `output_dir` | Path | Output directory for processed data |
| `temp_dir` | Path | Temporary directory for intermediates |
| `dole_data` | Dict | Chart metadata grouped by date |
| `files_by_name` | Dict | O(1) index: filename → Path |
| `tifs_by_dir` | Dict | O(1) index: directory → list of TIF paths |
| `shapefile_index` | Dict | O(1) index: normalized location → shapefile path |
| `compression` | String | ZSTD or LZW (auto-detected) |
| `output_format` | String | geotiff, mbtiles, tiles, or both |

#### 4.4.2 Key Methods

**Data Loading:**

| Method | Description |
|--------|-------------|
| `load_dole_data()` | Load CSV and group records by date |
| `_build_source_index()` | Build O(1) file lookup index |
| `_build_shapefile_index()` | Build O(1) shapefile lookup index |

**File Resolution:**

| Method | Description |
|--------|-------------|
| `resolve_filename(filename, download_link)` | Resolve CSV filename to file path(s) |
| `find_files(location, edition, timestamp, filename, download_link)` | Find source files for a location |
| `find_shapefile(location)` | Find shapefile for location (O(1) lookup) |

**Download Management:**

| Method | Description |
|--------|-------------|
| `download_file(url, target_path)` | Download with throttling and retry logic |
| `extract_zip(zip_path, extract_dir)` | Extract ZIP and update indexes |
| `_download_missing_files()` | Preparation phase: download all missing files upfront |

**GDAL Operations:**

| Method | Description |
|--------|-------------|
| `warp_and_cut(input_tiff, shapefile, output_tiff)` | Warp to EPSG:3857 and crop with cutline |
| `build_vrt(input_files, output_vrt)` | Combine multiple files into VRT mosaic |
| `create_geotiff(input_vrt, output_tiff, compress)` | Convert VRT to compressed GeoTIFF |
| `create_mbtiles(input_vrt, output_mbtiles, tile_format)` | Convert VRT to MBTiles |
| `generate_tiles(input_vrt, output_dir, zoom_levels)` | Generate WEBP tiles via gdal2tiles.py |

**Processing:**

| Method | Description |
|--------|-------------|
| `process_all_dates()` | Main processing loop with fallback logic |
| `_process_mbtiles_only()` | Resume mode: process MBTiles from existing VRTs |
| `generate_review_report()` | Generate pre-processing review matrix |

### 4.5 GDAL Configuration

Phase-specific GDAL configuration prevents oversubscription:

```python
def _configure_gdal_for_phase(phase: str, workers: int = 1):
    if phase == 'warp':
        # Single-threaded per worker to prevent oversubscription
        gdal.SetConfigOption('GDAL_NUM_THREADS', '1')

    elif phase == 'mbtiles':
        # Single worker gets all CPUs, multiple workers throttled
        if workers > 1:
            gdal.SetConfigOption('GDAL_NUM_THREADS', '1')
        else:
            gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    elif phase == 'geotiff':
        # Single-process translate with full CPU
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
```

**Global Settings:**
```python
gdal.SetConfigOption('GDAL_CACHEMAX', str(cache_size_mb))  # 75% RAM, max 8GB
gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
```

### 4.6 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INITIALIZATION PHASE                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. load_dole_data()        - Load CSV, group by date            │
│ 2. _build_source_index()   - Index all TIFFs/ZIPs for O(1)      │
│ 3. _build_shapefile_index()- Index shapefiles for O(1)          │
│ 4. generate_review_report()- Show preview matrix, get confirm   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPARATION PHASE                             │
├─────────────────────────────────────────────────────────────────┤
│ _download_missing_files()                                        │
│  - Collect all missing files from CSV                           │
│  - Download sequentially with throttling (--download-delay)     │
│  - Extract ZIPs and update indexes                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PER-DATE PROCESSING (earliest → latest)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    WARP PHASE                            │    │
│  │  (Parallel: ThreadPoolExecutor, CPU_COUNT - 2 workers)  │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  For each location in CSV record:                        │    │
│  │   1. find_shapefile(location) - O(1) lookup             │    │
│  │   2. find_files(...) - Resolve filename to path(s)       │    │
│  │   3. _prepare_warp_job() - Create job parameters         │    │
│  │   4. _warp_worker() - Execute warp_and_cut():            │    │
│  │      • gdal.Translate → /vsimem/ RGBA VRT               │    │
│  │      • gdal.Warp → EPSG:3857 with shapefile cutline     │    │
│  │      • Output: location_filename.tif (or .vrt)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  VRT BUILD PHASE                         │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  For each location with multiple warped files:           │    │
│  │   • build_vrt() → location_date.vrt                     │    │
│  │   • Store in vrt_library[location][date]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              FALLBACK MATRIX BUILD                       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  For each of 56 locations:                               │    │
│  │   • Find most recent available date ≤ current_date      │    │
│  │   • Add to date_matrix[location] = vrt_path             │    │
│  │                                                          │    │
│  │  Result: Complete coverage using fallback where needed   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 MOSAIC VRT BUILD                         │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  build_vrt(date_matrix.values(), mosaic_date.vrt)       │    │
│  │   • Combines all location VRTs into single mosaic       │    │
│  │   • Uses bilinear resampling, highest resolution        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  OUTPUT GENERATION                       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  If format='geotiff' or 'both':                          │    │
│  │   • create_geotiff() → date.tif (ZSTD/LZW compressed)   │    │
│  │                                                          │    │
│  │  If format='mbtiles':                                    │    │
│  │   • Queue for parallel MBTiles processing               │    │
│  │                                                          │    │
│  │  If format='tiles' or 'both':                            │    │
│  │   • generate_tiles() → date_tiles/z/x/y.webp            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MBTILES BATCH PROCESSING (if format='mbtiles')      │
├─────────────────────────────────────────────────────────────────┤
│  ProcessPoolExecutor with --parallel-mbtiles workers            │
│   • create_mbtiles_worker() for each queued VRT                 │
│   • Progress callback with retry logic                          │
│   • Memory-aware cache allocation per worker                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 Usage Examples

```bash
# Standard processing: GeoTIFF output with ZSTD compression
python newslicer.py -s /path/to/charts -o /path/to/output -f geotiff

# MBTiles output with WEBP tiles and custom zoom
python newslicer.py -f mbtiles --tile-format WEBP -z 0-13

# Preview mode (dry-run) to check downloads
python newslicer.py --preview --download-delay 12.0

# Resume MBTiles from existing VRTs
python newslicer.py --mbtiles-only --parallel-mbtiles 4

# VRT intermediates (reduces disk I/O)
python newslicer.py --warp-output vrt -f mbtiles

# Full processing with custom paths
python newslicer.py \
  -s /Volumes/drive/newrawtiffs \
  -o /Volumes/projects/sync \
  -t /Volumes/projects/.temp \
  -c master_dole.csv \
  -b shapefiles \
  -f mbtiles \
  --parallel-mbtiles 4
```

---

## 5. PMTiles Conversion Pipeline

### 5.1 Overview

After newslicer.py generates MBTiles or GeoTIFF outputs, they are converted to PMTiles format for efficient serverless web hosting.

### 5.2 Tools

| Tool | Description |
|------|-------------|
| `pmandupload.sh` | Interactive TUI shell script |
| `pmandupload-rs` | High-performance Rust implementation |

### 5.3 Conversion Steps

```bash
# Step 1: Add pyramid overviews to MBTiles
gdaladdo -r bilinear input.mbtiles 2 4 8 16 32 64 128 256

# Step 2: Convert MBTiles to PMTiles
pmtiles convert input.mbtiles output.pmtiles

# Step 3: Upload to R2/S3
rclone copy output.pmtiles r2:charts/sectionals/pmtiles/
```

### 5.4 PMTiles Benefits

- **Single file** instead of thousands of individual tiles
- **HTTP range requests** for efficient partial downloads
- **Serverless hosting** on static storage (R2, S3, GCS)
- **No tile server required** - browser fetches directly

---

## 6. Data Specifications

### 6.1 Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `master_dole.csv` | Chart source metadata | CSV |
| `dates.csv` | Timeline dates for frontend | CSV |
| `shapefiles/sectional/*.shp` | 56 sectional boundaries | ESRI Shapefile |

### 6.2 master_dole.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | String | Chart date (YYYY-MM-DD) |
| `location` | String | Chart location name |
| `edition` | String | Edition number |
| `filename` | String | Source filename (.tif or .zip) |
| `download_link` | String | Internet Archive Wayback URL |

### 6.3 Output File Sizes (Typical)

| Format | Size per Date | Notes |
|--------|---------------|-------|
| Warped TIFFs | ~200MB each | ~50 per date |
| Mosaic VRT | ~20KB | XML metadata only |
| GeoTIFF | 500MB - 1.5GB | LZW/ZSTD compressed |
| MBTiles | 100-300MB | PNG tiles at zoom 0-11 |
| PMTiles | 100-300MB | After conversion |

---

## 7. Deployment

### 7.1 Frontend Deployment

The frontend is a single HTML file with no build step required:

1. Upload `index.html` to static hosting
2. Upload `dates.csv` to same location
3. Ensure PMTiles are accessible at configured `baseUrl`

### 7.2 Processing Workflow

1. **Prepare sources:** Ensure raw TIFFs/ZIPs are in source directory
2. **Run newslicer:** `python newslicer.py -f mbtiles --parallel-mbtiles 4`
3. **Convert to PMTiles:** `./pmandupload.sh` or `./pmandupload-rs`
4. **Upload:** PMTiles uploaded to R2/S3 via rclone

### 7.3 Browser Compatibility

**Minimum Requirements:**
- ES6 support (async/await, arrow functions)
- Fetch API
- Clipboard API (with fallback)

**Tested Browsers:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

*End of File Specification Document*
