# FAA Sectional Chart Build Process Documentation

This document describes the complete build pipeline for processing FAA sectional charts from raw TIFFs to PMTiles for web distribution.

---

## Overview

The build process consists of two main stages:

1. **newslicer.py**: Processes raw chart TIFFs into georeferenced GeoTIFFs with proper mosaicking and fallback logic
2. **pmandupload.sh**: Converts GeoTIFFs to PMTiles format and uploads to cloud storage

---

## Stage 1: newslicer.py - Chart Processing

### Purpose
Processes sectional charts for multiple dates, creating a single GeoTIFF per date that mosaics all 56 sectional locations with intelligent fallback when specific dates are missing.

### Example Command

```bash
python3 newslicer.py \
  -s /Volumes/drive/newrawtiffs \
  -o /Volumes/drive/sync \
  -c /Users/ryanhemenway/archive.aero/master_dole.csv \
  -b /Users/ryanhemenway/archive.aero/shapefiles \
  -f geotiff \
  -z 0-11
```

### Command Line Arguments

- `-s, --source`: Source directory containing raw TIFF/ZIP files (default: /Volumes/drive/newrawtiffs)
- `-o, --output`: Output directory for processed GeoTIFFs (default: /Volumes/drive/sync)
- `-c, --csv`: Master CSV file mapping dates to chart files (default: master_dole.csv)
- `-b, --shapefiles`: Directory containing sectional shapefiles for cutting (default: shapefiles/)
- `-f, --format`: Output format - 'geotiff', 'tiles', or 'both' (default: geotiff)
- `-z, --zoom`: Zoom levels for tile generation (default: 0-11)

---

## Detailed Processing Steps for Example Date: 2013-02-15

Let's walk through what happens when processing the date `2013-02-15`:

### Step 1: Initialization & Scanning
```
[09:15:23] Scanning directory: /Volumes/drive/newrawtiffs...
[09:15:45] Indexed 3,247 files across 892 unique keys.
[09:15:46] Loading CSV: master_dole.csv...
[09:15:47] Loaded 156 dates.
```

**What happens:**
- Recursively scans source directory for all `.tif`, `.tiff`, and `.zip` files
- Builds index by edition number (e.g., `albuquerque_55.zip`) and timestamp (e.g., `20130215143022_houston.tif`)
- Loads CSV data mapping dates → locations → editions → filenames
- Detects GDAL compression support (ZSTD vs LZW)

### Step 2: Review Report Generation
```
[09:15:47] === Generating Processing Review ===
[09:15:52] Found 56 locations from CSV (52 with shapefiles, 4 shapefile-free).
```

**Generates matrix showing:**
```
Date        │ albuquerque │ anchorage │ atlanta │ ... │ washington
────────────┼─────────────┼───────────┼─────────┼─────┼───────────
2013-02-15  │ albuq_55.t… │ (→01-03)  │ atl_88…  │ ... │ wash_42.tif
2013-01-03  │ (→12-06)    │ anch_44…  │ (→12-06) │ ... │ (→12-06)
...
```

**Legend:**
- `filename.tif` = Direct match found for this date
- `(→01-03)` = No file for this date, will use fallback from 2013-01-03
- `---` = No file found at all
- `(N files)` = Multiple files matched

User must confirm with `y` to proceed.

### Step 3: Date Processing (2013-02-15)

```
[09:16:15] [1/156] Processing date: 2013-02-15
```

#### Step 3.1: File Discovery

For each location in the CSV (56 locations total):

```
[09:16:16]     albuquerque (edition 55): 20130215143022_albuquerque_55.zip
[09:16:16]     anchorage (edition 44): NOT FOUND (will use fallback)
[09:16:16]     atlanta (edition 88): atlanta_88.tif
[09:16:16]     hawaiian_islands (shapefile-free, edition 2): hawaiian_2.zip
```

**File resolution logic:**
1. Check if `filename` column exists in CSV for this date+location
2. If yes, resolve exact filename (handles both standalone and zip/internal format like `archive.zip/charts/houston.tif`)
3. If no filename, fall back to pattern matching:
   - Try timestamp-based: `(TS, '20130215143022')`
   - Try edition-based: `(EDITION, 'albuquerque', '55')`
   - Try variants: `albuquerque_sec`, `albuquerque_tac`, `albuquerque_north`, etc.

#### Step 3.2: Shapefile Discovery

For each location (except 4 special ones):

```
[09:16:17]       Finding shapefile for albuquerque...
[09:16:17]       Found: shapefiles/sectional/SEC_ALBUQUERQUE.shp
```

**Shapefile-free locations** (no cutting needed):
- `hawaiian_islands`
- `mariana_islands`
- `samoan_islands`
- `western_aleutian_islands`

**Search process:**
- Only searches `shapefiles/sectional/` directory (avoids TAC, Terminal, Helicopter shapefiles)
- Normalizes names (e.g., "ALBUQUERQUE" → "albuquerque")
- Matches by exact name or substring

#### Step 3.3: Parallel Warping (4 workers)

For **shapefile-based locations** (52 locations):

```
[09:16:18]   Processing 52 warp jobs with 4 parallel workers...
[09:16:20]     Warping albuquerque_55.tif...
[09:16:45]       ✓ Created albuquerque_albuquerque_55.tif (324.5 MB)
[09:16:46]     Warping atlanta_88.tif...
[09:17:12]       ✓ Created atlanta_atlanta_88.tif (412.3 MB)
... [parallel processing continues]
```

**Warp process for each file:**

1. **Extract from ZIP if needed:**
   ```bash
   # If source is albuquerque_55.zip:
   unzip albuquerque_55.zip -d .temp/2013-02-15/extract/
   ```

2. **Expand to RGBA (in-memory VRT):**
   ```python
   gdal.Translate("/vsimem/albuquerque_55_rgba.vrt", "albuquerque_55.tif",
                  format="VRT", rgbExpand="rgba")
   ```

3. **Warp to EPSG:3857 with shapefile cutline:**
   ```python
   gdal.Warp("albuquerque_albuquerque_55.tif",
             "/vsimem/albuquerque_55_rgba.vrt",
             format="GTiff",
             dstSRS='EPSG:3857',
             cutlineDSName='shapefiles/sectional/SEC_ALBUQUERQUE.shp',
             cutlineSRS='<shapefile_actual_srs>',  # Auto-detected
             cropToCutline=True,
             dstAlpha=True,
             resampleAlg=gdal.GRA_Bilinear,
             creationOptions=['TILED=YES', 'COMPRESS=ZSTD', 'BIGTIFF=YES'],
             multithread=True,
             warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'])
   ```

**For shapefile-free locations** (4 locations):

```
[09:17:15]     hawaiian_islands (shapefile-free, edition 2): hawaiian_2.zip
[09:17:16]       Extracting hawaiian_2.zip...
[09:17:17]       Extracted 1 TIFFs
[09:17:17]       Warping hawaiian_islands_sec.tif...
[09:17:42]         ✓ Warped hawaiian_islands_sec.tif (156.2 MB)
```

Same process but **no shapefile cutline** applied - just warp to EPSG:3857.

#### Step 3.4: VRT Building Per Location

For locations with multiple files:

```
[09:18:20]       Combining 2 warped files for dallas_ft_worth...
[09:18:21]       ✓ Built VRT: dallas_ft_worth_2013-02-15.vrt (2.3 KB)
```

For single files:
```
[09:18:22]       Single file, using directly
```

**Creates per-location VRTs:**
- Input: All warped TIFFs for that location
- Output: Location-specific VRT (e.g., `albuquerque_2013-02-15.vrt`)
- Only if multiple files, otherwise uses single warped TIFF directly

#### Step 3.5: Fallback Logic Application

```
[09:18:25]   Building date matrix with fallback logic...
```

**For each of 56 locations:**

```python
if date in vrt_library[location]:
    # Use exact date match
    date_matrix[location] = vrt_library[location][date]
elif vrt_library[location]:
    # Use most recent available date
    available_dates = sorted(vrt_library[location].keys())
    most_recent = available_dates[-1]  # Could be 2013-01-03
    date_matrix[location] = vrt_library[location][most_recent]
```

**Result:** 56 VRT/TIFF paths (one per location), some from current date, some from fallback.

#### Step 3.6: Mosaic Creation

```
[09:18:26]   Creating mosaic VRT from 56 locations...
[09:18:28]   ✓ Mosaic VRT created
```

**Mosaic VRT building:**
```python
gdal.BuildVRT(".temp/2013-02-15/mosaic_2013-02-15.vrt",
              [albuquerque_vrt, anchorage_vrt, atlanta_vrt, ..., washington_vrt],
              resampleAlg=gdal.GRA_Bilinear)
```

This creates a **virtual mosaic** of all 56 locations in a single coordinate system (EPSG:3857).

#### Step 3.7: Final GeoTIFF Creation

```
[09:18:29]   Creating GeoTIFF (this may take several minutes)...
[09:18:30]     Creating GeoTIFF with ZSTD compression...
[09:23:45]       ✓ GeoTIFF created: 2013-02-15.tif (8,934.2 MB)
```

**GeoTIFF conversion:**
```python
gdal.Translate("/Volumes/drive/sync/2013-02-15.tif",
               ".temp/2013-02-15/mosaic_2013-02-15.vrt",
               format='GTiff',
               creationOptions=['TILED=YES', 'COMPRESS=ZSTD',
                               'BIGTIFF=YES', 'PREDICTOR=2'])
```

**Output:** `/Volumes/drive/sync/2013-02-15.tif`
- Single tiled GeoTIFF
- EPSG:3857 Web Mercator projection
- ZSTD compression (or LZW fallback)
- BIGTIFF support for files >4GB
- Contains all 56 sectional locations mosaicked

```
[09:23:46]   ✓✓✓ 2013-02-15 COMPLETE
```

### Step 4: Process Remaining Dates

Repeat Steps 3.1-3.7 for all 156 dates in chronological order (earliest to latest):

```
[09:23:47] [2/156] Processing date: 2013-04-04
[09:28:22]   ✓✓✓ 2013-04-04 COMPLETE

[09:28:23] [3/156] Processing date: 2013-05-02
...
```

### Step 5: Completion

```
[12:45:33] === Processing Complete ===
```

**Final output directory structure:**
```
/Volumes/drive/sync/
├── 2011-10-15.tif
├── 2012-03-08.tif
├── 2012-09-05.tif
├── 2013-02-15.tif
├── 2013-04-04.tif
...
└── .temp/
    ├── 2013-02-15/
    │   ├── extract/
    │   ├── albuquerque_albuquerque_55.tif
    │   ├── atlanta_atlanta_88.tif
    │   ├── albuquerque_2013-02-15.vrt
    │   └── mosaic_2013-02-15.vrt
    └── 2013-04-04/
        └── ...
```

**Review CSV saved:** `processing_review_20240115_091552.csv`

---

## Stage 2: pmandupload.sh - PMTiles Conversion & Upload

### Purpose
Converts the final GeoTIFFs to PMTiles format and uploads them to cloud storage (Cloudflare R2).

### Example Command

```bash
./pmandupload.sh
```

Or with custom parallel workers:
```bash
JOBS=4 ./pmandupload.sh
```

### Environment Configuration

```bash
ROOT="/Volumes/drive/sync"          # Where GeoTIFFs are located
REMOTE="r2:charts/sectionals"       # rclone remote destination
JOBS="${JOBS:-3}"                   # Number of parallel workers (default: 3)
```

### Processing Steps for 2013-02-15.tif

#### Step 1: File Discovery

```bash
find "$ROOT" -type f \( -iname '*.tif' -o -iname '*.tiff' \) \
  -not -path '*/.temp/*' -print0
```

Finds all `.tif` and `.tiff` files, excluding temporary directories.

**Found:**
```
/Volumes/drive/sync/2011-10-15.tif
/Volumes/drive/sync/2012-03-08.tif
/Volumes/drive/sync/2013-02-15.tif
...
```

#### Step 2: Parallel Processing (3 workers by default)

Each file is processed by `process_one()` function in parallel:

```
==> [12847] Processing: /Volumes/drive/sync/2013-02-15.tif
```

##### Step 2.1: Convert to MBTiles

```bash
gdal_translate -of MBTILES \
  /Volumes/drive/sync/2013-02-15.tif \
  /Volumes/drive/sync/2013-02-15.mbtiles
```

**What this does:**
- Converts GeoTIFF to MBTiles format (SQLite-based tile storage)
- Generates base zoom level tiles
- Preserves georeferencing

**Output:** `/Volumes/drive/sync/2013-02-15.mbtiles`

##### Step 2.2: Generate Overviews

```bash
gdaladdo -r bilinear \
  /Volumes/drive/sync/2013-02-15.mbtiles \
  2 4 8 16 32 64 128 256
```

**What this does:**
- Adds pyramid/overview levels to MBTiles
- Uses bilinear resampling for smooth scaling
- Creates 8 overview levels (2x, 4x, 8x, ... 256x reduction)
- Enables efficient zooming at different scales

##### Step 2.3: Convert to PMTiles

```bash
pmtiles convert \
  /Volumes/drive/sync/2013-02-15.mbtiles \
  /Volumes/drive/sync/2013-02-15.pmtiles
```

**What this does:**
- Converts MBTiles (SQLite) to PMTiles (single-file cloud-optimized format)
- PMTiles supports HTTP range requests for efficient streaming
- No server-side tile serving required

**Output:** `/Volumes/drive/sync/2013-02-15.pmtiles`

##### Step 2.4: Cleanup Intermediate Files

```bash
rm -f /Volumes/drive/sync/2013-02-15.mbtiles
```

Removes the MBTiles file to save disk space (only PMTiles needed).

##### Step 2.5: Upload to Cloud Storage

```bash
rclone copyto \
  /Volumes/drive/sync/2013-02-15.pmtiles \
  r2:charts/sectionals/2013-02-15.pmtiles \
  -P \
  --s3-upload-concurrency 16 \
  --s3-chunk-size 128M \
  --buffer-size 128M \
  --s3-disable-checksum \
  --stats 1s
```

**What this does:**
- Uploads PMTiles to Cloudflare R2 (S3-compatible storage)
- Uses 16 concurrent upload streams
- 128MB chunks for optimal large file upload
- Disables checksums for speed (R2 has integrity checks)
- Shows progress every 1 second

**Final URL:** `https://data.archive.aero/sectionals/2013-02-15.pmtiles`

```
✔ [12847] Done: /Volumes/drive/sync/2013-02-15.tif
```

#### Step 3: Parallel Processing Continues

While 2013-02-15 is being uploaded, 2 other workers process other dates:

```
==> [12848] Processing: /Volumes/drive/sync/2013-04-04.tif
==> [12849] Processing: /Volumes/drive/sync/2013-05-02.tif
✔ [12848] Done: /Volumes/drive/sync/2013-04-04.tif
==> [12850] Processing: /Volumes/drive/sync/2013-06-27.tif
...
```

---

## Complete Example Workflow

### Full command sequence for processing and uploading date 2013-02-15:

```bash
# Stage 1: Process raw charts to GeoTIFF
python3 newslicer.py \
  -s /Volumes/drive/newrawtiffs \
  -o /Volumes/drive/sync \
  -c master_dole.csv \
  -b shapefiles \
  -f geotiff

# Output: /Volumes/drive/sync/2013-02-15.tif (8.9 GB)

# Stage 2: Convert to PMTiles and upload
./pmandupload.sh

# Output: https://data.archive.aero/sectionals/2013-02-15.pmtiles (3.2 GB)
```

---

## Key Technologies & Formats

### GDAL Operations
- **gdal.Translate**: Format conversion, VRT creation, RGBA expansion
- **gdal.Warp**: Reprojection, shapefile cutting, resampling
- **gdal.BuildVRT**: Virtual mosaic creation
- **gdal_translate (CLI)**: GeoTIFF to MBTiles conversion
- **gdaladdo**: Overview/pyramid generation

### Coordinate Systems
- **Input**: Various (typically EPSG:4326 WGS84)
- **Intermediate**: Shapefile native SRS (auto-detected)
- **Output**: EPSG:3857 Web Mercator (universal for web maps)

### Compression
- **Primary**: ZSTD (modern, high compression ratio)
- **Fallback**: LZW (older, wider compatibility)
- **Tile format**: WEBP (for gdal2tiles, not used in current geotiff-only mode)

### File Formats
- **Input**: TIFF, ZIP (containing TIFFs)
- **Intermediate**: VRT (virtual raster), temporary GeoTIFFs
- **Stage 1 Output**: GeoTIFF (tiled, compressed, georeferenced)
- **Stage 2 Intermediate**: MBTiles (SQLite-based tiles)
- **Stage 2 Output**: PMTiles (cloud-optimized single-file tiles)

---

## Performance Characteristics

### newslicer.py
- **Parallel warping**: 4 workers (ThreadPoolExecutor)
- **Memory usage**: 75% of available RAM for GDAL cache (max 8GB)
- **Disk I/O**: Heavy (multiple GB per location)
- **Typical runtime**: 3-6 hours for 156 dates (depends on source data)

### pmandupload.sh
- **Parallel processing**: 3 workers by default (tunable with JOBS env var)
- **Upload concurrency**: 16 concurrent S3 streams per file
- **Chunk size**: 128MB
- **Typical runtime**: 2-4 hours for 156 files (depends on network speed)

---

## Output Verification

### GeoTIFF Verification
```bash
gdalinfo /Volumes/drive/sync/2013-02-15.tif
```

Expected output:
- Driver: GTiff/GeoTIFF
- Size: ~30000 x ~20000 pixels (varies)
- Coordinate System: EPSG:3857
- Bands: 4 (R, G, B, Alpha)
- Block Size: 256x256 (tiled)
- Compression: ZSTD or LZW

### PMTiles Verification
```bash
pmtiles show /Volumes/drive/sync/2013-02-15.pmtiles
```

Or check deployed URL:
```bash
curl -I https://data.archive.aero/sectionals/2013-02-15.pmtiles
```

---

## Dependencies

### Required Software
- **Python 3.7+** with GDAL bindings
- **GDAL 3.0+** (gdal-bin package)
- **PMTiles CLI** (https://github.com/protomaps/go-pmtiles)
- **rclone** (for R2 uploads)
- **psutil** (Python package, optional but recommended)

### Installation
```bash
# macOS
brew install gdal python3 rclone

# Install Python packages
pip3 install gdal psutil

# Install PMTiles
go install github.com/protomaps/go-pmtiles/cmd/pmtiles@latest
```

---

## Error Handling

### newslicer.py
- **Missing source files**: Logs warning, uses fallback from previous date
- **Missing shapefile**: Skips location, logs error
- **ZSTD not available**: Automatically falls back to LZW compression
- **ZSTD decompression error**: Retries with LZW compression
- **Corrupt/small files**: Validates file size, skips files <1KB

### pmandupload.sh
- **Missing command**: Exits with error code 127
- **Upload failure**: rclone retries automatically
- Uses `set -euo pipefail` for strict error handling

---

## Fallback Logic Example

Given these dates processed in order:
1. 2013-01-03: Houston edition 42 available
2. 2013-02-15: Houston edition 43 missing
3. 2013-04-04: Houston edition 44 available

**Result for 2013-02-15:**
- Houston location uses edition 42 VRT from 2013-01-03
- Review report shows: `(→01-03)` for Houston on 2013-02-15
- Ensures no gaps in the final mosaic

**VRT Library state after processing 2013-02-15:**
```python
vrt_library['houston']['2013-01-03'] = Path('houston_2013-01-03.vrt')
vrt_library['houston']['2013-04-04'] = Path('houston_2013-04-04.vrt')
# 2013-02-15 reuses 2013-01-03 in date_matrix
```

---

## Directory Structure Summary

```
archive.aero/
├── newslicer.py                          # Stage 1: Chart processor
├── pmandupload.sh                        # Stage 2: PMTiles converter/uploader
├── master_dole.csv                       # Date → Location → Edition mapping
├── shapefiles/
│   └── sectional/                        # Sectional boundaries for cutting
│       ├── SEC_ALBUQUERQUE.shp
│       ├── SEC_ATLANTA.shp
│       └── ...

/Volumes/drive/
├── newrawtiffs/                          # Input: Raw chart TIFFs/ZIPs
│   ├── albuquerque_55.zip
│   ├── 20130215143022_houston.tif
│   └── ...
└── sync/                                 # Output: Processed GeoTIFFs & PMTiles
    ├── 2013-02-15.tif                   # GeoTIFF (after Stage 1)
    ├── 2013-02-15.pmtiles               # PMTiles (after Stage 2)
    ├── processing_review_*.csv          # Review reports
    └── .temp/                            # Temporary processing files
        └── 2013-02-15/
            ├── extract/                  # Extracted ZIPs
            ├── *_warped.tif             # Per-location warped files
            ├── *_2013-02-15.vrt         # Per-location VRTs
            └── mosaic_2013-02-15.vrt    # Final mosaic VRT
```

---

## dates.csv Integration

The `dates.csv` file tracks available processed dates:

```csv
date_iso,url
2013-02-15,https://data.archive.aero/sectionals/2013-02-15.pmtiles
```

This file is used by the frontend to populate the date picker and load the correct PMTiles for each historical date.
