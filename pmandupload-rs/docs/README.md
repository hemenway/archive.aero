# pmandupload-rs

High-performance Rust tool for processing MBTiles files: adds overviews, converts to PMTiles, and uploads to Cloudflare R2.

## Pipeline

```
MBTiles → gdaladdo (overviews) → PMTiles conversion → R2 upload
```

| Stage | Tool | Description |
|-------|------|-------------|
| 1. Overviews | `gdaladdo` | Adds pyramid levels (2, 4, 8, 16, 32, 64, 128, 256) for smooth zoom |
| 2. Convert | Native Rust | Streams tiles from MBTiles → PMTiles with gzip compression |
| 3. Upload | `rclone` | Uploads to R2 with optimized settings |

## Installation

### Prerequisites

1. **Rust toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Required system tools**:
   ```bash
   # Verify these are installed
   which gdaladdo rclone
   ```

3. **Configure rclone** (if not already done):
   ```bash
   rclone config
   # Set up 'r2' remote pointing to Cloudflare R2
   ```

### Build

```bash
cd pmandupload-rs
cargo build --release
```

Binary: `target/release/pmandupload`

### Install globally (optional)

```bash
cargo install --path .
```

## Usage

```bash
# Default settings
./target/release/pmandupload

# Custom options
./target/release/pmandupload \
  --root /Volumes/drive/upload \
  --remote r2:charts/sectionals \
  --jobs 4 \
  --cleanup
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-r, --root` | `/Volumes/drive/upload` | Directory containing `.mbtiles` files |
| `-R, --remote` | `r2:charts/sectionals` | Rclone remote path |
| `-j, --jobs` | `4` | Number of parallel workers |
| `--cleanup` | `false` | Delete MBTiles after successful upload |

## TUI Progress Display

The tool shows three levels of progress:

```
╔═══════════════════════════════════════════════════════════╗
║            PMTiles Converter & R2 Uploader                ║
╚═══════════════════════════════════════════════════════════╝

  Source:     /Volumes/drive/upload
  Remote:     r2:charts/sectionals
  Workers:    4
  Cleanup:    No

  Total:      50 MBTiles files
  To process: 48
  Skipped:    2 (PMTiles exists)

─── Processing ───

  ◐ Overall  [━━━━━━━━━━━━━━╺               ] 12/48 files (25%) | Elapsed: 2m | ETA: 6m
  ◐ Stage    [━━━━━━━━━━╺                   ] Overviews: 2 | Converting: 1 | Uploading: 1

  ◐ 2024-01-15   [━━━━━━━━━━━━━━━╺    ] [2/3] Converting... 45%
  ◐ 2024-02-01   [━━━━━━━╺            ] [1/3] Adding overviews...
  ✔ 2024-03-01     125.3 MB  (42.1s)
  ✔ 2024-04-01      98.7 MB  (38.2s)
```

**Progress bars:**
- **Overall** - Total files completed
- **Stage** - Distribution of work across stages
- **Per-file** - Individual file progress with current stage

## Skip Logic

Files are automatically skipped if:
- A `.pmtiles` file with the same name already exists in the same directory

This allows resuming interrupted runs.

## Performance

### Optimizations

- **Native PMTiles conversion** - No subprocess overhead (pmtiles CLI not required)
- **Streaming tile processing** - Constant memory usage regardless of file size
- **Parallel workers** - Multiple files processed simultaneously via rayon
- **GDAL multithreading** - `GDAL_NUM_THREADS=ALL_CPUS` for gdaladdo
- **Optimized rclone settings**:
  - Upload concurrency: 16
  - Chunk size: 128M
  - Buffer size: 128M
  - Checksum disabled (faster uploads)

### Benchmarks

| Metric | Typical Value |
|--------|---------------|
| Time per file | 30-60s (depends on tile count) |
| Memory usage | ~200MB per worker |
| Throughput | 4-8 files/minute with 4 workers |

## Output Format

PMTiles files are created with:
- **Tile format**: Preserved from MBTiles (typically WebP)
- **Compression**: Gzip
- **Coordinate system**: XYZ (converted from TMS)

## Error Handling

- Failed files are logged and summarized at the end
- Exit code is non-zero if any file fails
- Errors don't stop other files from processing

## Example Workflow

```bash
# 1. Generate MBTiles with newslicer.py
python newslicer.py --format mbtiles --zoom 0-11

# 2. Process and upload
cd pmandupload-rs
./target/release/pmandupload --root /output --cleanup

# 3. Verify on R2
rclone ls r2:charts/sectionals/
```

## Troubleshooting

### "Missing required command: gdaladdo"
Install GDAL:
```bash
# macOS
brew install gdal

# Ubuntu/Debian
sudo apt install gdal-bin
```

### "Missing required command: rclone"
Install and configure rclone:
```bash
# macOS
brew install rclone

# Configure R2 remote
rclone config
```

### "rclone upload failed"
Check your rclone configuration:
```bash
rclone lsd r2:
```

### Files being skipped unexpectedly
Check if `.pmtiles` files already exist:
```bash
ls /Volumes/drive/upload/*.pmtiles
```

## Dependencies

### System
- `gdaladdo` (GDAL)
- `rclone`

### Rust crates
- `pmtiles` - Native PMTiles reading/writing
- `rusqlite` - MBTiles (SQLite) database access
- `indicatif` - Progress bars
- `rayon` - Parallel processing
- `clap` - CLI argument parsing
- `anyhow` - Error handling
- `walkdir` - Directory traversal
