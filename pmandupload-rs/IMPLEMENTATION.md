# Implementation Details

## Architecture Comparison

### Bash Version (Original)
```
pmandupload.sh
├── Find TIFFs (single-threaded)
├── Sort by filename
└── For each file:
    ├── Spawn bash subprocess (bash -lc)
    ├── process_one() function
    ├── Spawn gdal_translate
    ├── Spawn gdaladdo
    ├── Spawn pmtiles
    ├── Spawn rm
    └── Spawn rclone
```

### Rust Version (New)
```
pmandupload (binary)
├── Find TIFFs (concurrent walk)
├── Collect and sort
├── rayon ThreadPool::install()
├── For each file (parallel):
    ├── process_one() function
    ├── Spawn gdal_translate (Command)
    ├── Spawn gdaladdo (Command)
    ├── Native PMTiles conversion (rusqlite + pmtiles-rs)
    │   ├── Read MBTiles SQLite database
    │   ├── Stream tiles with TMS→XYZ conversion
    │   ├── Write PMTiles with automatic deduplication
    │   └── Gzip compression
    ├── Remove MBTiles file (stdlib)
    └── Spawn rclone (Command)
```

## Key Differences

| Aspect | Bash | Rust |
|--------|------|------|
| **Parallelism Model** | Process-based (xargs) | Thread-based (rayon) |
| **Memory per Worker** | ~50MB (bash + job) | ~1MB (thread) |
| **Startup Time** | ~100ms per job | ~0ms (thread reuse) |
| **PMTiles Conversion** | Subprocess (pmtiles CLI) | Native (rusqlite + pmtiles-rs) |
| **Conversion Overhead** | ~50-100ms subprocess spawn | ~0ms (in-process) |
| **File Discovery** | Sequential walk | Can be parallel (current: sequential for ordering) |
| **Error Handling** | `set -e` stops on first error | Collects all errors, reports summary |
| **Logging** | PID-based (less relevant to parallelism) | Thread-aware (more relevant) |
| **Configuration** | Environment + command flags | CLI arguments + environment |

## Dependencies

### Not Used
- GDAL Rust bindings (we shell out to gdal_translate/gdaladdo for compatibility)
- Tokio (no async I/O needed, Command::output blocks fine)

### Core Dependencies
- `rayon`: Thread pool for parallelism
- `clap`: CLI argument parsing
- `walkdir`: Recursive directory traversal
- `tracing`: Structured logging
- `pmtiles` (0.19.1): Native PMTiles reading/writing
- `rusqlite` (0.32 with bundled feature): SQLite for MBTiles reading
- Standard library

### Why Native PMTiles?
- Eliminates subprocess spawn overhead (~50-100ms per file)
- Streaming tile processing (constant memory usage)
- Automatic tile deduplication (smaller output files)
- No external pmtiles CLI dependency
- Better error messages with file/tile context

## Settings Preservation

All original settings are maintained:

### GDAL Settings (Identical)
```
gdal_translate:
  -of MBTILES                    (same)
  -co TILE_FORMAT=WEBP           (same)
  -co QUALITY=90                 (same)
  GDAL_NUM_THREADS=ALL_CPUS     (same)

gdaladdo:
  -r bilinear                    (same)
  GDAL_NUM_THREADS=ALL_CPUS     (same)
  Zoom levels: 2,4,8,16,32,64,128,256 (same)
```

### Rclone Settings (Identical)
```
-P                             (progress)
--s3-upload-concurrency 16     (same)
--s3-chunk-size 128M           (same)
--buffer-size 128M             (same)
--s3-disable-checksum          (same)
--stats 1s                      (same)
```

### Processing Flow (Identical)
1. Find TIFFs sorted by filename
2. Skip if PMTiles already exists
3. TIFF → MBTiles (WebP 90)
4. Add overviews
5. MBTiles → PMTiles
6. Delete MBTiles
7. Upload via rclone

## Code Quality

### Error Handling
- All operations return `Result<T, anyhow::Error>`
- Failed files don't stop processing
- Exit code indicates overall success/failure
- Clear error messages with context

### Logging
- Uses `tracing` for structured logging
- Consistent log levels (info, warn, error)
- RUST_LOG environment variable for filtering

### Safety
- No unsafe code blocks
- No panics in file processing (only Result returns)
- Graceful degradation on missing commands
- Proper resource cleanup (drop semantics)

## Testing

To verify compatibility:

```bash
# Create test directory structure
mkdir -p /tmp/test_tiffs
cp sample.tif /tmp/test_tiffs/

# Run both versions (dry-run with --remote /tmp/output)
time bash pmandupload.sh  # Set REMOTE=/tmp/output before
time cargo run --release -- --root /tmp/test_tiffs --remote /tmp/output

# Compare outputs
diff <(ls /tmp/output_bash) <(ls /tmp/output_rust)
```

## Future Optimization Opportunities

1. **Parallel File Discovery**: Use rayon to walk directories in parallel
2. **Streaming Uploads**: Start uploads while processing remaining files
3. **GDAL via Library**: Use gdal-sys to avoid process spawning (requires GDAL dev libs)
4. **Memory Pooling**: Pre-allocate buffers for subprocess output
5. **Async I/O**: Use tokio for non-blocking subprocess management
6. **Progress Tracking**: Integrate progress bars (indicatif crate)
7. **Metrics**: Collect timing data per stage (TIFF→MB→PM→upload)

These are not included now to maintain simplicity and 1:1 compatibility.
