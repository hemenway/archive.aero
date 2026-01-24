# pmandupload-rs

High-performance Rust tool for processing MBTiles: adds overviews, converts to PMTiles, and uploads to R2.

```
MBTiles → gdaladdo → PMTiles → R2
```

## Quick Start

```bash
# Build
cargo build --release

# Run (uses defaults)
./target/release/pmandupload

# Custom options
./target/release/pmandupload \
  --root /Volumes/drive/upload \
  --remote r2:charts/sectionals \
  --jobs 4 \
  --cleanup
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-r, --root` | `/Volumes/drive/upload` | Directory with `.mbtiles` files |
| `-R, --remote` | `r2:charts/sectionals` | Rclone remote path |
| `-j, --jobs` | `4` | Parallel workers |
| `--cleanup` | `false` | Delete MBTiles after upload |

## Requirements

- `gdaladdo` (GDAL)
- `rclone` (configured with R2 remote)

**Not required**: pmtiles CLI (conversion is native Rust)

## Documentation

See [docs/README.md](docs/README.md) for full documentation including:
- Detailed pipeline explanation
- TUI progress display
- Performance optimizations
- Troubleshooting guide
