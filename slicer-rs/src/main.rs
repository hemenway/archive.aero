//! FAA Chart Newslicer - Rust Edition
//!
//! High-performance CLI tool for processing FAA sectional charts into
//! Cloud-Optimized GeoTIFFs (COGs) with parallel processing.

mod config;
mod csv_loader;
mod downloader;
mod file_index;
mod gdal_ops;
mod processor;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

/// FAA Chart Newslicer - Process sectional charts to COGs
#[derive(Parser, Debug, Clone)]
#[command(name = "slicer")]
#[command(version = "0.1.0")]
#[command(about = "FAA Chart Newslicer - Process sectional charts to COGs", long_about = None)]
pub struct Args {
    /// Source directory containing TIFF/ZIP files
    #[arg(short, long, default_value = "/Volumes/drive/newrawtiffs")]
    pub source: PathBuf,

    /// Output directory for COGs
    #[arg(short, long, default_value = "/Volumes/projects/sync")]
    pub output: PathBuf,

    /// Master Dole CSV file
    #[arg(short, long, default_value = "/Users/ryanhemenway/archive.aero/master_dole.csv")]
    pub csv: PathBuf,

    /// Directory containing shapefiles
    #[arg(short = 'b', long, default_value = "/Users/ryanhemenway/archive.aero/shapefiles")]
    pub shapefiles: PathBuf,

    /// Temporary directory for intermediate files
    #[arg(short, long, default_value = "/Volumes/projects/.temp")]
    pub temp_dir: PathBuf,

    /// Resampling algorithm for warping
    #[arg(short, long, default_value = "nearest", value_parser = ["nearest", "bilinear", "cubic", "cubicspline"])]
    pub resample: String,

    /// Intermediate warp output format
    #[arg(long, default_value = "tif", value_parser = ["tif", "vrt"])]
    pub warp_output: String,

    /// Optional projWin window (ULX ULY LRX LRY in EPSG:3857)
    #[arg(long, num_args = 4)]
    pub clip_projwin: Option<Vec<f64>>,

    /// Compression for GeoTIFF output
    #[arg(long, default_value = "AUTO", value_parser = ["AUTO", "ZSTD", "LZW", "DEFLATE", "NONE"])]
    pub compression: String,

    /// Number of threads for GeoTIFF compression
    #[arg(long, default_value = "4")]
    pub num_threads: String,

    /// Number of parallel GeoTIFF creation jobs
    #[arg(long, default_value = "3")]
    pub parallel_geotiff: usize,

    /// Seconds to wait between downloads
    #[arg(long, default_value = "8.0")]
    pub download_delay: f64,

    /// Maximum days to look back for fallback charts
    #[arg(long, default_value = "180")]
    pub fallback_window_days: i64,

    /// Skip download preparation phase
    #[arg(long, default_value = "false")]
    pub skip_downloads: bool,

    /// Skip user confirmation prompt
    #[arg(long, short = 'y', default_value = "false")]
    pub yes: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();

    info!("FAA Chart Newslicer - Rust Edition");
    info!("Source: {:?}", args.source);
    info!("Output: {:?}", args.output);
    info!("CSV: {:?}", args.csv);

    // Validate paths
    if !args.source.exists() {
        anyhow::bail!("Source directory not found: {:?}", args.source);
    }
    if !args.csv.exists() {
        anyhow::bail!("CSV file not found: {:?}", args.csv);
    }
    if !args.shapefiles.exists() {
        anyhow::bail!("Shapefiles directory not found: {:?}", args.shapefiles);
    }

    // Create output and temp directories
    std::fs::create_dir_all(&args.output)?;
    std::fs::create_dir_all(&args.temp_dir)?;

    // Run the processor
    processor::run(args)?;

    Ok(())
}
