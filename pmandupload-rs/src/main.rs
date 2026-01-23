use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pmtiles::{Compression, PmTilesWriter, TileCoord, TileType};
use rayon::prelude::*;
use rusqlite::Connection;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(name = "pmandupload")]
#[command(about = "Convert TIFFs to PMTiles and upload to R2", long_about = None)]
struct Args {
    /// Root directory containing TIFF files
    #[arg(short, long, default_value = "/Volumes/drive/upload")]
    root: PathBuf,

    /// Remote path (rclone format)
    #[arg(short, long, default_value = "r2:charts/sectionals")]
    remote: String,

    /// Number of parallel workers
    #[arg(short, long, default_value = "6")]
    jobs: usize,
}

struct RcloneFlags;

impl RcloneFlags {
    fn get() -> Vec<&'static str> {
        vec![
            "--s3-upload-concurrency",
            "16",
            "--s3-chunk-size",
            "128M",
            "--buffer-size",
            "128M",
            "--s3-disable-checksum",
            "--quiet",
        ]
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Verify required commands exist
    verify_dependencies()?;

    // Find all TIFF files, sorted chronologically
    let mut tiff_files = collect_tiff_files(&args.root)?;
    tiff_files.sort();

    let total_files = tiff_files.len();
    if total_files == 0 {
        println!("No TIFF files found in {}", args.root.display());
        return Ok(());
    }

    // Pre-count how many files need processing vs already exist
    let (to_process, already_done): (Vec<_>, Vec<_>) = tiff_files.into_iter().partition(|p| {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let pmtiles_path = p.parent().map(|d| d.join(format!("{}.pmtiles", stem)));
        pmtiles_path.map(|p| !p.exists()).unwrap_or(true)
    });

    let skipped_count = already_done.len();
    let to_process_count = to_process.len();

    println!("Found {} TIFF files ({} to process, {} already done)",
        total_files, to_process_count, skipped_count);
    println!("Using {} parallel workers\n", args.jobs);

    // Print skipped files
    for path in &already_done {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        println!("⊘ Skipped (exists): {}", stem);
    }

    if to_process_count == 0 {
        println!("\nAll files already processed!");
        return Ok(());
    }

    // Set up progress tracking
    let multi_progress = MultiProgress::new();
    let completed_count = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();

    // Main progress bar - only for files that need processing
    let main_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | ETA: {eta}")
        .unwrap()
        .progress_chars("█▓▒░  ");

    let main_pb = multi_progress.add(ProgressBar::new(to_process_count as u64));
    main_pb.set_style(main_style);
    main_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Configure rayon thread pool
    let errors: Vec<_> = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?
        .install(|| {
            to_process
                .par_iter()
                .filter_map(|tiff_path| {
                    let result = process_one_with_progress(
                        tiff_path,
                        &args.remote,
                        &multi_progress,
                        &completed_count,
                    );

                    main_pb.inc(1);

                    match result {
                        Ok(_) => None,
                        Err(e) => {
                            main_pb.println(format!("❌ Error: {} - {}", tiff_path.display(), e));
                            Some((tiff_path.clone(), e))
                        }
                    }
                })
                .collect()
        });

    main_pb.finish_and_clear();

    // Final summary
    let elapsed = start_time.elapsed();
    let completed = completed_count.load(Ordering::Relaxed);

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("                      PROCESSING COMPLETE                   ");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Total files:     {}", total_files);
    println!("  Processed:       {}", completed);
    println!("  Skipped:         {} (already existed)", skipped_count);
    println!("  Errors:          {}", errors.len());
    println!("  Total time:      {:.1}s", elapsed.as_secs_f64());
    if completed > 0 {
        println!(
            "  Avg time/file:   {:.1}s",
            elapsed.as_secs_f64() / completed as f64
        );
    }
    println!("═══════════════════════════════════════════════════════════");

    if !errors.is_empty() {
        println!("\nFailed files:");
        for (path, err) in &errors {
            eprintln!("  {} - {}", path.display(), err);
        }
        std::process::exit(1);
    }

    Ok(())
}

fn verify_dependencies() -> Result<()> {
    let commands = ["gdal_translate", "gdaladdo", "rclone"];

    for cmd in &commands {
        if !command_exists(cmd) {
            return Err(anyhow!("Missing required command: {}", cmd));
        }
    }

    Ok(())
}

fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn collect_tiff_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                (name.ends_with(".tif") || name.ends_with(".tiff"))
                    && !path.to_string_lossy().contains(".temp")
            } else {
                false
            }
        })
    {
        files.push(entry.path().to_path_buf());
    }

    Ok(files)
}

fn process_one_with_progress(
    tiff_path: &Path,
    remote: &str,
    multi_progress: &MultiProgress,
    completed_count: &Arc<AtomicU64>,
) -> Result<()> {
    let dir = tiff_path
        .parent()
        .ok_or_else(|| anyhow!("Failed to get parent directory"))?;
    let filename = tiff_path
        .file_name()
        .ok_or_else(|| anyhow!("Failed to get filename"))?
        .to_str()
        .ok_or_else(|| anyhow!("Invalid UTF-8 in filename"))?;
    let stem = Path::new(filename)
        .file_stem()
        .ok_or_else(|| anyhow!("Failed to get file stem"))?
        .to_str()
        .ok_or_else(|| anyhow!("Invalid UTF-8 in stem"))?;

    let mbtiles_path = dir.join(format!("{}.mbtiles", stem));
    let pmtiles_path = dir.join(format!("{}.pmtiles", stem));

    // Create a spinner for this file's stages (better for subprocess work)
    let spinner_style = ProgressStyle::default_spinner()
        .template("  {spinner:.cyan} {prefix:.bold.white} {msg:.dim}")
        .unwrap();

    let file_pb = multi_progress.add(ProgressBar::new_spinner());
    file_pb.set_style(spinner_style);
    file_pb.set_prefix(stem.to_string());
    file_pb.enable_steady_tick(std::time::Duration::from_millis(80));

    let start_time = Instant::now();

    // Stage 1: TIFF -> MBTiles
    file_pb.set_message("[1/4] TIFF → MBTiles (WebP)...");
    tiff_to_mbtiles(tiff_path, &mbtiles_path)?;

    // Stage 2: Add overviews
    file_pb.set_message("[2/4] Adding overviews...");
    add_overviews(&mbtiles_path)?;

    // Stage 3: MBTiles -> PMTiles
    file_pb.set_message("[3/4] MBTiles → PMTiles...");
    mbtiles_to_pmtiles_with_progress(&mbtiles_path, &pmtiles_path, &file_pb)?;

    // Remove MBTiles
    fs::remove_file(&mbtiles_path)?;

    // Stage 4: Upload
    file_pb.set_message("[4/4] Uploading to R2...");
    upload_pmtiles(&pmtiles_path, remote)?;

    let elapsed = start_time.elapsed();
    file_pb.finish_and_clear();

    completed_count.fetch_add(1, Ordering::Relaxed);
    multi_progress.println(format!("✔ {} ({:.1}s)", stem, elapsed.as_secs_f64()))?;

    Ok(())
}

fn tiff_to_mbtiles(tiff_path: &Path, mbtiles_path: &Path) -> Result<()> {
    let mut cmd = Command::new("gdal_translate");

    cmd.args([
        "-of",
        "MBTILES",
        "-co",
        "TILE_FORMAT=WEBP",
        "-co",
        "QUALITY=85",
        "-co",
        "ZOOM_LEVEL_STRATEGY=AUTO",
        "-q", // quiet mode
    ])
    .arg("--config")
    .arg("GDAL_NUM_THREADS")
    .arg("ALL_CPUS")
    .arg(tiff_path)
    .arg(mbtiles_path);

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("gdal_translate failed: {}", stderr.trim()));
    }

    Ok(())
}

fn add_overviews(mbtiles_path: &Path) -> Result<()> {
    let mut cmd = Command::new("gdaladdo");

    cmd.args(["-r", "bilinear", "-q"]) // quiet mode
        .arg("--config")
        .arg("GDAL_NUM_THREADS")
        .arg("ALL_CPUS")
        .arg(mbtiles_path)
        .args(["2", "4", "8", "16", "32", "64", "128", "256"]);

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("gdaladdo failed: {}", stderr.trim()));
    }

    Ok(())
}

fn read_mbtiles_metadata(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn
        .prepare("SELECT name, value FROM metadata")
        .context("Failed to prepare metadata query")?;

    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;

    let mut metadata = HashMap::new();
    for row in rows {
        let (name, value) = row?;
        metadata.insert(name, value);
    }
    Ok(metadata)
}

fn parse_tile_type(format: Option<&String>) -> Result<TileType> {
    match format.map(|s| s.as_str()) {
        Some("webp") | Some("WEBP") => Ok(TileType::Webp),
        Some("png") | Some("PNG") => Ok(TileType::Png),
        Some("jpg") | Some("jpeg") | Some("JPG") | Some("JPEG") => Ok(TileType::Jpeg),
        Some("pbf") | Some("mvt") => Ok(TileType::Mvt),
        Some(other) => Err(anyhow!("Unsupported tile format: {}", other)),
        None => Ok(TileType::Webp), // Default to Webp silently
    }
}

fn tms_to_xyz(row_tms: u32, zoom: u8) -> u32 {
    let max_row = (1u32 << zoom) - 1;
    max_row - row_tms
}

fn get_tile_count(conn: &Connection) -> Result<u64> {
    let mut stmt = conn
        .prepare("SELECT COUNT(*) FROM tiles")
        .context("Failed to prepare tile count query")?;

    let count: u64 = stmt.query_row([], |row| row.get(0))?;
    Ok(count)
}

fn mbtiles_to_pmtiles_with_progress(
    mbtiles_path: &Path,
    pmtiles_path: &Path,
    stage_pb: &ProgressBar,
) -> Result<()> {
    // Open MBTiles database
    let conn =
        Connection::open(mbtiles_path).context("Failed to open MBTiles database")?;

    // Read metadata
    let metadata = read_mbtiles_metadata(&conn)?;

    // Determine tile type from format
    let tile_type = parse_tile_type(metadata.get("format"))?;

    // Get total tile count for progress
    let total_tiles = get_tile_count(&conn)?;

    // Create PMTiles writer
    let file =
        File::create(pmtiles_path).context("Failed to create PMTiles file")?;

    let mut writer = PmTilesWriter::new(tile_type)
        .tile_compression(Compression::Gzip)
        .create(file)
        .context("Failed to initialize PMTiles writer")?;

    // Stream tiles from MBTiles to PMTiles
    let mut tile_count = 0u64;
    let mut stmt = conn
        .prepare(
            "SELECT zoom_level, tile_column, tile_row, tile_data
             FROM tiles
             ORDER BY zoom_level, tile_column, tile_row",
        )
        .context("Failed to prepare tiles query")?;

    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let zoom: u8 = row.get(0)?;
        let column: u32 = row.get(1)?;
        let row_tms: u32 = row.get(2)?;
        let data: Vec<u8> = row.get(3)?;

        // Convert TMS to XYZ coordinates
        let row_xyz = tms_to_xyz(row_tms, zoom);

        let coord = TileCoord::new(zoom, column, row_xyz).with_context(|| {
            format!(
                "Invalid tile coordinates: z={} x={} y={}",
                zoom, column, row_xyz
            )
        })?;

        writer.add_tile(coord, &data).with_context(|| {
            format!(
                "Failed to add tile at z={} x={} y={}",
                zoom, column, row_xyz
            )
        })?;

        tile_count += 1;

        // Update progress every 1000 tiles
        if tile_count % 1000 == 0 {
            let progress = 60 + ((tile_count as f64 / total_tiles as f64) * 25.0) as u64;
            stage_pb.set_position(progress.min(85));
            stage_pb.set_message(format!(
                "MBTiles → PMTiles ({}/{})",
                tile_count, total_tiles
            ));
        }
    }

    // Finalize the archive
    writer.finalize().context("Failed to finalize PMTiles")?;

    Ok(())
}

fn upload_pmtiles(pmtiles_path: &Path, remote: &str) -> Result<()> {
    let filename = pmtiles_path
        .file_name()
        .ok_or_else(|| anyhow!("Failed to get filename"))?
        .to_str()
        .ok_or_else(|| anyhow!("Invalid UTF-8 in filename"))?;

    let remote_path = format!("{}/{}", remote, filename);

    let mut cmd = Command::new("rclone");
    cmd.arg("copyto").arg(pmtiles_path).arg(&remote_path);

    // Add rclone flags
    for flag in RcloneFlags::get() {
        cmd.arg(flag);
    }

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("rclone upload failed: {}", stderr.trim()));
    }

    Ok(())
}
