use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pmtiles::{Compression, PmTilesWriter, TileCoord, TileType};
use rusqlite::Connection;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "pmandupload")]
#[command(about = "Add overviews to MBTiles, convert to PMTiles, and upload to R2", long_about = None)]
struct Args {
    /// Root directory containing MBTiles files
    #[arg(short, long, default_value = "/Volumes/drive/upload")]
    root: PathBuf,

    /// Remote path (rclone format)
    #[arg(short, long, default_value = "r2:charts/sectionals")]
    remote: String,

    /// Number of parallel workers
    #[arg(short, long, default_value = "4")]
    jobs: usize,

    /// Delete MBTiles after successful upload
    #[arg(long, default_value = "false")]
    cleanup: bool,
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

    // Find all MBTiles files, sorted chronologically
    let mut mbtiles_files = collect_mbtiles_files(&args.root)?;
    mbtiles_files.sort();

    let total_files = mbtiles_files.len();
    if total_files == 0 {
        println!("No MBTiles files found in {}", args.root.display());
        return Ok(());
    }

    // Pre-count how many files need processing vs already exist
    let (to_process, already_done): (Vec<_>, Vec<_>) = mbtiles_files.into_iter().partition(|p| {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let pmtiles_path = p.parent().map(|d| d.join(format!("{}.pmtiles", stem)));
        pmtiles_path.map(|p| !p.exists()).unwrap_or(true)
    });

    let skipped_count = already_done.len();
    let to_process_count = to_process.len();

    // Print header
    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║            PMTiles Converter & R2 Uploader                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Source:     {}", args.root.display());
    println!("  Remote:     {}", args.remote);
    println!("  Workers:    {}", args.jobs);
    println!("  Cleanup:    {}", if args.cleanup { "Yes" } else { "No" });
    println!();
    println!("  Total:      {} MBTiles files", total_files);
    println!("  To process: {}", to_process_count);
    println!("  Skipped:    {} (PMTiles exists)", skipped_count);
    println!();

    // Print skipped files
    if !already_done.is_empty() {
        println!("─── Skipped (already converted) ───");
        for path in &already_done {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            println!("  ⊘ {}", stem);
        }
        println!();
    }

    if to_process_count == 0 {
        println!("All files already processed!");
        return Ok(());
    }

    println!("─── Processing ───");
    println!();

    // Set up progress tracking
    let multi_progress = MultiProgress::new();
    let completed_count = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();

    // Overall progress bar (at the top)
    let overall_style = ProgressStyle::default_bar()
        .template("  {spinner:.green} Overall  [{bar:30.green/dim}] {pos}/{len} files ({percent}%) | Elapsed: {elapsed} | ETA: {eta}")
        .unwrap()
        .progress_chars("━━╺");

    let overall_pb = multi_progress.add(ProgressBar::new(to_process_count as u64));
    overall_pb.set_style(overall_style);
    overall_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Stage progress bar (shows current stage distribution)
    let stage_style = ProgressStyle::default_bar()
        .template("  {spinner:.cyan} Stage    [{bar:30.cyan/dim}] {msg}")
        .unwrap()
        .progress_chars("━━╺");

    let stage_pb = multi_progress.add(ProgressBar::new(100));
    stage_pb.set_style(stage_style);
    stage_pb.set_message("Starting...");
    stage_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Spacer
    multi_progress.println("")?;

    // Track stage counts for stage progress bar
    let stage_counts = Arc::new([
        AtomicU64::new(0), // Overviews
        AtomicU64::new(0), // Convert
        AtomicU64::new(0), // Upload
    ]);

    // Configure rayon thread pool
    let errors: Vec<_> = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?
        .install(|| {
            to_process
                .par_iter()
                .filter_map(|mbtiles_path| {
                    let result = process_one_with_progress(
                        mbtiles_path,
                        &args.remote,
                        args.cleanup,
                        &multi_progress,
                        &completed_count,
                        &stage_counts,
                        &stage_pb,
                    );

                    overall_pb.inc(1);

                    match result {
                        Ok(_) => None,
                        Err(e) => {
                            multi_progress.println(format!("  ❌ {} - {}",
                                mbtiles_path.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
                                e
                            )).ok();
                            Some((mbtiles_path.clone(), e))
                        }
                    }
                })
                .collect()
        });

    overall_pb.finish_and_clear();
    stage_pb.finish_and_clear();

    // Final summary
    let elapsed = start_time.elapsed();
    let completed = completed_count.load(Ordering::Relaxed);

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                    PROCESSING COMPLETE                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
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
    println!();

    if !errors.is_empty() {
        println!("─── Failed Files ───");
        for (path, err) in &errors {
            eprintln!("  {} - {}", path.display(), err);
        }
        std::process::exit(1);
    }

    Ok(())
}

fn verify_dependencies() -> Result<()> {
    let commands = ["gdaladdo", "rclone"];

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

fn collect_mbtiles_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                name.ends_with(".mbtiles") && !path.to_string_lossy().contains(".temp")
            } else {
                false
            }
        })
    {
        files.push(entry.path().to_path_buf());
    }

    Ok(files)
}

fn update_stage_progress(
    stage_counts: &[AtomicU64; 3],
    stage_pb: &ProgressBar,
    total: u64,
) {
    let overviews = stage_counts[0].load(Ordering::Relaxed);
    let convert = stage_counts[1].load(Ordering::Relaxed);
    let upload = stage_counts[2].load(Ordering::Relaxed);

    let msg = format!(
        "Overviews: {} | Converting: {} | Uploading: {}",
        overviews, convert, upload
    );
    stage_pb.set_message(msg);

    // Calculate overall stage progress (weighted)
    let progress = if total > 0 {
        ((overviews * 33 + convert * 33 + upload * 34) / total).min(100)
    } else {
        0
    };
    stage_pb.set_position(progress);
}

fn process_one_with_progress(
    mbtiles_path: &Path,
    remote: &str,
    cleanup: bool,
    multi_progress: &MultiProgress,
    completed_count: &Arc<AtomicU64>,
    stage_counts: &Arc<[AtomicU64; 3]>,
    stage_pb: &ProgressBar,
) -> Result<()> {
    let dir = mbtiles_path
        .parent()
        .ok_or_else(|| anyhow!("Failed to get parent directory"))?;
    let filename = mbtiles_path
        .file_name()
        .ok_or_else(|| anyhow!("Failed to get filename"))?
        .to_str()
        .ok_or_else(|| anyhow!("Invalid UTF-8 in filename"))?;
    let stem = Path::new(filename)
        .file_stem()
        .ok_or_else(|| anyhow!("Failed to get file stem"))?
        .to_str()
        .ok_or_else(|| anyhow!("Invalid UTF-8 in stem"))?;

    let pmtiles_path = dir.join(format!("{}.pmtiles", stem));

    // Create a progress bar for this file
    let file_style = ProgressStyle::default_bar()
        .template("  {spinner:.yellow} {prefix:<12.bold} [{bar:20.yellow/dim}] {msg}")
        .unwrap()
        .progress_chars("━━╺");

    let file_pb = multi_progress.add(ProgressBar::new(100));
    file_pb.set_style(file_style);
    file_pb.set_prefix(stem.to_string());
    file_pb.enable_steady_tick(std::time::Duration::from_millis(80));

    let start_time = Instant::now();

    // Stage 1: Add overviews
    file_pb.set_position(0);
    file_pb.set_message("[1/3] Adding overviews...");
    stage_counts[0].fetch_add(1, Ordering::Relaxed);
    update_stage_progress(stage_counts, stage_pb, 1);

    add_overviews(mbtiles_path)?;

    stage_counts[0].fetch_sub(1, Ordering::Relaxed);
    file_pb.set_position(33);

    // Stage 2: MBTiles -> PMTiles
    file_pb.set_message("[2/3] Converting to PMTiles...");
    stage_counts[1].fetch_add(1, Ordering::Relaxed);
    update_stage_progress(stage_counts, stage_pb, 1);

    mbtiles_to_pmtiles_with_progress(mbtiles_path, &pmtiles_path, &file_pb)?;

    stage_counts[1].fetch_sub(1, Ordering::Relaxed);
    file_pb.set_position(66);

    // Stage 3: Upload
    file_pb.set_message("[3/3] Uploading to R2...");
    stage_counts[2].fetch_add(1, Ordering::Relaxed);
    update_stage_progress(stage_counts, stage_pb, 1);

    upload_pmtiles(&pmtiles_path, remote)?;

    stage_counts[2].fetch_sub(1, Ordering::Relaxed);
    file_pb.set_position(100);

    // Cleanup if requested
    if cleanup {
        fs::remove_file(mbtiles_path).ok();
    }

    let elapsed = start_time.elapsed();
    let size_mb = pmtiles_path.metadata().map(|m| m.len() as f64 / 1024.0 / 1024.0).unwrap_or(0.0);

    file_pb.finish_and_clear();

    completed_count.fetch_add(1, Ordering::Relaxed);
    multi_progress.println(format!(
        "  ✔ {:<12} {:>6.1} MB  ({:.1}s)",
        stem, size_mb, elapsed.as_secs_f64()
    ))?;

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
    file_pb: &ProgressBar,
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

        // Update progress every 5000 tiles
        if tile_count % 5000 == 0 {
            // Map tile progress to 33-66% range (stage 2)
            let tile_pct = (tile_count as f64 / total_tiles as f64 * 100.0) as u64;
            let overall_pct = 33 + (tile_pct * 33 / 100);
            file_pb.set_position(overall_pct.min(65));
            file_pb.set_message(format!(
                "[2/3] Converting... {}%",
                tile_pct
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
