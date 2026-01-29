//! Main processing logic with parallel execution.

use crate::config::{MIN_FILE_SIZE, MIN_GEOTIFF_SIZE};
use crate::csv_loader::{get_all_locations, load_dole_data, DoleData};
use crate::downloader::{extract_zip, Downloader};
use crate::file_index::FileIndex;
use crate::gdal_ops::{build_vrt, create_geotiff, get_compression, get_shapefile_srs, warp_and_cut};
use crate::utils::{format_duration, normalize_name, parse_date, sanitize_filename};
use crate::Args;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{info, warn};

/// VRT library tracking warped files per location-date.
type VrtLibrary = HashMap<String, HashMap<String, PathBuf>>;

/// Run the main processing pipeline.
pub fn run(args: Args) -> Result<()> {
    let total_start = Instant::now();

    // Load CSV data
    let dole_data = load_dole_data(&args.csv)?;
    let all_locations = get_all_locations(&dole_data);

    info!(
        "Found {} locations from CSV",
        all_locations.len()
    );

    // Build file indexes
    let index = Arc::new(FileIndex::new());
    index.build_source_index(&args.source);
    index.build_shapefile_index(&args.shapefiles);

    // Count locations with shapefiles
    let shapefile_count = all_locations
        .iter()
        .filter(|loc| index.find_shapefile(loc).is_some())
        .count();
    info!(
        "{} locations have shapefiles ({} missing)",
        shapefile_count,
        all_locations.len() - shapefile_count
    );

    // Get compression method
    let compression = get_compression(&args.compression);
    info!("Using {} compression", compression);

    // Create downloader
    let downloader = Arc::new(Downloader::new(args.download_delay));

    // Skip user confirmation if --yes flag is set
    if !args.yes {
        print!("\nProceed with processing? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            info!("Processing cancelled by user.");
            return Ok(());
        }
    }

    // Download missing files if not skipped
    if !args.skip_downloads {
        download_missing_files(&args, &dole_data, &index, &downloader)?;
    }

    // Process all dates
    process_all_dates(&args, &dole_data, &all_locations, &index, &compression)?;

    let total_elapsed = total_start.elapsed();
    info!(
        "\n=== Processing Complete in {} ===",
        format_duration(total_elapsed.as_secs_f64())
    );

    Ok(())
}

/// Download all missing files upfront.
fn download_missing_files(
    args: &Args,
    dole_data: &DoleData,
    index: &Arc<FileIndex>,
    downloader: &Arc<Downloader>,
) -> Result<()> {
    info!("\n=== Preparing: Downloading missing files ===");

    // Collect missing files
    let mut missing: Vec<(&str, &str, &str, &str, bool)> = Vec::new();

    for (date, records) in dole_data {
        for rec in records {
            let Some(ref filename) = rec.filename else {
                continue;
            };
            let Some(ref download_link) = rec.download_link else {
                continue;
            };
            let location = rec.location.as_deref().unwrap_or("");

            // Check if file exists
            let is_zip = filename.to_lowercase().ends_with(".zip");
            if is_zip {
                let dir_name = &filename[..filename.len() - 4];
                let dir_path = args.source.join(dir_name);
                if !dir_path.exists() {
                    missing.push((filename, download_link, location, date, true));
                }
            } else {
                let file_path = args.source.join(filename);
                if !file_path.exists() {
                    missing.push((filename, download_link, location, date, false));
                }
            }
        }
    }

    if missing.is_empty() {
        info!("No missing files to download.");
        return Ok(());
    }

    // Sort by date (latest first)
    missing.sort_by(|a, b| b.3.cmp(a.3));

    info!(
        "Found {} missing files to download",
        missing.len()
    );
    info!(
        "Estimated time: ~{:.0} minutes",
        missing.len() as f64 * downloader.delay_secs() / 60.0
    );

    let mut downloaded = 0;
    let mut failed = 0;

    for (idx, (filename, download_link, location, date, is_zip)) in missing.iter().enumerate() {
        info!(
            "\n[{}/{}] {} - {}",
            idx + 1,
            missing.len(),
            date,
            location
        );

        let target_path = args.source.join(filename);

        match downloader.download(download_link, &target_path) {
            Ok(()) => {
                downloaded += 1;

                if *is_zip {
                    let dir_name = &filename[..filename.len() - 4];
                    let extract_dir = args.source.join(dir_name);

                    match extract_zip(&target_path, &extract_dir) {
                        Ok(tif_files) => {
                            index.add_extracted_dir(dir_name, tif_files);
                            info!("✓ Downloaded and extracted: {}", filename);
                        }
                        Err(e) => {
                            warn!("⚠ Downloaded but extraction failed: {} - {}", filename, e);
                            failed += 1;
                        }
                    }
                } else {
                    index.add_file(filename, target_path);
                    info!("✓ Downloaded: {}", filename);
                }
            }
            Err(e) => {
                warn!("✗ Failed to download: {} - {}", filename, e);
                failed += 1;
            }
        }
    }

    info!("\n=== Download Preparation Complete ===");
    info!("Downloaded: {}/{}", downloaded, missing.len());
    if failed > 0 {
        info!("Failed: {}", failed);
    }

    Ok(())
}

/// Process all dates with parallel warp operations.
fn process_all_dates(
    args: &Args,
    dole_data: &DoleData,
    all_locations: &[String],
    index: &Arc<FileIndex>,
    compression: &str,
) -> Result<()> {
    info!("\n=== Starting Chart Processing ===");

    // Sort dates oldest to newest (for fallback logic)
    let mut sorted_dates: Vec<_> = dole_data.keys().cloned().collect();
    sorted_dates.sort();

    let total_dates = sorted_dates.len();

    // Count dates needing GeoTIFF processing
    let dates_needing_geotiff: Vec<_> = sorted_dates
        .iter()
        .filter(|date| {
            let geotiff_path = args.output.join(format!("{}.tif", date));
            !geotiff_path.exists() || geotiff_path.metadata().map(|m| m.len()).unwrap_or(0) < MIN_GEOTIFF_SIZE
        })
        .cloned()
        .collect();

    info!(
        "GeoTIFFs to create: {} (skipping {} existing)",
        dates_needing_geotiff.len(),
        total_dates - dates_needing_geotiff.len()
    );

    // VRT library for fallback logic
    let vrt_library: Arc<Mutex<VrtLibrary>> = Arc::new(Mutex::new(HashMap::new()));

    // Rebuild VRT library from existing temp files
    rebuild_vrt_library(&args.temp_dir, all_locations, &vrt_library)?;

    // Collect GeoTIFF jobs
    let geotiff_jobs: Arc<Mutex<Vec<(PathBuf, PathBuf, String)>>> = Arc::new(Mutex::new(Vec::new()));

    // Progress bar
    let progress = ProgressBar::new(total_dates as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Invalid template")
            .progress_chars("=>-"),
    );

    // Process each date
    for (_date_idx, date) in sorted_dates.iter().enumerate() {
        progress.set_message(format!("Processing {}", date));

        // Check if output already exists
        let geotiff_path = args.output.join(format!("{}.tif", date));
        if geotiff_path.exists() && geotiff_path.metadata().map(|m| m.len()).unwrap_or(0) >= MIN_GEOTIFF_SIZE {
            progress.inc(1);
            continue;
        }

        let records = &dole_data[date];
        let temp_dir = args.temp_dir.join(date);
        std::fs::create_dir_all(&temp_dir)?;

        // Collect warp jobs
        let warp_jobs: Vec<_> = records
            .iter()
            .filter_map(|rec| {
                let location = rec.location.as_ref()?;
                let filename = rec.filename.as_ref()?;
                let norm_loc = normalize_name(location);

                // Find shapefile
                let shp_path = index.find_shapefile(location)?;

                // Find source files
                let src_files = index.resolve_filename(filename);
                if src_files.is_empty() {
                    return None;
                }

                Some((norm_loc, shp_path, src_files, date.clone()))
            })
            .collect();

        if warp_jobs.is_empty() {
            progress.inc(1);
            continue;
        }

        // Execute warp jobs in parallel
        let warp_results: Vec<_> = warp_jobs
            .par_iter()
            .flat_map(|(norm_loc, shp_path, src_files, _date)| {
                let shp_srs = get_shapefile_srs(shp_path).unwrap_or_else(|_| "EPSG:4326".to_string());

                src_files
                    .par_iter()
                    .filter_map(|src_file| {
                        let sanitized = sanitize_filename(
                            src_file.file_stem().unwrap_or_default().to_str().unwrap_or(""),
                        );
                        let output_suffix = if args.warp_output == "vrt" { ".vrt" } else { ".tif" };
                        let output_path = temp_dir.join(format!("{}_{}{}", norm_loc, sanitized, output_suffix));

                        // Skip if already exists
                        if output_path.exists() && output_path.metadata().map(|m| m.len()).unwrap_or(0) > MIN_FILE_SIZE {
                            return Some((norm_loc.clone(), output_path));
                        }

                        match warp_and_cut(
                            src_file,
                            shp_path,
                            &output_path,
                            &shp_srs,
                            &args.resample,
                            &args.warp_output,
                        ) {
                            Ok(true) => Some((norm_loc.clone(), output_path)),
                            _ => None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Group results by location
        let mut warped_by_location: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for (loc, path) in warp_results {
            if path.exists() && path.metadata().map(|m| m.len()).unwrap_or(0) > MIN_FILE_SIZE {
                warped_by_location.entry(loc).or_default().push(path);
            }
        }

        // Build VRTs for each location
        {
            let mut lib = vrt_library.lock().unwrap();
            for (norm_loc, warped_files) in warped_by_location {
                if warped_files.len() > 1 {
                    let loc_vrt = temp_dir.join(format!("{}_{}.vrt", norm_loc, date));
                    let refs: Vec<_> = warped_files.iter().map(|p| p.as_path()).collect();
                    if build_vrt(&refs, &loc_vrt).unwrap_or(false) {
                        lib.entry(norm_loc).or_default().insert(date.clone(), loc_vrt);
                    }
                } else if let Some(single) = warped_files.into_iter().next() {
                    lib.entry(norm_loc).or_default().insert(date.clone(), single);
                }
            }
        }

        // Build date matrix with fallback logic
        let date_matrix: HashMap<String, PathBuf> = {
            let lib = vrt_library.lock().unwrap();
            let date_dt = parse_date(date);

            all_locations
                .iter()
                .filter_map(|loc| {
                    let loc_data = lib.get(loc)?;

                    // Prefer exact date match
                    if let Some(path) = loc_data.get(date) {
                        return Some((loc.clone(), path.clone()));
                    }

                    // Fallback to most recent available date <= current date
                    if let Some(current_dt) = date_dt {
                        let fallback = loc_data
                            .iter()
                            .filter_map(|(d, p)| {
                                let dt = parse_date(d)?;
                                if dt <= current_dt && (current_dt - dt).num_days() <= args.fallback_window_days {
                                    Some((d, p, dt))
                                } else {
                                    None
                                }
                            })
                            .max_by_key(|(_, _, dt)| *dt);

                        if let Some((_, path, _)) = fallback {
                            return Some((loc.clone(), path.clone()));
                        }
                    }

                    None
                })
                .collect()
        };

        // Create mosaic VRT and queue GeoTIFF job
        if !date_matrix.is_empty() {
            let mosaic_vrts: Vec<_> = date_matrix.values().map(|p| p.as_path()).collect();
            let mosaic_vrt = temp_dir.join(format!("mosaic_{}.vrt", date));

            if build_vrt(&mosaic_vrts, &mosaic_vrt).unwrap_or(false) {
                geotiff_jobs.lock().unwrap().push((mosaic_vrt, geotiff_path, date.clone()));
            }
        }

        progress.inc(1);
    }

    progress.finish_with_message("Warp phase complete");

    // Process GeoTIFF jobs
    let jobs = geotiff_jobs.lock().unwrap();
    if !jobs.is_empty() {
        info!(
            "\n=== Processing {} GeoTIFFs ({} workers) ===",
            jobs.len(),
            args.parallel_geotiff
        );

        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.parallel_geotiff)
            .build()
            .unwrap();

        let completed = AtomicUsize::new(0);
        let total_jobs = jobs.len();
        let times: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));

        let clip_projwin = args.clip_projwin.as_deref();

        pool.install(|| {
            jobs.par_iter().for_each(|(input_vrt, output_tiff, date)| {
                let result = create_geotiff(
                    input_vrt,
                    output_tiff,
                    compression,
                    &args.num_threads,
                    clip_projwin,
                );

                let completed_count = completed.fetch_add(1, Ordering::SeqCst) + 1;

                match result {
                    Ok((true, elapsed)) => {
                        times.lock().unwrap().push(elapsed);
                        info!("✓✓✓ {} COMPLETE ({}/{})", date.to_uppercase(), completed_count, total_jobs);

                        // Calculate ETA
                        let t = times.lock().unwrap();
                        if !t.is_empty() {
                            let avg = t.iter().sum::<f64>() / t.len() as f64;
                            let remaining = total_jobs - completed_count;
                            let eta = avg * remaining as f64;
                            info!("  ETA: {} ({} remaining)", format_duration(eta), remaining);
                        }
                    }
                    Ok((false, _)) => {
                        warn!("✗ GeoTIFF failed for {}", date);
                    }
                    Err(e) => {
                        warn!("✗ GeoTIFF error for {}: {}", date, e);
                    }
                }
            });
        });
    }

    Ok(())
}

/// Rebuild VRT library from existing temp files for resume support.
fn rebuild_vrt_library(
    temp_dir: &Path,
    all_locations: &[String],
    vrt_library: &Arc<Mutex<VrtLibrary>>,
) -> Result<()> {
    if !temp_dir.exists() {
        return Ok(());
    }

    info!("Rebuilding VRT library from existing temp files...");

    let mut found_count = 0;
    let mut lib = vrt_library.lock().unwrap();

    for entry in std::fs::read_dir(temp_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let date = entry.file_name().to_string_lossy().to_string();
        if !date.contains('-') {
            continue;
        }

        let date_dir = entry.path();

        // Look for per-location VRTs
        for vrt_entry in std::fs::read_dir(&date_dir)? {
            let vrt_entry = vrt_entry?;
            let path = vrt_entry.path();

            if path.extension().map_or(false, |ext| ext == "vrt") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

                // Skip mosaic VRTs and intermediate VRTs
                if stem.starts_with("mosaic_") || stem.ends_with("_rgba") || stem.ends_with("_src") {
                    continue;
                }

                // Parse location from filename: {location}_{date}.vrt
                if stem.ends_with(&format!("_{}", date)) {
                    let location = &stem[..stem.len() - date.len() - 1];
                    if all_locations.contains(&location.to_string()) {
                        lib.entry(location.to_string())
                            .or_default()
                            .insert(date.clone(), path);
                        found_count += 1;
                    }
                }
            }
        }
    }

    if found_count > 0 {
        info!("  Restored {} location-date entries", found_count);
    } else {
        info!("  No existing VRTs found (fresh run)");
    }

    Ok(())
}
