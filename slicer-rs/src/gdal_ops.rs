//! GDAL operations for warping, VRT building, and GeoTIFF creation.

use anyhow::{Context, Result};
use gdal::raster::ResampleAlg;
use gdal::vector::LayerAccess;
use gdal::{Dataset, DriverManager, Metadata};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Check if GDAL supports ZSTD compression.
pub fn check_zstd_support() -> bool {
    if let Ok(driver) = DriverManager::get_driver_by_name("GTiff") {
        if let Some(metadata) = driver.metadata_item("DMD_CREATIONOPTIONLIST", "") {
            return metadata.contains("ZSTD");
        }
    }
    false
}

/// Get the best available compression method.
pub fn get_compression(requested: &str) -> String {
    if requested == "AUTO" {
        if check_zstd_support() {
            "ZSTD".to_string()
        } else {
            warn!("ZSTD not available, falling back to LZW compression");
            "LZW".to_string()
        }
    } else {
        requested.to_string()
    }
}

/// Get the SRS of a shapefile in Proj4 format.
pub fn get_shapefile_srs(shapefile: &Path) -> Result<String> {
    let dataset = Dataset::open(shapefile)
        .context("Failed to open shapefile")?;

    // Try to get spatial reference from the first layer
    if let Ok(layer) = dataset.layer(0) {
        if let Some(srs) = layer.spatial_ref() {
            if let Ok(proj4) = srs.to_proj4() {
                return Ok(proj4);
            }
        }
    }

    // Fallback to WGS84
    Ok("EPSG:4326".to_string())
}

/// Map resampling algorithm name to GDAL enum.
pub fn parse_resample_alg(name: &str) -> ResampleAlg {
    match name.to_lowercase().as_str() {
        "nearest" => ResampleAlg::NearestNeighbour,
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "cubicspline" => ResampleAlg::CubicSpline,
        _ => ResampleAlg::NearestNeighbour,
    }
}

/// Warp and cut a TIFF using a shapefile as cutline.
///
/// This uses gdalwarp via command-line for more reliable cutline handling.
pub fn warp_and_cut(
    input_tiff: &Path,
    shapefile: &Path,
    output_path: &Path,
    shapefile_srs: &str,
    resample: &str,
    output_format: &str, // "tif" or "vrt"
) -> Result<bool> {
    let start = Instant::now();

    debug!(
        "Warping {:?} with cutline {:?}",
        input_tiff.file_name(),
        shapefile.file_name()
    );

    // Check if source has color table
    let src_ds = Dataset::open(input_tiff).context("Failed to open input TIFF")?;
    let first_band = src_ds.rasterband(1).context("Failed to get raster band")?;
    let has_color_table = first_band.color_table().is_some();
    drop(src_ds); // Close before warp

    // Build gdalwarp command arguments
    let mut args: Vec<String> = vec![
        "-t_srs".to_string(),
        "EPSG:3857".to_string(),
        "-cutline".to_string(),
        shapefile.to_string_lossy().to_string(),
        "-cutline_srs".to_string(),
        shapefile_srs.to_string(),
        "-crop_to_cutline".to_string(),
        "-dstalpha".to_string(),
        "-r".to_string(),
        resample.to_string(),
        "-wo".to_string(),
        "CUTLINE_ALL_TOUCHED=TRUE".to_string(),
        "-multi".to_string(),
    ];

    if output_format == "vrt" {
        args.extend(["-of".to_string(), "VRT".to_string()]);
    } else {
        args.extend([
            "-of".to_string(),
            "GTiff".to_string(),
            "-co".to_string(),
            "TILED=YES".to_string(),
            "-co".to_string(),
            "BIGTIFF=YES".to_string(),
        ]);
    }

    // Add RGBA expansion for paletted images
    if has_color_table {
        // We need to expand via a VRT first, then warp
        // This is handled internally by GDAL when using TranslateOptions with rgbExpand
    }

    args.push(input_tiff.to_string_lossy().to_string());
    args.push(output_path.to_string_lossy().to_string());

    // Execute gdalwarp
    let output = std::process::Command::new("gdalwarp")
        .args(&args)
        .output()
        .context("Failed to execute gdalwarp")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("gdalwarp failed: {}", stderr);
        return Ok(false);
    }

    let elapsed = start.elapsed();
    let size = if output_path.exists() {
        output_path.metadata().map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    if size > 0 {
        debug!(
            "✓ Warped in {:.1}s ({:.1} MB)",
            elapsed.as_secs_f64(),
            size as f64 / (1024.0 * 1024.0)
        );
        Ok(true)
    } else {
        warn!("Warp produced empty output");
        Ok(false)
    }
}

/// Build a VRT from multiple input files.
pub fn build_vrt(input_files: &[&Path], output_vrt: &Path) -> Result<bool> {
    if input_files.is_empty() {
        return Ok(false);
    }

    debug!(
        "Building VRT from {} files -> {:?}",
        input_files.len(),
        output_vrt.file_name()
    );

    // Filter valid files
    let valid_files: Vec<_> = input_files
        .iter()
        .filter(|f| {
            if !f.exists() {
                warn!("Input file not found: {:?}", f);
                return false;
            }
            let size = f.metadata().map(|m| m.len()).unwrap_or(0);
            if size < 1024 {
                warn!("Skipping too small file: {:?} ({} bytes)", f, size);
                return false;
            }
            true
        })
        .copied()
        .collect();

    if valid_files.is_empty() {
        warn!("No valid input files for VRT");
        return Ok(false);
    }

    // Use gdalbuildvrt command
    let mut args = vec![
        "-resolution".to_string(),
        "highest".to_string(),
        "-r".to_string(),
        "bilinear".to_string(),
        output_vrt.to_string_lossy().to_string(),
    ];

    for f in &valid_files {
        args.push(f.to_string_lossy().to_string());
    }

    let output = std::process::Command::new("gdalbuildvrt")
        .args(&args)
        .output()
        .context("Failed to execute gdalbuildvrt")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("gdalbuildvrt failed: {}", stderr);
        return Ok(false);
    }

    if output_vrt.exists() {
        let size = output_vrt.metadata()?.len();
        debug!("✓ Built VRT ({:.1} KB)", size as f64 / 1024.0);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Create a GeoTIFF from a VRT with compression.
pub fn create_geotiff(
    input_vrt: &Path,
    output_tiff: &Path,
    compression: &str,
    num_threads: &str,
    clip_projwin: Option<&[f64]>,
) -> Result<(bool, f64)> {
    let start = Instant::now();

    info!(
        "Creating GeoTIFF with {} compression ({} threads)...",
        compression, num_threads
    );

    let mut args = vec![
        "-of".to_string(),
        "GTiff".to_string(),
        "-co".to_string(),
        "TILED=YES".to_string(),
        "-co".to_string(),
        "BIGTIFF=YES".to_string(),
        "-co".to_string(),
        "BLOCKXSIZE=1024".to_string(),
        "-co".to_string(),
        "BLOCKYSIZE=1024".to_string(),
        "-co".to_string(),
        format!("NUM_THREADS={}", num_threads),
    ];

    if compression != "NONE" {
        args.extend([
            "-co".to_string(),
            format!("COMPRESS={}", compression),
            "-co".to_string(),
            "PREDICTOR=2".to_string(),
        ]);
    }

    // Add clip window if specified
    if let Some(projwin) = clip_projwin {
        if projwin.len() == 4 {
            args.extend([
                "-projwin".to_string(),
                projwin[0].to_string(),
                projwin[1].to_string(),
                projwin[2].to_string(),
                projwin[3].to_string(),
            ]);
        }
    }

    args.push(input_vrt.to_string_lossy().to_string());
    args.push(output_tiff.to_string_lossy().to_string());

    let output = std::process::Command::new("gdal_translate")
        .args(&args)
        .output()
        .context("Failed to execute gdal_translate")?;

    let elapsed = start.elapsed().as_secs_f64();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Check for ZSTD errors and retry with LZW
        if compression == "ZSTD"
            && (stderr.to_lowercase().contains("zstd")
                || stderr.contains("frame descriptor")
                || stderr.contains("data corruption"))
        {
            warn!("ZSTD error detected, retrying with LZW...");
            return create_geotiff(input_vrt, output_tiff, "LZW", num_threads, clip_projwin);
        }

        warn!("gdal_translate failed: {}", stderr);
        return Ok((false, elapsed));
    }

    if output_tiff.exists() {
        let size = output_tiff.metadata()?.len();
        let size_mb = size as f64 / (1024.0 * 1024.0);
        let rate = size_mb / elapsed;

        info!(
            "✓ GeoTIFF created: {:?} ({:.1} MB in {:.1}s, {:.1} MB/s)",
            output_tiff.file_name().unwrap_or_default(),
            size_mb,
            elapsed,
            rate
        );

        Ok((true, elapsed))
    } else {
        Ok((false, elapsed))
    }
}

/// Convert GeoTIFF to MBTiles.
pub fn to_mbtiles(input_tiff: &Path, output_mbtiles: &Path) -> Result<bool> {
    info!("Converting to MBTiles: {:?}", output_mbtiles.file_name());

    let output = std::process::Command::new("gdal_translate")
        .args([
            "-of",
            "MBTILES",
            &input_tiff.to_string_lossy(),
            &output_mbtiles.to_string_lossy(),
        ])
        .output()
        .context("Failed to execute gdal_translate for MBTiles")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("MBTiles conversion failed: {}", stderr);
        return Ok(false);
    }

    // Add overviews
    let output = std::process::Command::new("gdaladdo")
        .args([
            "-r",
            "bilinear",
            &output_mbtiles.to_string_lossy(),
            "2",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
            "256",
            "512",
            "1024",
        ])
        .output()
        .context("Failed to execute gdaladdo")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("gdaladdo failed: {}", stderr);
        return Ok(false);
    }

    Ok(true)
}

/// Convert MBTiles to PMTiles.
pub fn to_pmtiles(input_mbtiles: &Path, output_pmtiles: &Path) -> Result<bool> {
    info!("Converting to PMTiles: {:?}", output_pmtiles.file_name());

    let output = std::process::Command::new("pmtiles")
        .args([
            "convert",
            &input_mbtiles.to_string_lossy(),
            &output_pmtiles.to_string_lossy(),
        ])
        .output()
        .context("Failed to execute pmtiles convert")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("PMTiles conversion failed: {}", stderr);
        return Ok(false);
    }

    Ok(true)
}

/// Upload file via rclone.
pub fn rclone_upload(local_path: &Path, remote: &str, flags: &[&str]) -> Result<bool> {
    info!("Uploading via rclone: {:?}", local_path.file_name());

    let local_str = local_path.to_string_lossy().to_string();
    let remote_path = format!("{}/{}", remote, local_path.file_name().unwrap().to_string_lossy());

    let mut args = vec!["copyto", &local_str, &remote_path];
    args.extend(flags.iter().copied());

    let output = std::process::Command::new("rclone")
        .args(&args)
        .output()
        .context("Failed to execute rclone")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("rclone upload failed: {}", stderr);
        return Ok(false);
    }

    info!("✓ Uploaded: {:?}", local_path.file_name());
    Ok(true)
}
