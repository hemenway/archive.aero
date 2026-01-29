//! Configuration and constants for the slicer.

use std::collections::HashMap;
use std::sync::LazyLock;

/// Abbreviation mappings for location normalization
pub static ABBREVIATIONS: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("hawaiian_is", "hawaiian_islands");
    m.insert("dallas_ft_worth", "dallas_ft_worth");
    m.insert("mariana_islands_inset", "mariana_islands");
    m
});

/// Default remote for rclone uploads
pub const UPLOAD_REMOTE: &str = "r2:charts/sectionals";

/// Default rclone flags for upload
pub const RCLONE_FLAGS: &[&str] = &[
    "-P",
    "--s3-upload-concurrency",
    "16",
    "--s3-chunk-size",
    "128M",
    "--buffer-size",
    "128M",
    "--s3-disable-checksum",
    "--stats",
    "1s",
];

/// Minimum file size (1KB) to consider a file valid
pub const MIN_FILE_SIZE: u64 = 1024;

/// Minimum GeoTIFF size (1MB) to consider it complete
pub const MIN_GEOTIFF_SIZE: u64 = 1024 * 1024;
