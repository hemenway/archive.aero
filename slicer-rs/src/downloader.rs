//! HTTP download functionality with throttling and retry logic.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Download manager with throttling and retry logic.
pub struct Downloader {
    /// Timestamp of last download (for throttling)
    last_download: Mutex<Instant>,

    /// Delay between downloads
    delay: AtomicU64, // stored as milliseconds

    /// Maximum number of retries
    max_retries: u32,

    /// HTTP client
    client: reqwest::blocking::Client,
}

impl Downloader {
    /// Create a new downloader with the specified delay between downloads.
    pub fn new(delay_secs: f64) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) archive.aero/slicer")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            last_download: Mutex::new(Instant::now() - Duration::from_secs(100)),
            delay: AtomicU64::new((delay_secs * 1000.0) as u64),
            max_retries: 3,
            client,
        }
    }

    /// Get the current delay in seconds.
    pub fn delay_secs(&self) -> f64 {
        self.delay.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Increase delay (for adaptive throttling on errors).
    pub fn increase_delay(&self) {
        let current = self.delay.load(Ordering::Relaxed);
        let new_delay = (current as f64 * 1.5).max(current as f64 + 5000.0) as u64;
        self.delay.store(new_delay, Ordering::Relaxed);
        warn!(
            "Increasing download delay from {:.1}s to {:.1}s",
            current as f64 / 1000.0,
            new_delay as f64 / 1000.0
        );
    }

    /// Download a file from URL to the target path.
    pub fn download(&self, url: &str, target: &Path) -> Result<()> {
        // Throttle: wait if necessary
        {
            let last = *self.last_download.lock().unwrap();
            let delay = Duration::from_millis(self.delay.load(Ordering::Relaxed));
            let elapsed = last.elapsed();

            if elapsed < delay {
                let wait = delay - elapsed;
                info!("Waiting {:.1}s to avoid rate limiting...", wait.as_secs_f64());
                std::thread::sleep(wait);
            }
        }

        let mut retries = 0;
        let mut backoff = Duration::from_secs(5);

        loop {
            match self.try_download(url, target) {
                Ok(()) => {
                    // Update last download time
                    *self.last_download.lock().unwrap() = Instant::now();
                    return Ok(());
                }
                Err(e) => {
                    retries += 1;

                    // Check for connection reset errors
                    let error_str = e.to_string();
                    if error_str.contains("Connection reset")
                        || error_str.contains("Connection aborted")
                    {
                        self.increase_delay();
                    }

                    if retries >= self.max_retries {
                        // Clean up partial file
                        let _ = std::fs::remove_file(target);
                        return Err(e).context(format!(
                            "Download failed after {} retries",
                            self.max_retries
                        ));
                    }

                    warn!(
                        "Download failed (attempt {}/{}): {}",
                        retries, self.max_retries, e
                    );
                    info!("Retrying in {:?}...", backoff);

                    // Remove partial file before retry
                    let _ = std::fs::remove_file(target);

                    std::thread::sleep(backoff);
                    backoff *= 2; // Exponential backoff
                }
            }
        }
    }

    /// Attempt a single download.
    fn try_download(&self, url: &str, target: &Path) -> Result<()> {
        info!("Downloading: {}", url);

        let response = self
            .client
            .get(url)
            .send()
            .context("Failed to send request")?
            .error_for_status()
            .context("Server returned error status")?;

        let total_size = response
            .content_length()
            .unwrap_or(0);

        // Create progress bar
        let pb = if total_size > 0 {
            let pb = ProgressBar::new(total_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-"),
            );
            Some(pb)
        } else {
            None
        };

        // Create target file
        let mut file = File::create(target).context("Failed to create target file")?;

        // Download in chunks
        let mut downloaded: u64 = 0;
        let mut reader = response;

        loop {
            let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer
            let bytes_read = std::io::Read::read(&mut reader, &mut buffer)?;

            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            if let Some(ref pb) = pb {
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("done");
        }

        let file_size = target.metadata()?.len();
        info!(
            "✓ Downloaded: {} ({:.1} MB)",
            target.file_name().unwrap_or_default().to_string_lossy(),
            file_size as f64 / (1024.0 * 1024.0)
        );

        Ok(())
    }
}

/// Extract a ZIP file to a directory.
pub fn extract_zip(zip_path: &Path, extract_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    use std::io::Read;

    if extract_dir.exists() {
        info!("Directory already exists: {:?}", extract_dir);
        // Return existing TIF files
        return Ok(walkdir::WalkDir::new(extract_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().is_file()
                    && e.path()
                        .extension()
                        .map_or(false, |ext| ext.eq_ignore_ascii_case("tif"))
            })
            .map(|e| e.path().to_path_buf())
            .collect());
    }

    info!("Extracting: {:?}", zip_path);

    let file = File::open(zip_path).context("Failed to open ZIP file")?;
    let mut archive = zip::ZipArchive::new(file).context("Failed to read ZIP archive")?;

    std::fs::create_dir_all(extract_dir)?;

    let mut tif_files = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => extract_dir.join(path),
            None => continue,
        };

        if file.is_dir() {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;

            // Track TIF files
            if outpath
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("tif"))
            {
                tif_files.push(outpath);
            }
        }
    }

    info!("✓ Extracted {} files", archive.len());
    Ok(tif_files)
}
