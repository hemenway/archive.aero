//! File indexing system for O(1) lookups.
//!
//! Uses DashMap for thread-safe concurrent access without locks.

use crate::utils::normalize_name;
use dashmap::DashMap;
use std::path::{Path, PathBuf};
use tracing::info;
use walkdir::WalkDir;

/// Thread-safe file index for fast lookups.
pub struct FileIndex {
    /// Direct TIF and ZIP files by filename
    pub files_by_name: DashMap<String, PathBuf>,

    /// TIF files grouped by parent directory name (for extracted ZIPs)
    pub tifs_by_dir: DashMap<String, Vec<PathBuf>>,

    /// Shapefiles indexed by normalized location name
    pub shapefile_index: DashMap<String, PathBuf>,
}

impl FileIndex {
    /// Create a new empty file index.
    pub fn new() -> Self {
        Self {
            files_by_name: DashMap::new(),
            tifs_by_dir: DashMap::new(),
            shapefile_index: DashMap::new(),
        }
    }

    /// Build the source file index by walking the source directory.
    ///
    /// Indexes:
    /// - All .tif and .zip files by filename
    /// - TIF files inside subdirectories, grouped by parent directory name
    pub fn build_source_index(&self, source_dir: &Path) {
        info!("Building source file index...");

        let source_dir_canon = source_dir
            .canonicalize()
            .unwrap_or_else(|_| source_dir.to_path_buf());

        for entry in WalkDir::new(source_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };

            let filename_lower = filename.to_lowercase();

            // Index TIF and ZIP files by name
            if filename_lower.ends_with(".tif") || filename_lower.ends_with(".zip") {
                self.files_by_name
                    .insert(filename.to_string(), path.to_path_buf());
            }

            // For TIFs in subdirectories, also index by parent directory
            if filename_lower.ends_with(".tif") {
                if let Some(parent) = path.parent() {
                    let parent_canon = parent
                        .canonicalize()
                        .unwrap_or_else(|_| parent.to_path_buf());

                    // Only index if parent is not source_dir itself
                    if parent_canon != source_dir_canon {
                        if let Some(parent_name) = parent.file_name().and_then(|n| n.to_str()) {
                            self.tifs_by_dir
                                .entry(parent_name.to_string())
                                .or_default()
                                .push(path.to_path_buf());
                        }
                    }
                }
            }
        }

        info!(
            "  Indexed {} files and {} directories",
            self.files_by_name.len(),
            self.tifs_by_dir.len()
        );
    }

    /// Build the shapefile index by scanning the shapefiles directory.
    pub fn build_shapefile_index(&self, shape_dir: &Path) {
        info!("Building shapefile index...");

        let sectional_dir = shape_dir.join("sectional");
        if !sectional_dir.exists() {
            tracing::warn!(
                "Sectional shapefile directory not found: {:?}",
                sectional_dir
            );
            return;
        }

        for entry in WalkDir::new(&sectional_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "shp") {
                if let Some(stem) = path.file_stem().and_then(|n| n.to_str()) {
                    let normalized = normalize_name(stem);
                    self.shapefile_index
                        .insert(normalized, path.to_path_buf());
                }
            }
        }

        info!("  Indexed {} shapefiles", self.shapefile_index.len());
    }

    /// Find a shapefile for the given location.
    ///
    /// First tries exact match, then partial/substring matches.
    pub fn find_shapefile(&self, location: &str) -> Option<PathBuf> {
        let norm_loc = normalize_name(location);

        // Exact match first
        if let Some(entry) = self.shapefile_index.get(&norm_loc) {
            return Some(entry.value().clone());
        }

        // Partial match (location is substring of shapefile name)
        for entry in self.shapefile_index.iter() {
            if entry.key().contains(&norm_loc) {
                return Some(entry.value().clone());
            }
        }

        None
    }

    /// Resolve a filename from CSV to actual file path(s).
    ///
    /// Handles:
    /// - Direct .tif files
    /// - .zip files (looks for extracted directory)
    pub fn resolve_filename(&self, filename: &str) -> Vec<PathBuf> {
        if filename.is_empty() {
            return vec![];
        }

        // Handle ZIP files (look for extracted directory)
        if filename.to_lowercase().ends_with(".zip") {
            let dir_name = &filename[..filename.len() - 4];

            // Check tifs_by_dir index
            if let Some(entry) = self.tifs_by_dir.get(dir_name) {
                return entry.value().clone();
            }

            return vec![];
        }

        // Direct TIF file
        if let Some(entry) = self.files_by_name.get(filename) {
            return vec![entry.value().clone()];
        }

        vec![]
    }

    /// Add a newly downloaded/extracted file to the index.
    pub fn add_file(&self, filename: &str, path: PathBuf) {
        self.files_by_name.insert(filename.to_string(), path);
    }

    /// Add TIF files from an extracted directory.
    pub fn add_extracted_dir(&self, dir_name: &str, tif_files: Vec<PathBuf>) {
        // Add to tifs_by_dir
        self.tifs_by_dir.insert(dir_name.to_string(), tif_files.clone());

        // Also add individual files to files_by_name
        for path in tif_files {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                self.files_by_name.insert(filename.to_string(), path);
            }
        }
    }
}

impl Default for FileIndex {
    fn default() -> Self {
        Self::new()
    }
}
