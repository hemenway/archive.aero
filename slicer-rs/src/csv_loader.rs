//! CSV loading and parsing for master_dole.csv data.

use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;
use tracing::info;

/// Regex for extracting Wayback Machine timestamps
static WAYBACK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/web/(\d{14})/").expect("Invalid regex"));

/// A record from master_dole.csv
#[derive(Debug, Deserialize, Clone)]
pub struct DoleRecord {
    /// Date in YYYY-MM-DD format
    pub date: String,

    /// Location/chart name
    pub location: Option<String>,

    /// Edition number or identifier
    pub edition: Option<String>,

    /// Filename of the chart file
    pub filename: Option<String>,

    /// Download URL (often Wayback Machine)
    pub download_link: Option<String>,

    /// Extracted Wayback Machine timestamp (not in CSV, derived)
    #[serde(skip)]
    pub wayback_ts: Option<String>,
}

/// Grouped data by date
pub type DoleData = HashMap<String, Vec<DoleRecord>>;

/// Load and parse the master_dole.csv file.
///
/// Returns a HashMap where keys are dates (YYYY-MM-DD) and values are
/// vectors of records for that date.
pub fn load_dole_data(csv_path: &Path) -> Result<DoleData> {
    info!("Loading CSV: {:?}", csv_path);

    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true) // Handle varying column counts
        .from_path(csv_path)?;

    let mut data: DoleData = HashMap::new();
    let mut record_count = 0;

    for result in rdr.deserialize() {
        let mut record: DoleRecord = match result {
            Ok(r) => r,
            Err(e) => {
                // Skip malformed rows with a warning
                tracing::warn!("Skipping malformed row: {}", e);
                continue;
            }
        };

        // Skip records without a date
        if record.date.is_empty() {
            continue;
        }

        // Extract Wayback Machine timestamp from download_link
        if let Some(ref link) = record.download_link {
            if link.contains("web.archive.org/web/") {
                if let Some(caps) = WAYBACK_RE.captures(link) {
                    record.wayback_ts = Some(caps[1].to_string());
                }
            }
        }

        data.entry(record.date.clone()).or_default().push(record);
        record_count += 1;
    }

    info!("Loaded {} records across {} dates", record_count, data.len());
    Ok(data)
}

/// Get all unique locations from the dole data.
pub fn get_all_locations(data: &DoleData) -> Vec<String> {
    use crate::utils::normalize_name;
    use std::collections::HashSet;

    let mut locations: HashSet<String> = HashSet::new();

    for records in data.values() {
        for rec in records {
            if let Some(ref loc) = rec.location {
                let norm = normalize_name(loc);
                if !norm.is_empty() {
                    locations.insert(norm);
                }
            }
        }
    }

    let mut sorted: Vec<_> = locations.into_iter().collect();
    sorted.sort();
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wayback_extraction() {
        let link = "https://web.archive.org/web/20150301120000/http://example.com/file.zip";
        if let Some(caps) = WAYBACK_RE.captures(link) {
            assert_eq!(&caps[1], "20150301120000");
        } else {
            panic!("Regex should match");
        }
    }
}
