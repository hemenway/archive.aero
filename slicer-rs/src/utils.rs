//! Utility functions for string normalization and sanitization.

use crate::config::ABBREVIATIONS;

/// Normalize a chart location name consistently.
///
/// Converts to lowercase, replaces spaces/hyphens/dots with underscores,
/// and applies known abbreviation mappings.
pub fn normalize_name(name: &str) -> String {
    let mut norm = name
        .trim()
        .to_lowercase()
        .replace(' ', "_")
        .replace('-', "_")
        .replace('.', "_");

    // Collapse multiple underscores
    while norm.contains("__") {
        norm = norm.replace("__", "_");
    }

    // Apply abbreviation mapping
    ABBREVIATIONS
        .get(norm.as_str())
        .map(|s| s.to_string())
        .unwrap_or(norm)
}

/// Sanitize a filename by removing special characters.
pub fn sanitize_filename(name: &str) -> String {
    let mut sanitized = name
        .replace(' ', "_")
        .replace('-', "_")
        .replace('.', "_")
        .replace('(', "")
        .replace(')', "")
        .replace(',', "");

    // Collapse multiple underscores
    while sanitized.contains("__") {
        sanitized = sanitized.replace("__", "_");
    }

    sanitized.to_lowercase()
}

/// Parse a date string in YYYY-MM-DD format.
pub fn parse_date(date_str: &str) -> Option<chrono::NaiveDate> {
    chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()
}

/// Format bytes as human-readable size.
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string.
pub fn format_duration(secs: f64) -> String {
    if secs >= 3600.0 {
        format!("{:.1} hours", secs / 3600.0)
    } else if secs >= 60.0 {
        format!("{:.1} min", secs / 60.0)
    } else {
        format!("{:.1}s", secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("Los Angeles"), "los_angeles");
        assert_eq!(normalize_name("New-York"), "new_york");
        assert_eq!(normalize_name("hawaiian_is"), "hawaiian_islands");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(
            sanitize_filename("Chart (2023).tif"),
            "chart_2023_tif"
        );
    }
}
