# Duplicate extraction cleanup — 2026-09-19

Completed the requested cleanup of
`/Volumes/projects/rawtiffs_dup_extractions_2026-08-29`.

- Removed **15 exact duplicates, 28,192,092 bytes (28.19 MB)**.
- Preserved the folder's 10,244-byte `.DS_Store` in the audit metadata directory.
- Removed all three empty source directories, including the requested root.
- Rehashed all 15 retained copies after deletion; all are unchanged. The active
  `master_dole_v2.csv` is unchanged. No verification issues remain.

The folder contained the TIFF, world-file and HTML sidecar triplets for the
**2026-05-14** Honolulu, Mariana and Samoan insets and the Eastern and Western
Aleutian chart halves. Each matches its catalogued source under
`/Volumes/projects/rawtiffs/faa_chart_splits/faasplit_05-14-2026_<location>/`.
The existing historical entries in `worklists/superseded_sources.csv` identify
these as duplicate extractions moved aside on 2026-08-29. This cleanup completes
that disposition; no unique charts required relocation or catalog changes.

Every source and retained copy was fully SHA-256 hashed, with regular-file,
path, device/inode/size/mtime/ctime checks before deletion. Finder metadata was
copied and hash-verified before its original was removed. Original file metadata
and extended attributes are recorded in the fixed plan and action log.

[Per-file manifest](data/dup_extractions_cleanup_2026-09-19/manifest.csv) ·
[Final verification](data/dup_extractions_cleanup_2026-09-19/final_summary.json)

Complete audit records: `worklists/data/dup_extractions_cleanup_2026-09-19/`
(gitignored). The inventory covered all 16 live files recursively, including
hidden metadata. No rendering or pixel-equivalence assumptions were used.
