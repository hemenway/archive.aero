# Deep archive consolidation — 2026-09-19

Completed the user's request to file everything in `/Volumes/projects/deep archive`
according to archive.aero's storage conventions and delete confirmed duplicates.

- **823 original files, 101.27 GB, moved** to
  `/Volumes/projects/rawtiffs_attic/deep_archive_2026-09-19/` with original paths and names.
- **2,680 duplicates, 157.95 GB, removed** after complete-file SHA-256 verification.
  2,674 proofs came from the earlier same-day audit, revalidated against current
  device/inode/size/mtime/ctime identities; six more were hashed in this cleanup.
- The old `deep archive` directory was emptied and removed.
- Every retained path exists; all 22,632 pre-existing rawtiffs files and the active
  catalog are unchanged. The final verification reported no issues.
- Three Finder metadata files recreated during the work were also preserved under
  `finder_metadata_regenerated/`. They are additional to the 823 inventoried files.

## Why the remaining files belong in the chart attic

The retained material is legacy non-sectional FAA content, whole-cycle ZIP bundles,
partial/failed downloads, and the Dallas collection's source variants, converted
TIFFs, GeoTIFFs/COGs, georeferencing scripts and sidecars. CLAUDE.md assigns recovered
chart material that is already represented, not yet catalogued, derived or outside
current pipeline scope to `rawtiffs_attic`; original source indexing stays in
`rawtiffs`. No unverified file was promoted or catalogued. The Hawaiian Islands SEC
95 ZIP also contains inset charts; its main chart already has a NARA catalog row.

| Preserved group | Files | GB |
| --- | ---: | ---: |
| Folder metadata | 1 | 0.00 |
| FAA_Charts | 7 | 0.03 |
| FAA_Charts2 | 584 | 51.41 |
| archive of just dallas sectionals | 231 | 49.84 |

## Evidence and limits

The [full manifest](data/deep_archive_cleanup_2026-09-19/manifest.csv) maps all 3,503
original paths to their retained copies. The JSONL manifest includes source inode,
size, condition, reason and available ZIP member metadata. A copy and `README.txt`
are stored beside the retained collection.

Complete logs, original file metadata, fixed inventories, plans, inspection results
and [final verification](data/deep_archive_cleanup_2026-09-19/final_summary.json)
are in `worklists/data/deep_archive_cleanup_2026-09-19/` (gitignored).
The earlier proof database remains in `worklists/data/projects_cleanup/2026-09-19/`.

Moves stayed on the same filesystem and retained file inode, size and mtime.
macOS changed its own `com.apple.provenance` attribute; original extended attributes
were exported before each operation. Finder rewrote some `.DS_Store` metadata;
these are explicitly distinguished from chart content in the manifest.

Unmatched does **not** mean unique. This work inspected ZIP directory structure but
did not decompress every member, render every chart, compare pixels or perform a
new edition-by-edition catalog audit. Partial downloads and HTML error responses
saved with `.zip` names remain preserved and labelled. Retired Dallas scripts may
refer to duplicate inputs now found through the manifest.
