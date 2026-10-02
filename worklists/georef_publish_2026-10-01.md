# Georef publication — 2026-10-01

Completed: **17 affected eras and 30 full-sheet chart artifacts**, including all
23 donated faces (12 sectional sheets) from Russ Roslewski and Mike O'Barr.
Other changes cover Tulsa 1944, San Francisco 1966, Denver 1972 and Wichita 1972;
Tulsa 1942's corrected end date was rebuilt alongside the new Tulsa 1943 edition.
The independent SF 1966 scan was warp-validated; its higher-resolution LOC
alternate remains the winning source for the era and permanent chart artifact.

- Each artifact was uploaded by copy, size-checked in R2 and read back byte for
  byte through every 1 MiB CDN block with no-cache. Existing addresses remain.
- Metadata: `metadata-1c04feec.bundle`; **3,761 active era references**.
  `1942-07-23_to_1943-01-20` was removed from active dates after the catalog's
  Tulsa end-date correction; its published object remains permanently available.
- Lanczos warps; LZW mosaics on the converter's tile grid. Converter:
  b300c9e plus the verified half-pixel sampling correction, SHA-256
  `555ae236f4f9426adb81cdbbba3b71eab60ad76a38bc8c9b70aae8fb60d66ec1`.
- Wichita WASP 418 sides were made explicit (`_01=south`, `_02=north`).
  Modern SP 33°20′/38°40′ reduced corner residuals from 317/283 m to
  117/74 m. Fold-edge cutlines were traced before mosaicking.
- Traced fold cutlines for all 22 donated half faces and retraced Denver 1972
  after the GCP edits. Cached DFW 1969 warps were cropped on their original
  pixel grid; sampled interior RGB pixels remained byte-identical.
- Cincinnati 2012 north: the west neatline prints 85°W; corrected two saved
  80°W entries to 85°W and verified complete paired coverage.
- The sparse 1969 era exceeded the initial one-hour conversion limit. It was
  converted from six tight crops of the completed mosaic at the same z13
  resolution; sample RGBA pixels and global bounds were verified identical.
  Alignment checks also cover all five companion charts.
- Other CDN regions may retain older same-key blocks for up to 24 hours;
  every block was refreshed and checked at the publishing region.
- All 17 archives passed 37 tile/mosaic probes at a worst displacement of
  0.04 px (0.3 px tolerance). The Tulsa test measured 0.00 px at three probes.
  Rendered QA and full alignment results live with the row-level evidence.

Local build: `/Volumes/drive/georef_publish_2026-10-01/`.
Evidence: `worklists/data/georef_publish_2026-10-01/` (gitignored), including the
catalog snapshot, selected keys, manifest, logs and upload/read-back ledgers.
Published chart records were merged into the standing chart manifest/upload log
before regenerating and publishing timeline_data.json and coverage.json.

Still open: Boston 1957, SF 1971 half grouping, the Juneau Batch19 hold,
Hawaiian and simviation GCPs, Dutch Harbor 2004 south, ESRI SF 2008,
GlidePlan, Dallas 1981 north and the Denver 1975 seam. The separate modern-era
northward-shift republish queue continues independently.
