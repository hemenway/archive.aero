# C2 budget change request — pending owner / four-section agreement

The frozen v1 builder and schema are unchanged. The production inputs are not
available in this checkout; this is an estimate, not a measurement of live data.

Run `python3 scripts/next_build_manifest.py --estimate-from dates.csv`. For 3,761
eras the deterministic model produces **193,883 bytes gzip**. It models 16% of
archives with continental coverage (150 z6 cells), the others with 1–12 cells,
4-decimal bounds derived from those cells, random 12-hex content hashes, 8% null bounds, and
one changing coverage segment per era. Bounds, hashes and dates alone already
make the 80,000-byte target difficult. Real repeated bounds may compress better;
actual directory-derived coverage and the real coverage sweep are the final test.

Proposed C2 v2: retain a <=80 KB bootstrap manifest with bases and overlay refs,
then replace the inline era array with decade-shard descriptors, each carrying an
immutable `next/eras.{decade}.{hash}.json` path, count, and start/end limits. Move
coverage verbatim to its own immutable JSON reference if required. Each era record
retains exactly the v1 shape and global chronological paint order. Consumers load
all era shards before advertising timeline readiness, or coordinate a subsequent
lazy-loading contract separately. This keeps caches immutable and avoids changing
per-era semantics to save bytes.

The shell and data-plane sections must agree on startup loading, failure handling
and new fields before this is adopted. **No split format is implemented here.**
The current builder fails above 80,000 bytes unless the owner explicitly passes
`--allow-over-budget`; staging validation also needs that flag for such artifacts.
