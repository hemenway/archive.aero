# Data contract implementation notes

Implemented against the supplied frozen v1 C1–C5. No production data was read,
no server-side copy was executed, no Worker was deployed, and no routes or
wrangler configuration were changed. Work is on `redesign/data-contract` in the
managed data-contract worktree. Generated fixtures are ignored locally by
`next/contract/.gitignore`.

## Worker

`tiles.js` is a small dependency-free PMTiles v3 decoder. This avoids adding an
npm/bundling dependency to the existing Worker and makes its Node fixture server
import exactly the same handler. Hilbert IDs, run-length entries, offset-zero
contiguous entries, root/leaf traversal, uncompressed and gzip directories, tile
compression, and metadata all follow the
[PMTiles v3 specification](https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md).
`DecompressionStream('gzip')` is a Workers runtime API. `encodeBody: 'manual'`
preserves stored gzip payloads without recompressing them; the latest downloaded
Workers types (5.20261002.1) confirm both APIs. Existing tests run unchanged.

The only storage access is the original `getBlock`, injected by index.js. The
canonical raw archive URL supplies its block-cache key, so range clients and
`/t/` reuse the same 1 MiB entries and existing bounded flights. No tile-response
Cache API entries are added. Each request retains at most two already-read blocks
to close the flight-settlement / asynchronous cache-put gap; no new cross-request
I/O promises are shared. All cache fills remain under the existing `waitUntil`.

`/t/` is a **reserved virtual namespace**. The bucket has historically published
`sectionals/`, `basemap/`, `airspace/` and static data keys, but no inventory was
available here to prove absence of `t/` objects. The owner must check a complete
object listing for keys beginning `t/` before deployment and prohibit that prefix
for uploads afterward. Under that explicit assumption `/t/` cannot shadow a real
key. It is impossible to both reserve every `/t/` URL and preserve an arbitrary
pre-existing object with that exact URL without such a namespace rule. All other
URLs retain the old range implementation. C1 responses and preflight use `*`
even if legacy `ALLOWED_ORIGIN` is narrower. `/t/` ignores client Range headers.

Parsed header/directory LRU accounting is capped at **16,777,216 bytes** per
isolate. Directory entries use one Float64Array, 32 bytes per entry; headers are
charged 512 bytes, plus key UTF-16 bytes and 128 bytes entry overhead. Map/engine
allocator overhead is implementation-dependent, so this is an accounted cache
bound, not a claim of exact heap size. Compressed and inflated directories and
parsed arrays each have an 8 MiB ceiling. A cold parse can transiently hold the
compressed input, gzip chunks, concatenated bytes and parsed array in addition to
the LRU and two local blocks. A production directory above those limits fails
retryably and must be repacked with smaller leaves. Metadata uses the same 8 MiB
limit. No persistent metadata payload or tile bytes live in this LRU.

No negative isolate cache is kept. 404 TTL is 60 seconds; 200/204 and metadata
use the exact one-year immutable C1 policy. HEAD resolves the directory but never
reads tile payloads just for a body. In a cold isolate header/directory block reads
can necessarily include nearby tile bytes. Analytics Engine retains today's
index, five blobs and three doubles and samples successful GET tile hits,
including tiny MVTs, while metadata and HEAD are excluded.

## Cold viewport measurement

Run `node next/contract/fixtures/benchmark.mjs`. Thirty unique 90,000-byte tile
payloads in Hilbert order, with a gzip leaf directory, produced:

| Mode | Browser requests | R2 GETs | Edge entries | Bytes delivered |
| --- | ---: | ---: | ---: | ---: |
| Current client: 16 KB probe + leaf + tile ranges | 32 | 3 | 3 | 2,716,427 |
| `/t/` | 30 | 3 | 3 | 2,700,000 |

Both use the already-improved block cache. Claiming another R2 reduction from
moving directories to the server would be incorrect for this cold viewport.
The benefit is fewer browser requests, no client-side directory state, and parsed
directory reuse across viewers. R2 read counts depend on actual tile size,
Hilbert locality, padding, leaf placement and edge-cache availability. Separate
tests prove thirty simultaneous lookups coalesce to one R2 read when they occupy
one block, and directory reuse avoids header/leaf reads after edge eviction.

## Builders

The Python codec reads directories independent of tile content and supports
uncompressed/gzip internal and tile compression. Its writer orders tiles by
Hilbert ID, joins identical consecutive tile runs and deduplicates identical
contents by SHA256 with a byte comparison. Tile payloads spool to a temporary
file; filename/directory/hash indexes still scale with the tile count. Large
production archives need disk scratch space and RAM for their index. It emits
gzip roots and optimized leaves with a root under the v3 16 KB metadata probe.

Versioning computes full SHA256, uses its first 12 hex digits, and includes the
extension-less legacy chart keys. A listing can supply trusted full SHA256 or
be combined with a local mirror. ETags are never treated as hashes. Optional
`--read-remote` hashes object bodies using environment credentials; optional
`--execute` performs managed multipart server-side copies. Neither was run.
Plans include zoom/bounds for pin building. Existing versioned objects are never
modified: Python execution refuses destinations without matching full SHA256
metadata and guards copy sources with their ETag. Freeze source writers while
local hashes and remote copies are being matched. Generated shell commands use `rclone copyto --immutable`; execution
inside the Python tool requires the explicit `--execute` and all three credential
variables, without using ambient boto profiles.

The manifest reuses `parse_key_dates` and `header_bounds` validation from today's
metadata bundle; the bundle import now exposes these helpers without requiring
pmtiles to be installed. Existing bundle CLI behavior still requires pmtiles.
C2 bounds round the valid raw bounds to four decimals. Coverage scans actual
root/leaf tile entries and projects tiles at z6 or above onto z6 cells; it never
uses bounds as coverage. Versioned filename hashes are trusted at this stage,
since C2 explicitly restricts reads to headers/directories; the versioning tool
is responsible for computing/verifying archive hashes. The Node validator
validates the checked-in JSON schema's supported keywords and fails closed if
an unsupported keyword is added. It also checks dates, sorting, unique coverage,
zoom order and smoke responses. JSON Schema is usable by other validators too.

**Manifest budget estimate: 193,883 bytes gzip for 3,761 eras.** See
[CHANGE_REQUEST_C2.md](CHANGE_REQUEST_C2.md). No format deviation was made.
The actual build prints its gzip size and fails above 80,000 bytes. The coverage
object is copied without editing any fields. Overlay members may be null.

Airfield array boundaries are individually aligned to four bytes, including odd
N. Details preserve every original property, normalize years, and include
`last_known_year`. Pair hashes cover binary plus details, so changing an airfield
name or URL cannot overwrite details referenced by an older manifest. The same
applies to all pin shard bytes and their indices. Shards expand ring bboxes by
margin degrees, split/wrap at the antimeridian, and contain original raw rings.
Locations without a ring are excluded just as today's pin query excludes them.
Per-chart pm paths use the supplied plan's actual headers; half-sheet lists get
a conservative union zoom/bbox. Wrapped or unknown header bounds get a world bbox
for `pmb`, which is safe for `fitBounds` but less precise than the ring.

Overviews require production WebP inputs. PMTiles has one tile type per archive;
retaining PNG bytes while labeling new WebP tiles would violate C1. Missing
children stay transparent. Existing lower levels also remain byte-identical.
Tiles at x=0 and x=2^z-1 have distinct parents: no longitude-bbox shortcut connects
the two sides of the world. Each parent decodes at most four children, mosaics
RGBA and uses Pillow LANCZOS before WebP quality 80. Stored compression and
metadata are preserved. New byte layout is immutable and always gets a new hash.

Measured synthetic overview case: 64 z8 tiles of 256 px, grid/line pattern,
635,131-byte source → 913,858-byte output; 26 new tiles; **278,727 bytes overhead
(43.9%) and 0.179 s** on this machine. This deliberately has no z9–11 payloads.
For a complete z8–11 pyramid the same added bytes are a much smaller fraction;
measure actual sources before budgeting. Dense rectangular new tile counts are
roughly N8*(1/4+1/16+1/64+1/256), with extra parents at sparse footprint edges.
Codec/disk and Pillow version can change timings, file size and thus hashes.

## Raster basemap

Headless Chromium + MapLibre GL is installed and tested here. It avoids native
MapLibre binding/ABI setup and uses the maintained Protomaps GL dark style with
English labels. It is a close style-family match to today's Protomaps Leaflet
renderer, not pixel-identical label placement. Assets and vector archive are
local; browser routing denies all external network requests. Fonts/sprites must
be mirrored from the official
[Protomaps assets](https://docs.protomaps.com/basemaps/maplibre). The renderer
uses a 128 px gutter, zero fade, flat north-up views and a 1x pixel ratio, then
crops 512 px and encodes WebP quality 80. Guttered independent label collision
layouts can still differ at tile boundaries; owner visual QA should include
cities, coastlines, adjacent tile seams and the antimeridian on the real cutout.
A missing asset, render error, idle timeout or missing WebP encoder fails the run.

A local synthetic earth/water/place source produced 8 tiles at z1–2 in **0.931 s
render time** (1.911 s including browser startup and packing), 9,250 encoded bytes,
and a deduplicated 5,505-byte archive. Every tile is 512x512 WebP. This validates
the pipeline and local font/sprite loading, not real-world complexity or full
production appearance. `make_basemap_proof.mjs` reproduces the source.

`python3 scripts/next_render_basemap.py --estimate` counts the full world at
z0–6 and exactly the union of `basemap_build.REGION_BOXES` at z7–13:
**4,559,534 tiles**. At assumed 22,000 bytes/tile and 0.15 s/tile: **100.31 GB**
and **189.98 serial hours**. At 10–40 KB/tile the pre-dedup size is 45.6–182.4 GB;
at 0.1–0.5 s/tile rendering is 126.7–633.3 hours. The actual 5.4 GB vector cutout
is not available here. Render a representative urban/rural/ocean sample and feed
measured averages to `--bytes-per-tile` and `--seconds-per-tile` before a full run.
Splitting non-overlapping zoom/bbox jobs across machines is possible, but each
job needs an input-specific directory and a final combined Hilbert packing pass.

The package versions used were Playwright 1.58.2, MapLibre GL 6.11.2, Protomaps
basemaps 5.7.2, PMTiles 3.0.6, and Pillow 12.1.0. Rendering needs MapLibre's v6
ES modules; no package manifests/locks in the existing checkout were changed.

## Configuration and verification

Owner deploy uses the existing BUCKET and optional TILES bindings and SAMPLE_RATE.
No extra route is needed because the data domain already reaches this Worker;
audit that fact in the owning account before deploy. No new environment variable
is required. The special C1 `*` CORS applies only to the reserved `/t/` namespace.
Review the account's Worker memory/CPU plan with real directory sizes before rollout.

Validated locally: old and new Worker tests, Python next tests, deterministic
fixture generation, schema validation and C1 HTTP smoke against the fixture
server, synthetic overview pixel checks, and local Chromium render/pack proof.
Node doubles do not emulate deployed Cache API/R2 cancellation or gzip transport;
the staging checks in MIGRATION.md are required before production cutover.
