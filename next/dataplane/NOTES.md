# Data plane — frozen v1 C1–C7

Implemented on `redesign/data-plane` in an isolated worktree. Production viewer,
server endpoints, build scripts and root dependency files are untouched. Runtime
modules have no dependencies. Validation date: October 2, 2026 (America/Chicago).

## Run

From the repository root, with its existing npm development dependencies installed:

```sh
node --test next/dataplane/test/*.test.mjs
npx playwright test --config next/dataplane/playwright.config.mjs
node next/dataplane/bench/planning.mjs
node next/dataplane/bench/run.mjs
node next/dataplane/bench/concurrency.mjs
```

Playwright owns port 4182. The session and concurrency benchmarks bind ephemeral
ports and close their servers/browsers on exit. Standalone mock server:

```sh
LATENCY=80 MBPS=10 node next/dataplane/bench/server.mjs
LATENCY=120 MBPS=5 CONCURRENCY=8 node next/dataplane/bench/run.mjs
```

`bench/results.json` and `bench/concurrency-results.json` contain measured output,
not golden assertions. Timing varies with desktop load and GC. The dependencies
used here were Node v26.7.0 and Playwright 1.58.2 on macOS arm64.

## API and ownership

`index.js` exports `createDataPlane`. Planning is synchronous on the main thread;
network scheduling, decoding, encoded-cache ownership, geometry and queries run
in a module Worker. `worker: false`, missing Worker support, startup Worker failure,
or a supplied custom `fetch` selects the same core on the main thread. Functions
cannot be cloned into a Worker; an injected fetch deliberately uses this fallback.
A terminal failure after Worker startup emits an error and rejects later RPCs,
rather than leaving unresolved promises.

The public manifest is exactly C7: ISO frames/dateBounds, coverage, eraCount and
feature flags. DrawItems use C6 slash keys and only `key`, `dst`, `src`, and optional
`clip`. The shell installs solo rings in its renderer with the supplied clip id.
Neither planner wraps input coordinates; they are already wrapped by contract.
Tile-size selection (256 charts, 512 basemap) remains the shell/renderer's job.

Each `tile` listener owns its bitmap and passes it to renderer.upload, which
closes it. Subscribe one bitmap consumer. Worker bitmaps are transferred, never
copied or retained by the Worker. The encoded LRU retains its own original bytes;
encoded fallback messages transfer a separate copy. Decode dedupe lasts until
main-thread acknowledgement, so repeated demand cannot start duplicate decodes.
Generation changes preserve decodes still needed for basemap, ancestors, or
prefetch; superseded bitmaps close without delivery. Failed decode acknowledgements
release the key for a later retry.

Feature detection first tries createImageBitmap in the Worker. A missing API or
Worker codec failure transfers encoded bytes for main-thread createImageBitmap.
If unavailable there, an Image decodes via an object URL, then an OffscreenCanvas
produces a transferable bitmap if supported. The last fallback is the decoded
Image as a WebGL TexImageSource with an explicit close method; its object URL is
revoked in every path. `decodeBitmap: false` forces the fallback for testing.

C4 airfield arrays are zero-copy views over one binary buffer. Worker RPC responses
clone that single backing buffer before transferring views, preserving index/data
for subsequent calls and lazy details validation. Details load only when requested;
JSON promises are shared and rejected promises are removed. C5 loads only the
clicked z5 shard; missing shard ids do no network work. Query results preserve
per-chart pm/pmz/pmb and the legacy containment/distance ranking with six results.

Readiness counts **draw items**, including repeated source keys, rather than giving
every unique key equal weight. Only successful decode delivery counts. Eviction
immediately removes readiness; immutable 204 absence still counts as complete.
An empty plan is ready. The core retains no GPU bitmaps: shell texture evictions
must call markEvicted, then setDemand again. Context restoration must mark all
lost texture keys evicted before re-requesting.

Options in addition to C7 are `concurrency` (8), `cacheBytes`, `mobile`,
`decodeBitmap`, `decodeConcurrency` (4), `timeout` (20 s per fetch attempt) and
the shell's `earlyFetches` (a Map of absolute tile URL → Promise<Response>). The
encoded cache defaults to 32 MiB desktop / 12 MiB mobile; iOS/iPadOS/Android
detection supplies mobile automatically. Explicit options win. `stats()` is
synchronous; in Worker mode it uses the newest posted snapshot. It adds requests,
cancelled, bytes and retries to C7's three required counters. `off(event, fn)` and
the `metadata` event (airspace metadata loaded) are additive.

`earlyFetches` is stripped from the options posted to the Worker (Promises cannot
be cloned). Each URL under `tileBase` becomes a scheduler placeholder for its tile
key; the facade reads every response as it arrives and transfers status, content
type and bytes to the Worker (`adopt`/`prime`), and the first demand for that key
consumes the adopted response instead of fetching it again. A rejected early fetch
falls back to an ordinary fetch. Relative `tileBase`/`fileBase` values in a
fixture manifest are resolved against the manifest URL at load.

Decodes are bounded by `decodeConcurrency`: a burst of cache hits after texture
eviction queues rather than starting hundreds of `createImageBitmap` calls, and
queue entries whose keys left demand are skipped when dequeued. Bitmaps are
decoded with `premultiplyAlpha: 'premultiply'` for the renderer's blend mode.

## Planning and scheduling

Manifest validation rejects unsupported versions, invalid era keys/hashes,
invalid dates/intervals, bounds, zooms and z6 coverage indices. Starts/ends/zoom
ranges are parallel typed arrays, with bounds/coverage/path arrays. A separately
sorted start index and prefix maximum ends allow binary search plus end checks
without changing manifest paint order. Frames are sorted unique starts; maximum
coverage is latest exclusive end minus one day.

Every destination computes its geographic bounds once. Eras below minimum zoom,
outside bounds, or absent from z6 coverage contribute nothing. Null bounds and
null coverage never cull. Coverage below z6 checks all intersecting z6 cells;
overzoom uses the archive's ancestor. Per-destination source coordinates and key
suffixes are reused across contributing eras during that call, reducing dense
allocation cost without memoizing whole plans or reusing mutable output arrays.

The scheduler uses a binary min-heap ordered by priority, centre distance, then
stable insertion sequence. Demand is deduped by slash key; shared source tiles
fetch once. Each job owns an AbortController. Replacing demand aborts obsolete
running jobs, drops obsolete queued jobs and cancels obsolete backoff timers.
Active demand reprioritizes queued jobs without replacing an ongoing shared fetch.

New archives receive ancestors at max(minz, destination zoom minus 3), clamped to
native max, before current full resolution. Repeated demand retains unfinished
ancestors at current priority instead of aborting bootstrap. Scrub lookahead grows
from 2 to 8 frames with velocity. Idle demand requests ±1 full resolution and ±2/±3
as ancestors. Playback requests the next three frames at full resolution. Solo
replaces the era set and suppresses temporal prefetch. Basemap and airspace current
tiles join current priority. Prefetched successful results are delivered and can
buffer playback; renderer texture ownership/eviction limits decoded memory.

204 absence is remembered for the session separately from the byte LRU. Only
fully read successful bodies enter the LRU. Network failures, timeouts, 408, 429
and 5xx retry at 150/300 ms — or after `Retry-After`, capped at 5 s — at most
twice, each attempt with a fresh AbortController; ordinary 4xx fail immediately. Rejected fetch, manifest,
inventory and auxiliary JSON promises never enter a permanent success cache.
Failed keys can be requested again on later demand. Content-versioned `/t/` paths
and browser immutable caching replace legacy byte-range/ETag machinery.

## Legacy failure modes reviewed

Read Utils, MetaBundle/BundleSource, _renderTile, applyLoadedRanges, ChartIndex,
prefetchDates/showFrame, AirspaceLayer, AirfieldsLayer and airspace_build.py.
The old code guards truncated/corrupt bundle prefixes, wrong magic/version,
ignored Range, short group reads, unavailable gzip APIs, metadata group memory
pressure, 416/ETag changes, cached rejected promises, duplicate overzoom reads,
canvas allocation failure, superseded rendering and object-URL leaks. The new
endpoint removes archive-directory/bundle offsets and Range from the client;
immutable content hashes prevent same-path archive replacement. Explicit success
caches, abortable demand, decoding acknowledgement, bounded bytes and resource
closure address the remaining corresponding client hazards. No CSV/bundle fallback
is mixed into C2: an invalid manifest rejects initialization.

## iOS investigation

`git log -S 'useImageBitmap: !DeviceInfo.isIOS' -- src/viewer.js` and blame trace the
current line to module extraction commit `5843587` (September 21, 2026). Searching
index.html finds `1916dff` (July 1, 2026, "ios fix"), which expanded the earlier
Mobile Safari exclusion to every iOS browser. That commit also explicitly freed
canvas backing stores on tile unload, tolerated null canvas contexts, closed
superseded object URLs, reduced iOS caches and added a zoom floor. Its comment says
roughly 1,200 low-zoom canvases / 320 MB caused iOS jetsam termination.

The original Mobile Safari exclusion appeared in `05384b4` (February 13, 2026),
alongside canvas/overzoom work and smaller Mobile Safari archive caches. The commit
message concerns docs and contains no specific WebKit bug id, failed codec, or
measured bitmap leak. Therefore history establishes an iOS memory-pressure
mitigation context, **not** a proved createImageBitmap-specific defect.

Playwright's iPhone 13 WebKit profile exposes createImageBitmap and OffscreenCanvas
on both main thread and Worker. Real bitmap decode/transfer/draw and forced encoded
fallback pass in it. Feature detection replaces the blanket UA bitmap ban, while
mobile byte limits and explicit bitmap close remain. Desktop-hosted mobile WebKit
is not physical-device jetsam coverage; sustained real iPhone memory-pressure
validation remains an integration check.

## Airspace

The dependency-free MVT reader supports protobuf scalar values, packed/unpacked
fields, signed geometry deltas, buffered coordinates, polygon closure and unknown
fields. Invalid/truncated data rejects. HTTP gzip is handled by fetch; no manual
decompression is attempted.

`class` and `efloor` produce exact C6 normalized-Mercator LineBatch buffers with
styles 0–7, region codes 0–2, Days intervals and the current sentinel 2147483647.
Class E5/E6/E7 polygon outlines are excluded; efloor vertex direction is preserved
for the controlled-side ribbon. Renderer applies day/class/region masks and the
z7 ribbon floor. Separate integer rings stay in the Worker after batch transfer,
including exclusion polygon boundaries. Removed demand releases retained rings
and emits `{tileId, batch: null}` to remove the renderer batch (additive event
behavior matching renderer.setAirspaceTile's existing nullable argument).

Metadata comes from `/t/{path}/metadata`, loaded lazily on the first demand that
carries airspace tiles (or the first query), never on boot; a failed load retries
with exponential backoff (1 s doubling to 60 s) rather than on every camera move,
and `airspaceStatus` reports the failure even when the caller passes
`enabled: true`. Each region chooses its newest held cycle on/before the day, valid
for cycle_days (normally 28) exclusively. Gaps and dates
past the newest held cycle produce no region bit. Queries retain all altitude,
hours and descriptive properties, exclude notches, dedupe version hits, take the
lowest governing floor per badge and order floors, BADGE_ORDER and REGION_ORDER.
They query currently demanded decoded tiles, as renderStack did; there is no hidden
query-only geometry fetch. Additive `airspaceStatus(day, bounds, options)` returns
panel text lines for availability, held cycle, series gaps and expiry.

## Verification and mocks

46 Node tests pass: interval/order/bounds, planning/clamping/coverage/solo/wrap,
heap priority/centre order, cancellation/dedupe/retries/failure recovery, byte LRU,
C4 views/visibility, lazy selected-shard retry, legacy ChartIndex VM parity,
hand-built MVT, cycle/status rules, polygon holes/antimeridian, fallback ownership,
readiness/eviction, terminal Worker failure, and vector batch removal.

12 Playwright tests pass across Chromium and webkit-mobile: Worker bitmap decode
and transfer, forced Worker encoded fallback, main-thread fallback, stale request
cancellation observed by the real mock server, GPU eviction/re-request, repeated
C4 transfers with one backing buffer, lazy details, and region metadata.

The local C1/C2 mock implements GET/HEAD tile statuses, coordinate/path validation,
204 outside zoom range, metadata, immutable success headers and gzip MVT. It serves
synthetic C2 archives plus C4 and C5 fixtures within next/dataplane. A shared bandwidth
budget covers all simultaneous responses. The PNG is deliberately a noisy 256 px,
~90 KB stress tile encoded by the contract fixtures' dependency-free `png()`;
measurements are not forecasts for production WebP sizes.
The mock uses catalogued synthetic paths, not physical PMTiles archives. Agent 1's
canonical fixtures should replace this local catalog after merge; no canonical
fixtures were available in this isolated branch.

## Benchmark

The session uses 24 destination tiles at z10, 31 unique four-month frame starts
across 1950–1960, eight spatial era regions per frame, concurrency 8, 80 ms response
latency and one shared 10 Mbps bandwidth cap. It waits for complete plans at 30
scrub steps, also issues an eight-demand rapid reversal burst, then replays four
previously visited frames at 2 s intervals. Playback readiness is therefore a
buffered revisit measurement; it is not evidence of cold playback throughput.
A listener closes delivered bitmaps as a renderer would; GPU uploads are outside
this data-plane benchmark. Init/manifest time is excluded from first-plan timing.

Measured final run:

| Metric | Result |
| --- | ---: |
| First complete current chart plan | 3995.9 ms |
| Scrub step completion p50 | 3475.8 ms |
| Scrub step completion p95 | 3835.5 ms |
| Rapid cached reversal burst (8 demands) | 98.2 ms |
| Buffered playback frame readiness | 0.8–2.3 ms |
| Server tile-body bytes transferred, including cancelled bodies | 121,349,516 B |
| Fully read bytes recorded by scheduler | 121,075,780 B |
| Server-observed tile requests / cancellations | 1,218 / 7 |
| Scheduler request starts / cancellations | 1,219 / 8 |
| Encoded bytes resident at end | 33,493,300 B (below 32 MiB) |
| Retry count / data errors | 0 / 0 |
| Spatial planning: 300 tiles × 80 eras, p50 / p95 | 0.123 / 0.173 ms |
| Dense planning: same workload, 24,000 items, p50 / p95 | 0.268 / 0.574 ms |

Both planning workloads meet the 1 ms budget at p95 in this run. The dense case
retains the prior plan while generating the next, so its allocation/GC pressure
is included. Every iteration constructs fresh output arrays/items; timings do not
rely on cached identical-date plans. Server and scheduler cancellation counts may
differ because a request can abort before reaching the server. Wire totals exclude
HTTP headers and manifest/auxiliary JSON, and measure tile response bodies.


Concurrency first-plan samples (same latency/bandwidth/tile workload): cap 4 =
4,104 ms; cap 8 = 3,923 ms; cap 12 = 3,966 ms. One sample each is not a statistical
ranking; it supports retaining the proposed default of 8 without increasing caps.
