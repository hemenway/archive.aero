# Shell, boot and integration — Agent 4

Branch: `redesign/shell`. Work was isolated from the renderer checkout and its
existing edits in a managed worktree. The frozen shared contract is copied
verbatim into `next/CONTRACT.md`; byte comparison against the provided attachment
passes. The current viewer, deployment workflows, `site-files.json` and `CNAME`
are untouched. No deployment was performed.

## Implemented

The shell imports renderer/data-plane implementations through esbuild aliases.
`--stubs` selects both fixture modules. With the real modules missing the default
build falls back to stubs; otherwise supply the real immutable manifest URL.
C7 signatures and data shapes are used directly, with the additive early-fetch
option below. The 720-byte store batches shallow state notifications into one
microtask. URL writes are coalesced and spaced at least 250 ms apart.

The page is the production page. `build.mjs` reads the repository-root
`index.html`, drops what only the Leaflet viewer needs (the three vendor
scripts, the viewer preload and entry script, the Leaflet and `styles.css`
links, the Google Fonts links, the Plausible block, comments) and injects its own
head and app scripts. `#map` gains `tabindex="0"`, the `#mapCanvas` canvas and a
Leaflet-shaped control container. Production's `initApp` moves `#toolsControl`
and `#utilRail` beside Leaflet's zoom control at runtime; here that structure is
built at build time, with Leaflet 1.9.4's zoom markup (`#zoomInBtn`,
`#zoomOutBtn`) and an empty `#mapAttribution`. The rest of the body is the
production markup, unchanged. There is no `#fatalError`: the splash's
`#loadingStatus` is the error surface, as in production. Every edit asserts how
many places it touched, and the markup that remains must be static (no script,
inline handler or foreign stylesheet), so a change to `index.html` that the build
cannot place stops it. The minifier collapses whitespace but keeps one space
wherever inline elements or text meet, and checks its output against its input
token by token.

The stylesheet is the root `styles.css`, unchanged, followed by `shell/next.css`
and generated `@font-face` rules. `next.css` holds only what `styles.css` took
for granted from Leaflet (the control, zoom bar, attribution and tooltip base
rules of Leaflet 1.9.4, and what Leaflet's runtime gave the map container:
position, overflow, the typography the Layers panel inherits, the grab cursor
and touch-action) and the canvas rules. With `--links-origin`, the one selector
in `styles.css` that matches a site link by `href` is rewritten with the links.
Barlow, production's display face, is self-hosted: the latin 500, 600 and 700
woff2 files from `@fontsource/barlow` are copied to `fonts/` under content-hashed
names with immutable caching, so the page makes no third-party font request (the
CSP has `font-src 'self'`). The service worker does not precache them.

The UI restores a shared camera before demand, uses zoom 10
when coordinates are supplied without zoom, clamps dates, fits the lower 48 by
default and never geolocates automatically. Preference access is guarded.
Keyboard ownership, inert background, focus trapping/restoration, Escape layering,
map-center inspection, pointer cancellation and playback live-region silence are
covered by browser tests.

Camera/date/visibility changes update demand and plans. Opacity and status/class
filters become renderer uniforms without replanning. Tile events transfer bitmap
ownership to the renderer; eviction clears data-plane residency. Absent items are
removed from renderer plans so 204 responses do not hold a swap indefinitely.
Only a dead data Worker is fatal: a tile that fails (404, decode error, an
upload the renderer rejects) leaves the plans for 30 s so the rest of its tile
still swaps in, is logged, and surfaces one toast per 10 s; airspace metadata
failures show in the layer status. Context restoration re-demands the current
state after the renderer reports every lost texture evicted. The page survives
the back/forward cache (teardown skips persisted `pagehide`). Worker stats
are sampled during demand because C7 does not expose a loading-change event.
Playback waits for current-plan readiness, advances every two seconds and wraps.

The executable head script runs before the external stylesheet. It reads a
non-executable JSON block containing newest-era paths, bounds and zoom ranges,
computes the first viewport's tile URLs, and launches high-priority fetches before
the app initializes. The viewport is the whole window: as in production, the
header lies over the map rather than above it. Historical share dates outside that newest frame skip
speculative chart requests, but still warm the basemap. The build preloads the
hashed manifest and module graph, bundles a separate hashed Worker, generates CSP
script hashes, and commits all outputs and sourcemaps.

## Additive option for Agent 3: earlyFetches

```js
createDataPlane({ manifestUrl, earlyFetches })
// earlyFetches: Map<string /* absolute tile URL */, Promise<Response>>
```

The head script sets `window.__earlyFetches`. The shell passes the exact Map to
`createDataPlane`. When a demanded tile URL exists in it, consume that Promise
instead of issuing a second fetch, remove the entry, then process its response
with the ordinary 200/204/error rules. Use the normal decode/cache/event path;
bitmap ownership and C7 APIs do not change. No Range requests are introduced.
Rejections have an early catch attached to avoid an unhandled rejection before
the data plane adopts a Promise. The stub supports this option and boot tests
assert no duplicate tile URL requests.

**Promises are not structured-cloneable**, so the data plane strips `earlyFetches`
from the options it posts to its Worker and adopts the Map itself: each URL under
`tileBase` becomes a scheduler placeholder for its tile key, the main thread reads
the response (status, content type, bytes) as it arrives and transfers it to the
Worker, and the first demand for that key consumes it instead of fetching again.
A rejected early fetch falls back to an ordinary fetch. On the main-thread path the
promises are handed to the scheduler directly. Arbitrary keys/paths must never be
interpreted as source HTML.

The builder rewrites the data plane's literal
`new URL('./worker.js', import.meta.url)` to the hashed Worker filename; confirm
that reference after merge. Real modules and canonical fixtures have not been
merged into this isolated shell branch.

## Budgets and measurements

Measurements are for the committed stub build, not production acceptance
(2026-10-03, the first build of the production-derived page; the JavaScript rows
move while the app rewrite lands):

| Item | Actual bytes | Limit bytes |
| --- | ---: | ---: |
| Critical JS, gzip (the app entry's static chunks + Worker + executable inline scripts) | 17,368 | 51,200 |
| Lazy JS, gzip (chunks reached only through `import()`; informational) | 5,505 | — |
| CSS, gzip | 7,377 | 12,288 |
| CSS, uncompressed (informational) | 35,241 | — |
| HTML, uncompressed | 19,005 | 24,576 |
| HTML, gzip (informational) | 5,768 | — |
| Worker, gzip (included above) | 345 | — |
| Complete executable boot script, uncompressed | 1,366 | 1,536 |
| Fonts, three woff2 files (no budget) | 67,568 | — |

Metadata is in a separate `application/json` script, not executable JavaScript.
`dist/budgets.json` records the reproducible measurements. HTML uses the stricter
uncompressed interpretation of its budget. About 16.6 KB of it is the production
markup after minification (`index.html` is 22,955 bytes); the inline newest-frame
metadata grows by roughly 100 bytes per overlapping era, so the 24 KiB limit
leaves room for both (gzip stays under 6 KB). The stylesheet is production's,
backdrop filters included: 6.5 KB of the gzip figure is `styles.css`, the rest
`next.css` and the font faces. Only the app entry's static import closure is
preloaded and counted as critical JS; a chunk loaded through `import()` is
reported as `lazyJS`, cached and precached like the others, and has no limit.
Source maps and fonts are excluded from transfer budgets, preload hints and
precaching.

A separate visual/boot check at 1280×800 (Chromium) observed 19 startup requests,
13 tile URLs and 18 completed resource entries by first chart paint. iPhone 13
WebKit observed 18 startup requests, 11 tile URLs and 17 completed entries by
paint. Both had zero duplicate tile URL requests. Tests enforce a 50-resource
startup ceiling and reject unexpected requests across the browser context.
Desktop and mobile layer/timeline layouts were visually inspected. Those counts
predate the production-derived page, which adds two same-origin Barlow requests
(600 and 700) at first paint.

## Validation

- `npm run test:next`: 54 tests across Chromium, desktop WebKit and mobile WebKit.
- `node next/build.mjs --check`: all committed outputs match; all budgets pass.
- `cmp` against the shared-contract attachment: exact match.
- `npm test` and the production frontend sources remain unchanged.

Coverage includes dates and slider keys, playback wrap/pause/readiness, exact
100-ms loader grace, URL restore/throttle, pin/solo/share, default view, storage
failure, scoped keys, skip link, focus trap and restore, airfield keyboard browser,
filters without new tile fetches, explicit GPS/IP Locate, manifest/renderer/Worker
errors, warning timing, early-request adoption and service-worker policy.

The service-worker test checks shell/manifest/Worker precaching, stale-cache
cleanup and tile exclusion in all three browsers. Chromium also reloads with
Playwright's network offline mode. WebKit's offline reload reports an internal
Playwright engine error, so its navigation fallback is exercised with an injected
HTTP 503 from the fixture server instead. Real-device offline and GPU-memory
checks remain on the integration checklist.

## Integration limitations

The fixture renderer approximates tiles and airfields with a 2D canvas and draws
a synthetic airspace line batch. It does not claim WebGL swap/fallback performance,
clip-ring fidelity or mobile GPU-memory acceptance. The stub data plane implements
C7 over a small in-memory C2 manifest and synthetic tile bytes. Its fixture fields
and pins must be replaced by Agent 1's canonical binary/shard fixtures in the
merged end-to-end suite. True production transfer sizes, real tile decode/cache
memory, metadata status, texture pressure and interrupted WebKit playback need
the renderer/data-plane integration pass described in `next/README.md`.

Hosting is documented in `next/HOSTING.md`, including immutable headers, the
same-origin data route, CSP, preview/switch and rollback. Generated preview
headers target `/next/`; retarget them for the root switch and preserve previous
hashed assets during the rollout window. Before `/` becomes the new viewer,
change the unsupported-renderer fallback to the retained legacy viewer's URL.
