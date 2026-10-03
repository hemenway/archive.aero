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

Static chrome includes the collections/menu, layers, airfield browser, utility
rail, heat strip, date input, cards, solo bar, loading pill, warning, toast and
shortcuts dialog. The UI restores a shared camera before demand, uses zoom 10
when coordinates are supplied without zoom, clamps dates, fits the lower 48 by
default and never geolocates automatically. Preference access is guarded.
Keyboard ownership, inert background, focus trapping/restoration, Escape layering,
map-center inspection, pointer cancellation and playback live-region silence are
covered by browser tests. System fonts replace the production Google Font.

Camera/date/visibility changes update demand and plans. Opacity and status/class
filters become renderer uniforms without replanning. Tile events transfer bitmap
ownership to the renderer; eviction clears data-plane residency. Absent items are
removed from renderer plans so 204 responses do not hold a swap indefinitely.
Context restoration re-demands the current chart and basemap keys. Worker stats
are sampled during demand because C7 does not expose a loading-change event.
Playback waits for current-plan readiness, advances every two seconds and wraps.

The executable head script runs before the external stylesheet. It reads a
non-executable JSON block containing newest-era paths, bounds and zoom ranges,
computes the first viewport's tile URLs, and launches high-priority fetches before
the app initializes. Historical share dates outside that newest frame skip
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

**Promises are not structured-cloneable.** Agent 3 must take `earlyFetches` out of
options before posting initialization to the Worker. Adopt responses on the main
thread, then prime the Worker cache using transferred bytes and response metadata,
or deliver them through its existing main-thread decode bridge. Passing the Map
through the current generic options clone would fail and trigger its fallback;
forcing all work onto the main thread would lose the intended Worker path. This
is the remaining explicit cross-branch integration requirement, not a change to
the frozen contract. Arbitrary keys/paths must never be interpreted as source HTML.

The builder rewrites the data plane's literal
`new URL('./worker.js', import.meta.url)` to the hashed Worker filename; confirm
that reference after merge. Real modules and canonical fixtures have not been
merged into this isolated shell branch.

## Budgets and measurements

Measurements are for the committed stub build, not production acceptance:

| Item | Actual bytes | Limit bytes |
| --- | ---: | ---: |
| Critical JS, gzip (app/chunks + Worker + executable inline scripts) | 13,600 | 51,200 |
| CSS, gzip | 2,224 | 12,288 |
| HTML, uncompressed | 10,220 | 10,240 |
| HTML, gzip (informational) | 3,902 | — |
| Worker, gzip (included above) | 345 | — |
| Complete executable boot script, uncompressed | 1,369 | 1,536 |

Metadata is in a separate `application/json` script, not executable JavaScript.
`dist/budgets.json` records the reproducible measurements. HTML uses the stricter
uncompressed interpretation of its budget. The HTML budget has little spare room;
production newest-frame metadata must pass the same check. CSS contains no
backdrop filter. Source maps are excluded from transfer budgets, preload hints
and precaching.

A separate visual/boot check at 1280×800 (Chromium) observed 19 startup requests,
13 tile URLs and 18 completed resource entries by first chart paint. iPhone 13
WebKit observed 18 startup requests, 11 tile URLs and 17 completed entries by
paint. Both had zero duplicate tile URL requests. Tests enforce a 50-resource
startup ceiling and reject unexpected requests across the browser context.
Desktop and mobile layer/timeline layouts were visually inspected.

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
