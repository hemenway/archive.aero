# WebGL2 renderer — frozen contract v1

`index.js` exports `createRenderer` and `RendererUnsupportedError` with every C7 method. The runtime has no dependencies, fetches, workers, DOM controls, text rendering, Leaflet, or protomaps. Only the demo builds controls and synthetic labeled tiles. The production viewer is not modified.

## Run and verify

From the repository root:

```sh
node next/renderer/server.mjs
# Open http://127.0.0.1:4181/demo.html
npx playwright test --config next/renderer/playwright.config.mjs
node next/renderer/benchmark.mjs
```

Stop the demo server before running the test or benchmark command: both start their own server on port 4181. The demo needs only a local HTTP server for ES modules; it works without internet access. It makes no data requests. All tiles come from `OffscreenCanvas` / `createImageBitmap`, with a canvas fallback where OffscreenCanvas is unavailable. The era slider schedules independent random delays of 30–480 ms and paints overlapping archive items with distinct colors, including overzoom and clip rings. The two clipped archives are demanded only for tiles intersecting their footprints; the third covers the view.

Verified **44/44 Playwright cases**: 22 in Chromium, 22 in iPhone 13 mobile WebKit, no skips. Tests include every requested regression, randomly delayed overlapping era swaps sampled every animation frame, shader/CPU visibility agreement, bitmap closure, the four-upload cap, focused keys, pinch/double tap/inertia, no idle draws, CSS/DPR resizing, restoration of all overlay GPU buffers, repeated full-size demo scrubs within the default budget, and picking at the south Mercator edge. WebKit and Chromium both support WEBGL_lose_context on this host. Browser output goes to `/tmp/archive-renderer-test-results`, not the repository.

## Camera and demand

The renderer defaults to C7's **minZoom 4**, maxZoom 14, with a normalized Web Mercator centre. This intentionally uses the new overview floor rather than the old Leaflet floor of 6. The camera's x stays wrapped into [0, 1): float32 shader uniforms lose sub-pixel precision past the first world copy, and callers compare longitudes, not copies. Animations interpolate towards the nearest copy of their target, so a dateline crossing never goes the long way round. Public tile x coordinates are canonical; the renderer repeats world copies itself. `project` chooses the copy nearest the camera; `unproject` returns continuous longitude. Round trips across the dateline should compare longitude modulo 360. Latitude clamps to ±85.05112878°. `fitBounds` accepts `[west, south, east, north]`, handles a west>east antimeridian crossing, floors its fitted zoom, and fits at minZoom on a canvas narrower than its padding instead of throwing.

`visibleTiles(256)` uses round(camera.zoom); `visibleTiles(512)` uses max(0, round(camera.zoom)−1). The demand includes one tile of buffer on each side, reduced to half a tile when the shorter canvas CSS dimension is under 500 px. It is deduplicated across world copies and sorted by squared distance to the view centre. Only intersecting tiles draw. The renderer itself repeats world copies.

HiDPI currently keeps the same chart resolution as today's viewer: zoom is based on CSS pixels, while the backing store, circles and lines use devicePixelRatio. Going one chart level deeper at DPR≥2 would make raster charts sharper, but needs roughly four times as many source tiles and additional bandwidth/GPU storage. Do that only as a coordinated future contract change; 512 px basemap labels are already baked into the input raster.

Pointer capture supports drag, velocity-based inertia, pinch without rotation, and touch double tap. Wheel deltas normalize pixel/line/page modes, use 120 px per level and a 60 ms debounce, then animate to an integer over 220 ms. Double-click and +/− advance one level; arrows pan 80 CSS px. `flyTo` takes 1.5 seconds. Enter belongs to the shell. ResizeObserver and window resize update CSS size and DPR; redundant resize notifications skip reallocation.

## Atomic compositing and ownership

An uploaded bitmap becomes resident on a subsequent animation frame, so `hasTexture` is false while it is queued. Uploading transfers ownership immediately. The renderer closes the bitmap after `texSubImage3D`, and closes duplicate deliveries, queued bitmaps on destruction, and all pending bitmaps on context loss.

Each destination cell stores its last completed ordered item list and the plan sequence it completed at. A replacement becomes complete only when **all** items resolve to resident exact textures or resident lower-zoom ancestors from the same archive path. Then that cell switches lists in one frame. A later full-resolution upload sharpens the existing plan. Superseded deliveries can enter the cache but cannot revive an obsolete plan. Empty `items` explicitly completes and clears a destination; it is how a caller deliberately removes its charts.

Completed parents and child mosaics also backfill zoom transitions before the new destination plans arrive. When a new plan arrives, a cell inherits the **most recently completed** snapshot among its own and its ancestors', so a parent finished for a newer date supersedes the cell's own snapshot of an older one (zooming back in after a date change shows the new date, never the stale finer tiles). A snapshot stays drawable only while every item still resolves to a resident texture of its archive — exact or coarser ancestor; once an item has been evicted off-screen the snapshot is dropped rather than drawn partially. The old image is never cleared as part of staging. UV offsets and extents are computed from the resolved source coordinates and destination coordinates, independently for each item. Paint order is basemap, ordered chart items, airspace, airfields, pin.

Rings are projected once, unwrapped at the antimeridian, uploaded once, then drawn into the stencil buffer using a triangle fan with INVERT. This implements an even/odd mask for concave rings without triangulation. Opacity is a uniform. Hidden charts skip all chart draw calls. Crossfade is intentionally omitted: transitions are immediate atomic swaps (the optional crossfade default is 0).

Blending is premultiplied throughout: the data plane decodes bitmaps with `premultiplyAlpha: 'premultiply'` (WebGL ignores `UNPACK_PREMULTIPLY_ALPHA_WEBGL` for ImageBitmap sources), every shader emits premultiplied colour, and the blend function is `ONE, ONE_MINUS_SRC_ALPHA`, so LINEAR filtering across coverage edges and chart collars never pulls in the black of transparent texels.

## Texture pool and pressure

The default **physical texture-storage** budget is 64 MiB on desktop, 24 MiB on iOS/iPadOS, or on coarse-pointer devices with a shorter screen dimension ≤820 CSS px. The iPadOS heuristic includes a MacIntel platform with touch points. `maxTextureBytes` overrides it in bytes; values must be finite and fit at least one 256 px RGBA tile.

256 px charts use RGBA8 TEXTURE_2D_ARRAY pages; 512 px basemap tiles use separate pages. Pages grow lazily by at most eight layers, respecting the remaining byte budget and MAX_ARRAY_TEXTURE_LAYERS. No mipmaps are allocated. Stats include unused layers in allocated pages. One resident chart costs 256 KiB; one basemap tile costs 1 MiB. LRU eviction excludes every exact key referenced by the current chart/basemap plans, all currently drawn completed snapshots, and ancestors needed for pending swaps. Evictions emit `{key}`. Entire unpinned pages can be retired and repurposed across tile sizes.

Uploads stop after four images or about four milliseconds of frame CPU work, whichever comes first. Textures the current plans need jump the queue; superseded deliveries (dates already scrubbed past) only fill idle budget and are dropped — with an `evict` event so the data plane forgets them — under texture pressure or once more than 64 bitmaps are waiting, instead of holding decoded memory. One browser GL upload cannot be interrupted and may itself exceed that time cap. When all available storage is pinned by needed textures, the queue stops, old chart pixels remain visible, and the renderer emits an additional `texturepressure` event. There is no polling animation loop; a plan, camera, or upload change retries the queue. Demand must then shrink, use fewer native sources/ancestors, or be created with a larger configured budget. Breaking C7's pinning rule to finish an oversized plan would be incorrect.

For the measured 1440×900 view there are 63 buffered chart destinations and 30 basemap destinations. Three distinct native-resolution chart items per destination require **77.25 MiB of resident images** even before page slack, so that stress case is measured with a 128 MiB configured budget. Using three z8 source parents for the same z9 destinations needs 45 MiB of resident images and fits the default desktop budget. The data plane's lower-zoom-first delivery can provide this fallback. The renderer does not fabricate source tiles or request them itself.

The offline demo at 1440×900 and its initial zoom 6 uses 97 resident textures and 50 MiB of allocated texture pages with the default 64 MiB budget (69 draw calls). Clipped archive demand is limited to its footprints.

The GPU byte count excludes browser framebuffer/compositor allocations and drivers' internal overhead. At DPR 1, the 1440×900 RGBA color buffer alone is about 4.94 MiB, with additional stencil/depth storage; DPR 2 multiplies viewport storage by four. Queued ImageBitmaps live in CPU/driver decode storage outside the texture budget.

## Overlays and recovery

Airfields consume C4's `{mx,my,start,end,status}` arrays. A packed instance buffer and a 256×256 uniform pick grid are built only at `setAirfields`. The vertex shader and CPU picker implement the ordered C4 visibility rule, including open fields ignoring end year and undated fields remaining visible. Date and status changes update uniforms; no geometry buffer re-upload occurs. Sprites have the requested status fills, a 2 CSS px white stroke, zoom radii 4/5.5/7/8, and a +1 coarse-pointer radius. Picking adds stroke half-width and 4/14 px tolerance, wraps horizontally, and returns the closest visible index.

Airspace accepts C6 LineBatch typed arrays. CPU work expands polylines once at tile delivery into joined triangle geometry with previous/next positions and cumulative normalized-world distance. The vertex shader applies day/class/region masks and extrudes in CSS pixels (the viewport maps these to device pixels). Miter joins are capped at 2.5×. Casings, class widths, zoom factors, D/E dashes and colors match the existing values. E-floor codes 6/7 use the same `(dy,-dx)` side and three stepped opacity bands, peaking at 0.5, only from zoom 7. Each airspace tile scissors its buffered geometry to its bounds to avoid duplicate translucent seams. This approximates the existing round joins with capped miters; it does not render labels.

`webglcontextlost` cancels animation and uploads, clears residency, closes pending bitmaps and emits `contextlost`; every resident or queued key, and any upload attempted while the context is lost, is reported through `evict`, so the data plane's residency bookkeeping follows without a separate reset. Restoration recreates every program, VAO, ring/field/line buffer, empties completed chart snapshots, and emits `contextrestored`; the shell re-demands its current state. The renderer retains CPU overlay data for that recovery. `destroy` removes listeners/observers/timers and deletes its GPU resources.

`resize` draws synchronously after reallocating the backing store (which clears it), so a window resize, rotation or URL-bar collapse never paints an empty frame; a `(resolution: …dppx)` media query re-arms on each resize so a window dragged to a display with a different pixel ratio reallocates too.

There is no idle requestAnimationFrame loop. Stable camera/plan redraws use cached draw lists, numeric loops, stored uniform locations and a reused render-event object. Only plan/viewport changes rebuild draw lists; year/day/style updates have no renderer geometry allocations. The additional `render` event is diagnostic. `stats().frameMs` is CPU submission time, **not GPU completion time**.

## Measurements

Measured on 2026-10-02 (America/Chicago), Chromium 145 headless, ANGLE/Vulkan SwiftShader, DPR 1. Headless software GPU results are **indicative only**, especially the dense 8,000-airfield test; they do not establish hardware browser frame rates. Both scenes draw three chart items per visible destination, 96 polylines covering all eight styles, and 8,000 shader-filtered airfields. Each reports 120 measured redraws after 20 warmup frames. The synchronized time includes a forced 1×1 readPixels after submission, to wait for raster completion; it also includes readback overhead.

| Metric | Native chart sources | z8 parents overzoomed at z9 |
| --- | ---: | ---: |
| Configured budget | 128 MiB | 64 MiB |
| CPU submission median / p95 | 0.10 / 0.20 ms | 0.10 / 0.20 ms |
| Synchronized frame median / p95 | 79.90 / 127.50 ms | 82.00 / 121.90 ms |
| Resident textures | 219 | 90 |
| Resident image bytes | 77.25 MiB | 45.00 MiB |
| Allocated GPU texture pages | 80 MiB | 48 MiB |
| GPU geometry buffers | 256,768 B | 256,768 B |
| Draw calls per redraw | 122 | 122 |
| Upload count / elapsed | 219 / 6.48 s | 90 / 1.89 s |
| Upload throughput | 33.81 tiles/s · 11.93 MiB/s | 47.53 tiles/s · 23.76 MiB/s |

Bundled/minified: **29,705 bytes**; gzip −9: **10,328 bytes** (10.09 KiB), esbuild **0.28.2**.

Run `node next/renderer/benchmark.mjs` to overwrite the raw results in `measurements.json`. Image generation/decode happens before the upload measurement, and overlays are added after uploads, so upload throughput measures the budgeted upload queue plus raster tile drawing and frame pacing, not decoding or fetching.

The module size includes index.js, camera.js, shaders.js and texture-pool.js bundled as one ES module; it excludes the demo, tests and benchmark. The 20 KB gzip target is met. esbuild was obtained with npx in a scratch directory; no package dependency or lockfile change was added:

```sh
renderer_repo=$(pwd)
mkdir -p /tmp/archive-renderer-size
cd /tmp/archive-renderer-size
npx --yes --package esbuild@0.28.2 esbuild "$renderer_repo/next/renderer/index.js" \
  --bundle --minify --format=esm --target=es2022 --outfile=renderer.min.js
gzip -c -9 renderer.min.js > renderer.min.js.gz
wc -c renderer.min.js renderer.min.js.gz
```

The GL upload/context behavior follows the [Khronos WebGL2 specification](https://registry.khronos.org/webgl/specs/latest/2.0/).
