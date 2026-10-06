## The shared contract (frozen, v1)

All four sections code to this. It is reproduced word for word in every agent's prompt, and the shell agent commits it as `next/CONTRACT.md`.

### C1. Archive paths and the tile endpoint

- Every published archive has a content-versioned **path** of the form `{dir}/{name}.{hash}`. Its R2 key is the path plus `.pmtiles`. `hash` is the first 12 hex characters of the SHA-256 of the archive's bytes. Examples:
  - era archives: `sectionals/{eraKey}.{hash}`
  - per-chart artifacts: `sectionals/chart/{slug}/{date}.{hash}`
  - basemap: `basemap/{name}.{hash}`
  - airspace: `airspace/{name}.{hash}`
- `GET` or `HEAD` `https://data.archive.aero/t/{path}/{z}/{x}/{y}` (no file extension):
  - **200** with the tile's bytes. `Content-Type` comes from the archive header's tile type: webp → `image/webp`, png → `image/png`, jpg → `image/jpeg`, mvt → `application/vnd.mapbox-vector-tile`. If the header says tiles are gzip-compressed, the stored bytes are sent with `Content-Encoding: gzip`.
  - **204** when the archive exists but has no such tile, including any z outside the archive's zoom range.
  - **404** when the path is unknown. **400** for malformed coordinates: z outside 0–24, or x or y outside 0 to 2^z − 1.
  - 200 and 204 responses carry `Cache-Control: public, max-age=31536000, immutable`. 404 responses carry `Cache-Control: public, max-age=60`.
  - `Access-Control-Allow-Origin: *`. Clients never send `Range` to `/t/`.
- `GET https://data.archive.aero/t/{path}/metadata` returns the archive's JSON metadata: 200, `application/json`, same immutable caching.
- Every existing URL and the current range-proxy behavior stay exactly as they are, because the production viewer depends on them.

### C2. Manifest

- Lives at `https://data.archive.aero/next/manifest.{hash}.json`, immutable. The page gets this URL at build time, the way `bundleUrl` works today.

```json
{
  "version": 1,
  "generated": "2026-10-03T00:00:00Z",
  "tileBase": "https://data.archive.aero/t/",
  "fileBase": "https://data.archive.aero/",
  "eras": [
    { "k": "1950-01-01_to_1950-07-01", "h": "0123456789ab", "b": [-98.0, 31.0, -95.0, 34.0], "z": [4, 11], "c": [1234, 1235] }
  ],
  "basemap": { "p": "basemap/raster-20261003.0123456789ab", "z": [0, 13], "tileSize": 512 },
  "airspace": { "p": "airspace/class-20260916b.0123456789ab", "z": [0, 11] },
  "airfields": { "bin": "next/airfields.0123456789ab.bin", "details": "next/airfields.0123456789ab.json" },
  "pins": { "z": 5, "margin": 2, "base": "next/pins.0123456789ab/", "shards": [312, 313] },
  "coverage": { "...": "today's coverage.json content, verbatim" }
}
```

- **eras**, sorted by start date, then end date, then `k`:
  - `k` keeps today's era-key format, `{start}_to_{end}`, with the end date exclusive. Start-only keys are never emitted.
  - The era's path is `sectionals/{k}.{h}`.
  - `b` is the lon/lat bounds `[w, s, e, n]` rounded to 4 decimal places, or `null`, meaning never cull this era.
  - `z` is the `[min, max]` zoom actually present in the archive.
  - `c` lists, in ascending order, the z6 tiles (as indices `y * 64 + x`) in which the archive has at least one tile. `null` means "assume coverage wherever `b` overlaps."
- `basemap` may carry `"format": "mvt"` with a Protomaps `flavor` and label `lang` (added 2026-10-05): the archive then holds production's vector tiles, each painted by the data plane into the same 512 px tile a raster basemap would supply. Without `format` the tiles are raster images.
- `basemap`, `airspace`, `airfields` and `pins` may each be `null`, meaning that feature is off. Paths in `p` resolve against `tileBase`. File paths (`bin`, `details`, `base`) resolve against `fileBase`.
- **Rules every consumer applies the same way:**
  - An era is in effect on date D when start ≤ D < end. ISO date strings compare correctly as plain strings.
  - Paint order is manifest order: later eras draw on top.
  - Timeline frames are the sorted unique start dates.
  - The date bounds are min = the earliest start, and max = the latest end minus one day.
- Budget: the production manifest is at most 80 KB gzipped.

### C3. Overviews

Every era archive gains z4–z7 levels, built by downsampling its own z8 tiles; the existing tiles stay byte-identical. After the migration, every era's `z[0]` is 4. Clients must still honor whatever `z` says, because fixtures and not-yet-migrated archives may start at z8: a destination tile whose z is below an era's minimum gets no item from that era.

### C4. Airfields binary

`airfields.bin` is little-endian:
- bytes 0–7: the ASCII magic `AAAF1\0\0\0`
- bytes 8–11: uint32 count N
- bytes 12–15: uint32 0 (reserved)
- then these arrays, in order, each starting at a 4-byte-aligned offset (zero padding between them):
  - `Float32[N] mx`, `Float32[N] my`: Web Mercator coordinates in [0, 1], with y = 0 at the north edge
  - `Uint16[N] start`: first year, 0 = unknown
  - `Uint16[N] end`: last year, 0 = none or unknown
  - `Uint8[N] status`: 0 = open today, 1 = gone, 2 = unknown

`details` is a JSON array of N objects, index-aligned with the binary. Each object holds every field that today's airfield card and airfield browser display (see `AirfieldsLayer` in `src/viewer.js`: `openCard`, `fmtSpan`, `_updateBrowser`), including `last_known_year`.

Visibility at year Y with the status filter S, exactly as today's `AirfieldsLayer._visible` decides it, checked in this order:
1. Hidden if the status isn't in S.
2. Otherwise visible if Y is unknown.
3. Otherwise visible if start = 0 and end = 0.
4. Otherwise hidden if start is set and Y < start.
5. Otherwise visible if the status is open.
6. Otherwise hidden if end is set and Y > end.
7. Otherwise visible.

### C5. Pin shards

- Each shard is a file at `{fileBase}{pins.base}{i}.json`, where i = `y * 32 + x` of a z5 tile (`pins.z`). It contains every location whose ring bounding box, expanded by `margin` degrees, intersects that z5 tile. Handle the antimeridian the way `ChartIndex._build` does.
- A shard has the same shape as today's `timeline_data.json` (`{ "locations": { name: { era, ref, charts: [...] } }, "rings": { ref: [...] } }`), so `ChartIndex`'s logic ports unchanged. The one difference: each chart's `pm` (its per-chart artifact) is a versioned path or a list of paths, accompanied by `pmz: [minz, maxz]` and `pmb: [w, s, e, n]`.
- The client loads the shard containing the clicked point (after normalizing its longitude) and ranks results the way `ChartIndex.query` does. A shard index missing from `pins.shards` means there are no charts there.

### C6. Shared types

```
TileCoord = { z, x, y }                 // integers; x already wrapped into [0, 2^z)
TileKey   = `${path}/${z}/${x}/${y}`    // source coordinates; fetch URL = tileBase + TileKey
DrawItem  = { key: TileKey, dst: TileCoord, src: TileCoord, clip?: string }
            // src.z <= dst.z, and src is dst itself or one of its ancestors
TilePlan  = { dst: TileCoord, items: DrawItem[] }   // items ordered bottom to top
ChartPlan = { id: string, tiles: TilePlan[] }
Days      = integer days since 1970-01-01 UTC
LineBatch = {
  positions: Float32Array,  // Web Mercator x,y pairs, vertex order as stored in the MVT
  starts:    Uint32Array,   // index of the first vertex of polyline i; length = lines + 1
  from:      Int32Array,    // per polyline, Days; in effect from this day
  to:        Int32Array,    // per polyline, Days, exclusive; 2147483647 = still current
  style:     Uint8Array,    // per polyline, airspace style code (below)
  rg:        Uint8Array     // per polyline region: 0 = us, 1 = fr, 2 = br
}
```

Airspace style codes, matching `AirspaceLayer._paintRules`:
- From the `class` layer: 0 = A, 1 = B, 2 = C, 3 = D, 4 = E (excluding the E5/E6/E7 floor types), 5 = unclassed.
- From the `efloor` layer: 6 = 700 ft E-floor ribbon (magenta), 7 = any other E-floor ribbon (blue). Ribbons draw only at zoom 7 and above, on the side `RibbonSymbolizer` uses.

Airspace filtering:
- `classMask` bit 0 turns on A–D plus unclassed (codes 0–3 and 5). Bit 1 turns on E (codes 4, 6 and 7).
- `regionMask` has one bit per `rg` value.
- A polyline is visible when from ≤ day < to and both masks pass.

### C7. Module APIs (ES modules, no dependencies)

**Renderer**, `next/renderer/index.js`:

```js
export class RendererUnsupportedError extends Error {}
export function createRenderer(canvas, { maxTextureBytes, minZoom = 4, maxZoom = 14 } = {}) // throws RendererUnsupportedError without WebGL2
r.getCamera()                                   // → { x, y, zoom }  Web Mercator centre, float zoom
r.setCamera({ x, y, zoom }, { animate = false } = {})
r.fitBounds([w, s, e, n], { padding = 0, maxZoom } = {})
r.flyTo({ lng, lat }, zoom)
r.project({ lng, lat })                         // → { x, y } CSS px relative to the canvas
r.unproject({ x, y })                           // → { lng, lat }
r.visibleTiles(tileSize)                        // 256 | 512 → TileCoord[] the renderer will draw at this moment, centre first
r.hasTexture(key)                               // → boolean
r.upload(key, bitmap)                           // ImageBitmap; the renderer closes it after uploading
r.setChartPlan(plan)                            // ChartPlan
r.setBasemapPlan(tilePlans)                     // TilePlan[]
r.setChartStyle({ opacity = 1, hidden = false })
r.setClipRing(id, ring)                         // ring: [[lng, lat], ...] or null to remove
r.setAirfields(arrays)                          // { mx, my, start, end, status } typed arrays (C4), or null
r.setAirfieldFilter({ year, statusMask })       // year null = show all; statusMask bit0 open, bit1 gone, bit2 unknown
r.setAirspaceTile(tileId, batch)                // tileId "z/x/y"; LineBatch or null to remove
r.setAirspaceFilter({ day, classMask, regionMask })
r.setPin(lngLat)                                // { lng, lat } or null
r.pick(clientX, clientY)                        // → { kind: 'airfield', index } or null
r.on(event, fn); r.off(event, fn)
  // 'move' (each frame while the camera changes), 'moveend',
  // 'click' { lng, lat, clientX, clientY, picked }, 'evict' { key },
  // 'contextlost', 'contextrestored' (every texture is gone; the app re-requests them)
r.stats()                                       // → { textures, textureBytes, drawCalls, frameMs }
r.resize(); r.destroy()
```

Swap rules:
- For each destination tile, the renderer keeps drawing the last plan it completed for that tile until every item of the new plan is drawable, then swaps them all in a single frame.
- An item is drawable if its own texture is resident, or if a resident texture from the same archive path covers it at a lower zoom. The lower-zoom one is drawn upscaled, then sharpened when the real texture arrives.
- The basemap never shows through a tile in the middle of a swap.
- A texture referenced by the current plan, or by the last completed plan still on screen, is never evicted.

**Data plane**, `next/dataplane/index.js`:

```js
export async function createDataPlane({ manifestUrl, fetch, worker = true } = {})  // resolves once the manifest is loaded
dp.manifest            // { frames: string[], dateBounds: { min, max }, coverage, eraCount, hasAirspace, hasAirfields, hasPins }
dp.planCharts(date, tiles, { solo } = {})    // → ChartPlan; synchronous and pure
  // solo: { paths: string[], zoom: [minz, maxz], clip?: { id, ring } } replaces the era set
dp.planBasemap(tiles)                         // → TilePlan[]
dp.setDemand({ date, chartTiles, basemapTiles, airspaceTiles, center: { x, y },
               scrub: { direction /* -1 | 0 | 1 */, velocity /* frames per second */, playing }, solo })
  // replaces all outstanding demand: fetch the current plan first (centre first, low-zoom
  // ancestors before full resolution), prefetch according to the scrub state, cancel anything no longer wanted
dp.on('tile', ({ key, bitmap }) => {})        // ownership of the bitmap passes to the listener
dp.on('absent', ({ key }) => {})
dp.on('airspace', ({ tileId, batch }) => {})
dp.on('error', ({ key, error }) => {})
dp.markEvicted(key)
dp.readiness(date, tiles)                     // → 0..1, the share of that plan's items delivered and not evicted
dp.loadAirfields()                            // → Promise<{ mx, my, start, end, status }>
dp.airfieldDetails(index)                     // → Promise<object>
dp.airspaceRegionMask(day)                    // → bitmask of regions with an airspace cycle in effect that day
dp.queryPin(lng, lat, date)                   // → Promise<[{ location, chart, contains, dist }]>, ChartIndex.query semantics, top 6
dp.queryAirspace(lng, lat, day)               // → Promise<[...]>, the data behind AirspaceLayer.renderStack (data, not HTML)
dp.stats()                                    // → { inflight, queued, bytesCached }
dp.destroy()
```
