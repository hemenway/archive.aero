# Upstream report: protomaps-leaflet label-index pruning loop

Draft for a GitHub issue on protomaps/protomaps-leaflet. Status: not filed.
Related: #212 (closed 2026-02-18, "CenteredTextSymbolizer can lead to
renderTile loop") — same symptom, different cause; the maintainer's reply
there says the library is in minimal maintenance mode. This one triggers
with the stock flavor and no custom symbolizers, so it is worth having on
record even if the answer is "migrate to MapLibre".

Our fix: archive.aero commit 80d4cb8 (raise `labelers.maxLabeledTiles`
after creating the layer). Reproduction harness: the Playwright fuzz script
described below.

---

## Title

Label index prunes on-screen tiles above 16 data tiles per zoom → endless layout/rerender loop, tab freezes

## Body

### Summary

`leafletLayer()` hard-codes `maxLabeledTiles = 16` for the label index
(`src/frontends/leaflet.ts`, both in the constructor and in `clearLayout()`).
Any viewport that shows more than 16 data tiles at one zoom — a 2560×1321
window at `devicePixelRatio` 2 shows 18–24 — makes `Index.pruneOrNoop()`
evict tiles that are still on screen. From there the layout / invalidate /
rerender cycle can feed itself without ever leaving the microtask queue.
The main thread pins one core, memory grows without bound (I measured a
15.7 GB WebContent process in Safari before killing it), and the page never
recovers.

Not engine-specific: reproduced in WebKit and Chromium. Not symbolizer-
specific: stock `flavor: 'dark'` labels, no `CenteredTextSymbolizer`
(cf. #212).

### Environment

- protomaps-leaflet 5.0.0, Leaflet 1.9.4, pmtiles 3.0.6
- Basemap: a PMTiles extract of the Protomaps daily planet build
  (z0–6 world + z7–13 regional), served with HTTP range requests
- Layer options: `{ url, flavor: 'dark', lang: 'en', maxDataZoom: 13 }`
- Map options: `zoomSnap: 1`, `zoomAnimation: true`
- Viewport 2560×1321 CSS px, `devicePixelRatio` 2
- Safari 27.0 on macOS 27.0 (original freeze); Playwright WebKit 26.0 and
  Chromium 145 (scripted reproduction)

### Steps to reproduce

1. Open a map with the layer above in a window at least ~2500 px wide
   (data tiles are 512 CSS px at `levelDiff` 1, so this puts 18–24 data
   tiles at one zoom in view).
2. Zoom and pan around at z5–z10 with animation on. A few zoom steps are
   usually enough; it is timing-dependent, so it can take a dozen.
3. The tab stops responding.

A scripted reproduction that trips it in 10 of 12 runs (both engines):
random `zoomIn()` / `zoomOut()` / `panBy()` / `setView()` steps 0.3–1.8 s
apart, with a guard that counts `CanvasRenderingContext2D.measureText`
calls between 50 ms timer beats and throws once a single burst passes
150 000 calls (a normal full-screen layout is under 5 000). The throw also
recovers the page, which is how the stack below was captured.

### What happens

Native sample of the frozen Safari process (3 s, 1567 samples): every
sample is on the main thread inside one `DOMTimer::fired` →
`MicrotaskQueue::performMicrotaskCheckpoint` → `drainWithoutUseCallOnEachMicrotask`,
i.e. one microtask checkpoint that never drains. Hot leaves are
`measureText`, `setTimeout` (the `timer()` await in `renderTile` — each
lap installs one more timer, which is where the memory goes), `String.split`
and `+string` (from `pruneOrNoop`), and GC sweeping rope strings.

JS stack at the guard trip:

```
Labelers.add → Labeler.add → Labeler.layout → GroupSymbolizer.place
→ OffsetTextSymbolizer.place → OffsetSymbolizer.place → TextSymbolizer.place
→ measureText
  (called from the renderTile continuation, src/frontends/leaflet.ts)
```

### Mechanism (reading `src/labeler.ts`)

1. `Labeler.layout()` inserts a tile's labels and, for any label that
   collides with or crosses into a neighbouring tile, adds that neighbour
   to the invalidated set via `findInvalidatedTiles()` — but only if
   `index.hasPrefix(neighbourKey)` is true, i.e. the neighbour is
   currently in the index.
2. At the end of `layout()`, `pruneOrNoop()` runs for every data tile just
   added. Once `keysForDs > maxLabeledTiles` it calls `pruneKey(maxKey)`
   for the farthest tile *of the keys iterated so far* — and it keeps
   calling it on every later iteration of the same loop, so one `layout()`
   can evict several tiles, not just the farthest one. With more than 16
   data tiles on screen, the evicted tiles are visible ones.
3. `layout()` then fires the callback with the invalidated set, which
   `rerenderTile()`s each of them. Step 1 checked membership *before* step
   2 pruned, so an invalidated tile can already be gone from the index by
   the time its `renderTile()` continuation runs.
4. That continuation calls `labelers.add()`; the tile is missing from the
   index, so it lays out again, invalidates *its* neighbours, prunes the
   farthest tiles from *its* position (different ones), fires the callback,
   and so on. When the invalidation graph over the visible tiles is
   connected — it usually is, labels cross tile edges everywhere — the
   cycle sustains itself. Every hop is a promise continuation (the tile
   data is cached, so `getDisplayTile()` resolves immediately), so nothing
   ever yields to the event loop; the `await timer()` in `renderTile` comes
   after `labelers.add`, so its timers only pile up.

### Workaround

Raising the cap above anything the screen plus Leaflet's `keepBuffer` can
hold makes pruning only ever touch off-screen tiles, which is what it was
designed for:

```js
const layer = protomapsL.leafletLayer({ url, flavor: 'dark' });
const cap = 256; // > visible data tiles + keepBuffer, at any window size
layer.labelers.maxLabeledTiles = cap;
const clearLayout = layer.clearLayout.bind(layer);
layer.clearLayout = () => { clearLayout(); layer.labelers.maxLabeledTiles = cap; };
```

With this in place the same fuzz sequences ran 0 for 20 (12 runs in the
browser with the cap injected, 8 against the patched build).

### Suggested fix

Either or both:

- Expose `maxLabeledTiles` as a `leafletLayer` option (it is already a
  constructor parameter of `Labelers`), and derive the default from the
  map size instead of a constant, e.g.
  `ceil(width / 512 + 3) * ceil(height / 512 + 3)`.
- Make `pruneOrNoop()` never evict a tile that is still displayed (pass
  the set of live tile keys from the frontend, or check
  `layer._tiles`), and prune at most one key per call — the `pruneKey`
  call belongs after the loop, not inside it.

The second option removes the root cause; the first is a one-line change
that makes it unreachable on real screens.
