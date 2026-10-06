// The production basemap, painted here instead of by a Leaflet layer: the same Protomaps vector tiles, flavor and
// rules (protomaps-leaflet's paint and label code), drawn one data tile at a time onto a 512 px canvas and handed to
// the renderer as a bitmap. Runs wherever the data-plane core runs; in the Worker that needs OffscreenCanvas 2D.
// Nothing here touches the DOM-only parts of protomaps-leaflet (its Leaflet layer, sprites, font loading).
import Point from '@mapbox/point-geometry';
import { VectorTile } from '@mapbox/vector-tile';
import Protobuf from 'pbf';
import { paint, Labelers, paintRules, labelRules } from 'protomaps-leaflet';
import { namedFlavor } from '@protomaps/basemaps';
const SIZE = 512, BUFFER = 16;
// protomaps-leaflet keeps its tile parser private; this is the same reading of a tile, scaled to SIZE units.
function parse(bytes) {
  const tile = new VectorTile(new Protobuf(bytes)), out = new Map();
  for (const [name, layer] of Object.entries(tile.layers)) {
    const features = [], scale = SIZE / layer.extent;
    for (let i = 0; i < layer.length; i++) {
      const f = layer.feature(i), geom = f.loadGeometry();
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity, n = 0;
      for (const ring of geom) for (const p of ring) { p.x *= scale; p.y *= scale; n++; if (p.x < x1) x1 = p.x; if (p.x > x2) x2 = p.x; if (p.y < y1) y1 = p.y; if (p.y > y2) y2 = p.y; }
      features.push({ id: f.id, geomType: f.type, geom, numVertices: n, bbox: { minX: x1, minY: y1, maxX: x2, maxY: y2 }, props: f.properties });
    }
    out.set(name, features);
  }
  return out;
}
const canvasOf = size => typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(size, size) : Object.assign(document.createElement('canvas'), { width: size, height: size });
export function createPainter({ flavor = 'dark', lang = 'en', repaint = () => {} } = {}) {
  const style = namedFlavor(flavor), paints = paintRules(style), labels = labelRules(style, lang);
  const canvas = canvasOf(SIZE), ctx = canvas.getContext('2d'), scratch = canvasOf(1).getContext('2d');
  if (!ctx || !scratch) throw new Error('No 2D canvas for the basemap on this thread');
  // key -> {src, prepared}: tiles the renderer holds, kept so a neighbour's label can be drawn into them later.
  const live = new Map(), dirty = new Set(); let flushing = false, lastPaint = 0;
  const later = (fn, ms) => setTimeout(fn, ms);
  const keyOf = new Map(); // "z/x/y" of a data tile -> key
  // A label that crosses into tiles already painted invalidates them. Names arrive as display tiles (256 px, one zoom
  // deeper than the data); each maps back to the data tile that was painted.
  const invalidated = names => {
    for (const name of names) { const [x, y, z] = name.split(':').map(Number), key = keyOf.get(`${z - 1}/${x >> 1}/${y >> 1}`); if (key && live.has(key)) dirty.add(key); }
    if (dirty.size && !flushing) { flushing = true; later(flush, 120); }
  };
  // The cap is far above what a view holds: protomaps-leaflet 5.0.0 prunes label tiles past it and a pruned tile that
  // is still on screen lays out again, which never settles (the production viewer raises its cap for the same reason).
  const labelers = new Labelers(scratch, labels, 512, invalidated);
  const draw = async ({ src, prepared }) => {
    const z = src.z + 1, tiles = new Map([['', [prepared]]]), origin = prepared.origin;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.fillStyle = style.background; ctx.fillRect(0, 0, SIZE, SIZE);
    paint(ctx, z, tiles, labelers.getIndex(z) ?? null, paints, { minX: origin.x - BUFFER, minY: origin.y - BUFFER, maxX: origin.x + SIZE + BUFFER, maxY: origin.y + SIZE + BUFFER }, origin, false);
    return canvas.transferToImageBitmap ? canvas.transferToImageBitmap() : createImageBitmap(canvas);
  };
  // Repaints wait until first paints have gone quiet (each new tile can invalidate its neighbours again), then go one
  // per turn of the event loop so nothing else on this thread waits behind them.
  async function flush() {
    if (performance.now() - lastPaint < 100) { later(flush, 120); return; }
    const key = dirty.values().next().value, entry = live.get(key); dirty.delete(key);
    try { if (entry) repaint(key, await draw(entry)); }
    finally { if (dirty.size) later(flush, 0); else flushing = false; }
  }
  return {
    // One data tile z/x/y covers 512 px of the display zoom one level deeper, where its labels are laid out.
    async paint(key, src, bytes) {
      const entry = { src, prepared: { z: src.z + 1, origin: new Point(src.x * SIZE, src.y * SIZE), data: parse(bytes), scale: 1, dim: SIZE, dataTile: { z: src.z, x: src.x, y: src.y } } };
      live.set(key, entry); keyOf.set(`${src.z}/${src.x}/${src.y}`, key);
      labelers.add(entry.prepared.z, new Map([['', [entry.prepared]]]));
      lastPaint = performance.now();
      return draw(entry);
    },
    forget(key) { const entry = live.get(key); if (!entry) return; live.delete(key); dirty.delete(key); keyOf.delete(`${entry.src.z}/${entry.src.x}/${entry.src.y}`); },
    // Parsed geometry is kept only for tiles still demanded, and label layouts only for the zooms those tiles are at.
    retain(wanted) {
      const zooms = new Set();
      for (const [key, entry] of live) { if (wanted.has(key)) zooms.add(entry.prepared.z); else this.forget(key); }
      if (zooms.size) for (const z of [...labelers.labelers.keys()]) if (!zooms.has(z)) labelers.labelers.delete(z);
    },
  };
}
