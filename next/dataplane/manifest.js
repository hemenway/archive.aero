export const DAY_MS = 86400000;
export function day(value) {
  if (Number.isInteger(value)) return value;
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new TypeError('Expected ISO date or integer Day');
  const t = Date.parse(value);
  if (!Number.isFinite(t) || new Date(t).toISOString().slice(0, 10) !== value) throw new TypeError('Invalid date');
  return t / DAY_MS;
}
export const isoDay = d => new Date(d * DAY_MS).toISOString().slice(0, 10);
export function upperBound(array, value) {
  let lo = 0, hi = array.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (array[mid] <= value) lo = mid + 1; else hi = mid; }
  return lo;
}
// Manifest order is retained; the search index is independently sorted.
export function parseManifest(raw) {
  if (!raw || raw.version !== 1 || !Array.isArray(raw.eras)) throw new Error('Unsupported manifest');
  const count = raw.eras.length;
  const starts = new Int32Array(count), ends = new Int32Array(count);
  const minZoom = new Uint8Array(count), maxZoom = new Uint8Array(count);
  const bounds = [], coverage = [], paths = [], keys = [];
  raw.eras.forEach((e, i) => {
    if (typeof e.k !== 'string' || !/^\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}$/.test(e.k) || !/^[a-f0-9]{12}$/.test(e.h)) throw new Error('Invalid era key/hash');
    const range = e.k.split('_to_');
    starts[i] = day(range[0]); ends[i] = day(range[1]);
    if (starts[i] >= ends[i]) throw new Error('Empty/reversed era interval');
    const z0 = e.z?.[0], z1 = e.z?.[1];
    if (!Number.isInteger(z0) || !Number.isInteger(z1) || z0 < 0 || z1 > 24 || z0 > z1) throw new Error('Invalid era zoom range');
    minZoom[i] = z0; maxZoom[i] = z1;
    const b = e.b ?? null;
    if (b && (!Array.isArray(b) || b.length !== 4 || !b.every(Number.isFinite) || b[0] > b[2] || b[1] > b[3])) throw new Error('Invalid era bounds');
    bounds.push(b); paths.push(`sectionals/${e.k}.${e.h}`); keys.push(e.k ?? paths[i]);
    if (typeof paths[i] !== 'string' || !paths[i]) throw new Error('Missing archive path');
    if (e.c != null && (!Array.isArray(e.c) || !e.c.every(v => Number.isInteger(v) && v >= 0 && v < 4096))) throw new Error('Invalid z6 coverage');
    coverage.push(e.c == null ? null : new Set(e.c));
  });
  const order = Array.from({length: count}, (_, i) => i).sort((a,b) => starts[a] - starts[b] || a-b);
  const sortedStarts = Int32Array.from(order, i => starts[i]);
  const prefixEnd = new Int32Array(count);
  order.forEach((i, j) => { prefixEnd[j] = j ? Math.max(prefixEnd[j-1], ends[i]) : ends[i]; });
  const frames = Array.from(new Set(starts)).sort((a,b)=>a-b);
  // Demand planning looks eras up by path on every camera move; a Map keeps that O(1).
  const pathIndex = new Map(paths.map((p, i) => [p, i]));
  return { raw, starts, ends, minZoom, maxZoom, bounds, coverage, paths, keys, order, sortedStarts, prefixEnd, pathIndex,
    frames: frames.map(isoDay), frameDays: Int32Array.from(frames), dateBounds: count ? {min: isoDay(frames[0]), max: isoDay(Math.max(...ends)-1)} : {min:null,max:null} };
}
export function erasAt(manifest, date) {
  const d = day(date), out = [];
  for (let j = upperBound(manifest.sortedStarts, d)-1; j >= 0 && manifest.prefixEnd[j] > d; j--) {
    const i = manifest.order[j]; if (manifest.ends[i] > d) out.push(i);
  }
  return out.sort((a,b)=>a-b);
}
export async function loadManifest(url, fetcher = fetch, signal) {
  const response = await fetcher(url, {signal});
  if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
  const raw = await response.json();
  // Production manifests carry absolute bases; fixtures may use relative ones.
  // Resolve them once, against the manifest URL, so every consumer can build URLs.
  if (raw && typeof raw === 'object') {
    const origin = new URL(String(url), globalThis.location?.href ?? 'http://localhost/').href;
    for (const base of ['tileBase', 'fileBase']) if (typeof raw[base] === 'string') raw[base] = new URL(raw[base], origin).href;
  }
  return parseManifest(raw);
}
