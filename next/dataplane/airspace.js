import { decodeMVT } from './mvt.js';

export const REGION_ORDER = Object.freeze(['us', 'fr', 'br']);
export const BADGE_ORDER = Object.freeze(['A', 'B', 'C', 'D', 'E', 'TMA', 'CTA', 'CTR', 'ATZ']);
export const AIRSPACE_STYLES = Object.freeze({ A: 0, B: 1, C: 2, D: 3, E: 4, unclassed: 5, ribbon700: 6, ribbonOther: 7 });
export const CURRENT = 2147483647;
const DAY_MS = 86400000;
const E_FLOOR = new Set(['E5', 'E6', 'E7']);

// Timeline API inputs are Days; only source metadata/properties use ISO or
// YYYYMMDD. Keeping conversion at the boundary avoids calendar/DST arithmetic.
export function sourceDay(value) {
  const s = String(value), m = /^(\d{4})-?(\d{2})-?(\d{2})$/.exec(s);
  if (!m) throw new Error(`Invalid airspace date: ${s}`);
  const y = Number(m[1]), mo = Number(m[2]), d = Number(m[3]);
  const t = new Date(0);
  t.setUTCFullYear(y, mo - 1, d); t.setUTCHours(0, 0, 0, 0);
  if (t.getUTCFullYear() !== y || t.getUTCMonth() !== mo - 1 || t.getUTCDate() !== d) throw new Error(`Invalid airspace date: ${s}`);
  return Math.floor(t.getTime() / DAY_MS);
}
const iso = day => new Date(day * DAY_MS).toISOString().slice(0, 10);
const fmtDay = day => new Date(day * DAY_MS).toLocaleDateString('en-US', { timeZone: 'UTC', year: 'numeric', month: 'short', day: 'numeric' });
export const badge = p => p.cls || p.lt || '?';
export const floorFt = p => p.lo == null ? 0 : p.loc === 'STD' ? p.lo * 100 : p.lo;
export const shortName = p => (p.name || p.id || '').replace(/\s+CLASS\s+[A-G]\d?\s*$/i, '').trim() || p.name || '';
export function fmtAlt(val, code) {
  if (code === 'UNLTD') return 'unlimited';
  if (val == null) return null;
  if (code === 'SFC') return val ? `${val.toLocaleString('en-US')} AGL` : 'SFC';
  if (code === 'STD') return `FL${val}`;
  return `${val.toLocaleString('en-US')} ${code || 'MSL'}`;
}
export function altSpan(p) {
  const lo = fmtAlt(p.lo, p.loc), hi = fmtAlt(p.hi, p.hic);
  if (lo && hi) return p.loc === 'MSL' && p.hic === 'MSL' ? `${p.lo.toLocaleString('en-US')} – ${hi}` : `${lo} – ${hi}`;
  return lo ? `from ${lo}` : hi ? `to ${hi}` : '';
}

function styleOf(layer, p) {
  if (layer === 'efloor') return p.k === '700' ? 6 : 7;
  if (layer !== 'class' || p.cls === 'E' && E_FLOOR.has(p.lt)) return null;
  if (!p.cls) return 5;
  return Object.hasOwn(AIRSPACE_STYLES, p.cls) && /^[A-E]$/.test(p.cls) ? AIRSPACE_STYLES[p.cls] : null;
}
const typeOn = (p, classMask) => !!(classMask & (p.cls === 'E' ? 2 : 1));
function interval(p) { return { from: sourceDay(p.from), to: p.to == null ? CURRENT : sourceDay(p.to) }; }

/** The batch owns its buffers; polygons have separate integer-coordinate
 * rings which can safely stay in the Worker after batch buffers transfer. */
export function decodeAirspaceTile(input, tile) {
  const decoded = input?.layers ? input : decodeMVT(input);
  const { z, x, y } = tile;
  if (!Number.isInteger(z) || z < 0 || z > 24 || !Number.isInteger(x) || !Number.isInteger(y)
      || x < 0 || y < 0 || x >= 2 ** z || y >= 2 ** z) throw new Error('Invalid airspace tile coordinates');
  const lines = [], polygons = [];
  let vertices = 0;
  for (const layer of decoded.layers) {
    if (!['class', 'efloor'].includes(layer.name)) continue;
    for (const feature of layer.features) {
      const p = feature.properties, rg = REGION_ORDER.indexOf(p.rg);
      if (rg < 0) continue;
      const dates = interval(p);
      if (dates.to <= dates.from) throw new Error('Invalid airspace feature interval');
      if (layer.name === 'class' && feature.type === 3) {
        const bounds = [Infinity, Infinity, -Infinity, -Infinity];
        for (const ring of feature.geometry) for (let i = 0; i < ring.length; i += 2) {
          bounds[0] = Math.min(bounds[0], ring[i]); bounds[1] = Math.min(bounds[1], ring[i + 1]);
          bounds[2] = Math.max(bounds[2], ring[i]); bounds[3] = Math.max(bounds[3], ring[i + 1]);
        }
        polygons.push({ properties: p, rings: feature.geometry, extent: layer.extent, bounds, ...dates });
      }
      const style = styleOf(layer.name, p);
      if (style == null || (layer.name === 'class' ? feature.type !== 3 : feature.type !== 2)) continue;
      for (const path of feature.geometry) {
        if (path.length < 4) continue;
        lines.push({ path, extent: layer.extent, style, rg, ...dates });
        vertices += path.length / 2;
      }
    }
  }
  const batch = {
    positions: new Float32Array(vertices * 2), starts: new Uint32Array(lines.length + 1),
    from: new Int32Array(lines.length), to: new Int32Array(lines.length),
    style: new Uint8Array(lines.length), rg: new Uint8Array(lines.length)
  };
  let v = 0;
  const scale = 2 ** -z;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    batch.starts[i] = v;
    batch.from[i] = line.from; batch.to[i] = line.to; batch.style[i] = line.style; batch.rg[i] = line.rg;
    for (let j = 0; j < line.path.length; j += 2) {
      batch.positions[v * 2] = (x + line.path[j] / line.extent) * scale;
      batch.positions[v * 2 + 1] = (y + line.path[j + 1] / line.extent) * scale;
      v++;
    }
  }
  batch.starts[lines.length] = v;
  return { batch, polygons };
}

export const batchTransfers = batch => Object.values(batch).map(a => a.buffer);

function inBox([w, s, e, n], lng, lat) {
  if (lat < s || lat > n) return false;
  if (e < w) e += 360;
  const turns = Math.round(((w + e) / 2 - lng) / 360);
  return lng + 360 * turns >= w && lng + 360 * turns <= e;
}
function boxesIntersect([w, s, e, n], [wv, sv, ev, nv]) {
  if (n < sv || s > nv) return false;
  if (e < w) e += 360;
  if (ev < wv) ev += 360;
  const turns = Math.round(((wv + ev) / 2 - (w + e) / 2) / 360);
  return [turns - 1, turns, turns + 1].some(k => e + 360 * k >= wv && w + 360 * k <= ev);
}

// Even-odd rings work for holes and multiple polygon exteriors independent
// of winding. A point on a real edge is counted as inside.
export function containsRings(rings, x, y) {
  let inside = false;
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 2; i < ring.length; j = i, i += 2) {
      const ax = ring[j], ay = ring[j + 1], bx = ring[i], by = ring[i + 1];
      const dx = bx - ax, dy = by - ay, cross = (x - ax) * dy - (y - ay) * dx;
      if ((dx || dy) && Math.abs(cross) < 1e-7 && x >= Math.min(ax, bx) && x <= Math.max(ax, bx)
          && y >= Math.min(ay, by) && y <= Math.max(ay, by)) return true;
      if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) inside = !inside;
    }
  }
  return inside;
}

export class AirspaceIndex {
  constructor(metadata = {}) { this.tiles = new Map(); this.setMetadata(metadata); }
  setMetadata(metadata = {}) {
    this.meta = metadata.archive_aero || metadata;
    this.regions = this.meta.regions || {};
    this.cycleDays = this.meta.cycle_days || 28;
    if (!Number.isInteger(this.cycleDays) || this.cycleDays < 1) throw new Error('Invalid airspace cycle_days');
    this.cycles = Object.create(null);
    for (const rg of REGION_ORDER) {
      this.cycles[rg] = [...new Set((this.regions[rg]?.cycles || []).map(sourceDay))].sort((a, b) => a - b);
    }
  }
  cycleFor(rg, day) {
    if (!Number.isInteger(day)) return null;
    const cycles = this.cycles[rg] || [];
    let lo = 0, hi = cycles.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (cycles[mid] <= day) lo = mid + 1; else hi = mid; }
    const best = cycles[lo - 1];
    return best == null || day >= best + this.cycleDays ? null : best;
  }
  regionMask(day) {
    let mask = 0;
    REGION_ORDER.forEach((rg, i) => { if (this.cycleFor(rg, day) != null) mask |= 1 << i; });
    return mask;
  }
  setTile(key, tile, decoded) {
    const result = decoded?.polygons ? decoded : decodeAirspaceTile(decoded, tile);
    // Integer rings stay retained here; only batch arrays are transferred.
    this.tiles.set(key, { tile: { ...tile }, polygons: result.polygons });
    return result.batch;
  }
  deleteTile(key) { this.tiles.delete(key); }
  clear() { this.tiles.clear(); }
  regionsInView(bounds) {
    return REGION_ORDER.filter(rg => this.regions[rg] && (!bounds || (this.regions[rg].boxes || []).some(box => boxesIntersect(box, bounds))));
  }
  regionsAt(lng, lat) {
    return REGION_ORDER.filter(rg => (this.regions[rg]?.boxes || []).some(box => inBox(box, lng, lat)));
  }
  regionStatus(rg, day) {
    const r = this.regions[rg];
    if (!r) return '';
    const cycle = this.cycleFor(rg, day), cycles = this.cycles[rg], base = `${r.name} · ${r.source}`;
    if (cycle != null) return `${base} cycle ${fmtDay(cycle)}`;
    if (!Number.isInteger(day) || !cycles.length) return base;
    if (day < cycles[0]) return `${r.name} · no ${r.source} data before ${fmtDay(cycles[0])}`;
    let prev = null, next = null;
    for (const c of cycles) { if (c <= day) prev = c; else { next = c; break; } }
    return next != null
      ? `${r.name} · no ${r.source} data ${fmtDay(prev + this.cycleDays)} to ${fmtDay(next - 1)}`
      : `${r.name} · no ${r.source} data after ${fmtDay(prev + this.cycleDays - 1)} yet`;
  }
  status(day, bounds, { configured = true, enabled = true, loaded = true, loadError = null } = {}) {
    if (!configured) return ['Airspace data not configured'];
    if (!enabled) return [loadError ? `Airspace data unavailable: ${loadError}` : 'US (FAA NASR), France (SIA), Brazil (DECEA)'];
    if (!loaded) return ['Loading airspace…'];
    const inView = this.regionsInView(bounds);
    return inView.length ? inView.map(rg => this.regionStatus(rg, day)) : ['No airspace data for this area'];
  }
  query(lng, lat, day, { classMask = 3 } = {}) {
    if (!Number.isFinite(lng) || !Number.isFinite(lat) || lat < -90 || lat > 90) return [];
    const here = this.regionsAt(lng, lat).filter(rg => this.cycleFor(rg, day) != null);
    if (!here.length) return [];
    const mx = ((lng / 360 + 0.5) % 1 + 1) % 1;
    const sin = Math.sin(Math.max(-85.0511287798066, Math.min(85.0511287798066, lat)) * Math.PI / 180);
    const my = 0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI);
    const seen = new Set(), rows = [];
    for (const { tile, polygons } of this.tiles.values()) {
      const n = 2 ** tile.z, worldX = mx * n;
      const tx = worldX + Math.round((tile.x + 0.5 - worldX) / n) * n - tile.x, ty = my * n - tile.y;
      for (const polygon of polygons) {
        const p = polygon.properties, key = `${p.rg}|${p.v ?? JSON.stringify(p)}`;
        if (p.ex || seen.has(key) || !here.includes(p.rg) || !typeOn(p, classMask)
            || day < polygon.from || day >= polygon.to) continue;
        const py = ty * polygon.extent, [w, s, e, north] = polygon.bounds;
        if (py < s || py > north) continue;
        // At z0 either antimeridian edge can carry buffered rings. The
        // nearest tile centre alone cannot choose the correct world copy.
        if (![tx, tx - n, tx + n].some(qx => {
          const px = qx * polygon.extent;
          return px >= w && px <= e && containsRings(polygon.rings, px, py);
        })) continue;
        seen.add(key);
        rows.push(p);
      }
    }
    const order = p => BADGE_ORDER.indexOf(badge(p));
    rows.sort((a, b) => floorFt(a) - floorFt(b) || order(a) - order(b) || REGION_ORDER.indexOf(a.rg) - REGION_ORDER.indexOf(b.rg));
    const governing = new Set();
    return rows.filter(p => { const k = badge(p); if (governing.has(k)) return false; governing.add(k); return true; })
      .map(p => ({ ...p, badge: badge(p), floorFt: floorFt(p), altSpan: altSpan(p), shortName: shortName(p),
        cycle: iso(this.cycleFor(p.rg, day)), source: this.regions[p.rg]?.source }));
  }
  // Sources to credit on the map while the layer is drawn.
  credits() {
    return REGION_ORDER.filter(rg => this.regions[rg]).map(rg => ({ source: this.regions[rg].source, url: this.regions[rg].source_url ?? null }));
  }
  // What the pin card's "Airspace here" section needs in one call: the
  // regions under the point with a cycle in effect, and their governing rows.
  stack(lng, lat, day) {
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) return { here: [], rows: [] };
    const here = this.regionsAt(lng, lat).filter(rg => this.cycleFor(rg, day) != null).map(rg => {
      const r = this.regions[rg];
      return { rg, name: r.name, source: r.source, note: r.note ?? null, cycle: iso(this.cycleFor(rg, day)) };
    });
    return { here, rows: here.length ? this.query(lng, lat, day) : [] };
  }
}
