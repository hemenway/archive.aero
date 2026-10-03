import { project } from '../camera.js';
import { fields } from './manifest.js';
export async function createDataPlane({ manifestUrl, fetch: fetcher = globalThis.fetch.bind(globalThis), worker = true, earlyFetches = new Map() }) {
  const response = await fetcher(manifestUrl);
  if (!response.ok) throw new Error('Manifest failed to load');
  const source = await response.json();
  if (source.version !== 1 || !source.eras?.length) throw new Error('Manifest failed validation');
  const frames = [...new Set(source.eras.map(e => e.k.split('_to_')[0]))].sort();
  const ends = source.eras.map(e => e.k.split('_to_')[1]).sort();
  const manifest = { frames, dateBounds: { min: frames[0], max: new Date(Date.parse(ends.at(-1)) - 86400000).toISOString().slice(0, 10) }, coverage: source.coverage, eraCount: source.eras.length, hasAirspace: !!source.airspace, hasAirfields: !!source.airfields, hasPins: !!source.pins };
  const events = new Map(), resident = new Set(), pending = new Map(), absent = new Set();
  const emit = (type, payload) => { for (const cb of events.get(type) || []) cb(payload); };
  let engine, failed = false, generation = 0;
  if (worker) {
    engine = new Worker(__WORKER_URL__, { type: 'module' });
    engine.onerror = event => { event.preventDefault(); failed = true; pending.clear(); emit('error', { error: new Error('Data worker failed. Reload to retry.') }); };
    engine.onmessage = ({ data }) => {
      const wanted = pending.get(data.key);
      if (!wanted) { data.bitmap.close(); return; }
      pending.delete(data.key); resident.add(data.key); emit('tile', data);
    };
  }
  const intersects = (e, t) => {
    if (!e.b) return true;
    const nw = project(e.b[3], e.b[0], t.z), se = project(e.b[1], e.b[2], t.z);
    return (t.x + 1) * 256 >= nw[0] && t.x * 256 <= se[0] && (t.y + 1) * 256 >= nw[1] && t.y * 256 <= se[1];
  };
  function planCharts(date, tiles, { solo } = {}) {
    const eras = solo ? solo.paths.map(p => ({ p, z: solo.zoom, b: null })) : source.eras.filter(e => { const [a, b] = e.k.split('_to_'); return date >= a && date < b; }).map(e => ({ ...e, p: `sectionals/${e.k}.${e.h}` }));
    return { id: `${date}:${solo?.paths.join(',') || ''}`, tiles: tiles.map(dst => ({ dst, items: eras.filter(e => dst.z >= e.z[0] && intersects(e, dst)).map(e => {
      const z = Math.min(dst.z, e.z[1]), scale = 2 ** (dst.z - z), src = { z, x: Math.floor(dst.x / scale), y: Math.floor(dst.y / scale) };
      return { key: `${e.p}/${src.z}/${src.x}/${src.y}`, src, dst, ...(solo?.clip && { clip: solo.clip.id }) };
    }) })) };
  }
  const planBasemap = tiles => source.basemap ? tiles.filter(dst => dst.z >= source.basemap.z[0]).map(dst => {
    const z = Math.min(dst.z, source.basemap.z[1]), scale = 2 ** (dst.z - z), src = { z, x: Math.floor(dst.x / scale), y: Math.floor(dst.y / scale) };
    return { dst, items: [{ key: `${source.basemap.p}/${z}/${src.x}/${src.y}`, src, dst }] };
  }) : [];
  function setDemand(demand) {
    generation++;
    if (demand.airspaceTiles.length) { const positions = [-97.4, 32.3, -96.3, 32.3, -96.3, 33.3, -97.4, 33.3, -97.4, 32.3]; const mercator = []; for (let i = 0; i < positions.length; i += 2) { const [x, y] = project(positions[i + 1], positions[i], 0); mercator.push(x / 256, y / 256); } emit('airspace', { tileId: 'fixture', batch: { positions: new Float32Array(mercator), starts: new Uint32Array([0, 5]), from: new Int32Array([-3653]), to: new Int32Array([2147483647]), style: new Uint8Array([3]), rg: new Uint8Array([0]) } }); }
    const plans = [...planCharts(demand.date, demand.chartTiles, { solo: demand.solo }).tiles, ...planBasemap(demand.basemapTiles)];
    const needed = new Set(plans.flatMap(t => t.items.map(i => i.key)));
    for (const [key, task] of pending) if (!needed.has(key)) { task.controller.abort(); pending.delete(key); }
    for (const tile of plans) for (const item of tile.items) {
      if (resident.has(item.key) || absent.has(item.key) || pending.has(item.key) || failed) continue;
      const controller = new AbortController(), task = { controller, generation }; pending.set(item.key, task);
      const url = new URL(item.key, new URL(source.tileBase, new URL(manifestUrl, location.href))).href;
      const request = earlyFetches.get(url) || fetcher(url, { signal: controller.signal }); earlyFetches.delete(url);
      Promise.resolve(request).then(async resp => {
        if (pending.get(item.key) !== task) return;
        if (resp.status === 204) { pending.delete(item.key); absent.add(item.key); emit('absent', { key: item.key }); return; }
        if (!resp.ok) throw new Error('Tile unavailable');
        await resp.arrayBuffer();
        if (pending.get(item.key) !== task) return;
        if (engine) engine.postMessage({ key: item.key });
        else { const c = new OffscreenCanvas(256, 256); const ctx = c.getContext('2d'); ctx.fillStyle = '#c9b78b'; ctx.fillRect(0, 0, 256, 256); pending.delete(item.key); resident.add(item.key); emit('tile', { key: item.key, bitmap: c.transferToImageBitmap() }); }
      }).catch(error => { if (pending.get(item.key) !== task) return; pending.delete(item.key); if (error.name !== 'AbortError') emit('error', { key: item.key, error }); });
    }
  }
  return {
    manifest, planCharts, planBasemap, setDemand,
    on(type, cb) { if (!events.has(type)) events.set(type, new Set()); events.get(type).add(cb); },
    markEvicted(key) { resident.delete(key); },
    readiness(date, tiles) { const keys = planCharts(date, tiles).tiles.flatMap(t => t.items.map(i => i.key)); return !keys.length ? 1 : keys.filter(k => resident.has(k) || absent.has(k)).length / keys.length; },
    async loadAirfields() { const mx = [], my = []; for (const f of fields) { const [x, y] = project(f.lat, f.lng, 0); mx.push(x / 256); my.push(y / 256); } return { mx: new Float32Array(mx), my: new Float32Array(my), start: new Uint16Array(fields.map(f => f.start_year || 0)), end: new Uint16Array(fields.map(f => f.end_year || 0)), status: new Uint8Array(fields.map(f => ['open', 'gone', 'unknown'].indexOf(f.status))) }; },
    async airfieldDetails(index) { return fields[index]; },
    airspaceRegionMask(day) { return day >= -3653 ? 1 : 0; },
    async queryPin(lng, lat, date) { return source.eras.filter(e => { const [a, b] = e.k.split('_to_'); return date >= a && date < b && (!e.b || lng >= e.b[0] && lng <= e.b[2] && lat >= e.b[1] && lat <= e.b[3]); }).slice(0, 6).map(e => ({ location: { name: 'Test Dallas' }, chart: { d: e.k.split('_to_')[0], e: e.k.split('_to_')[1], ed: '1', pm: `sectionals/${e.k}.${e.h}`, pmz: e.z, pmb: e.b }, contains: true, dist: 0 })); },
    async queryAirspace(lng, lat, day) { return day >= -3653 ? [{ name: 'Test controlled airspace', class: 'D', region: 'us' }] : []; },
    stats() { return { inflight: pending.size, queued: 0, bytesCached: resident.size * 262144 }; },
    destroy() { for (const p of pending.values()) p.controller.abort(); engine?.terminate(); events.clear(); }
  };
}
