import { project, unproject, visibleTiles } from '../camera.js';
export class RendererUnsupportedError extends Error {}
export function createRenderer(canvas, { minZoom = 4, maxZoom = 14 } = {}) {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new RendererUnsupportedError('Canvas unavailable');
  const events = new Map(), textures = new Map(), completed = new Map();
  let camera = { x: 0.5, y: 0.5, zoom: minZoom }, plan = { tiles: [] }, base = [], style = { opacity: 1 }, fields, fieldFilter = {}, spaceFilter = {}, spaceTiles = new Map(), pin, queued = false, dead = false, drag, calls = 0;
  const emit = (type, value) => { for (const cb of events.get(type) || []) cb(value); };
  const geographic = () => ({ ...unproject(camera.x * 256, camera.y * 256, 0), zoom: camera.zoom });
  const point = p => { const [x, y] = project(p.lat, p.lng, camera.zoom), n = 256 * 2 ** camera.zoom; return { x: x - camera.x * n + canvas.clientWidth / 2, y: y - camera.y * n + canvas.clientHeight / 2 }; };
  const inverse = p => { const n = 256 * 2 ** camera.zoom; return unproject(camera.x * n + p.x - canvas.clientWidth / 2, camera.y * n + p.y - canvas.clientHeight / 2, camera.zoom); };
  const fieldVisible = i => !!(fieldFilter.statusMask & 1 << fields.status[i]) && (fieldFilter.year == null || (!fields.start[i] && !fields.end[i]) || ((!fields.start[i] || fieldFilter.year >= fields.start[i]) && (fields.status[i] === 0 || !fields.end[i] || fieldFilter.year <= fields.end[i])));
  function drawTile(tile, opacity) {
    const n = 256 * 2 ** camera.zoom, scale = 2 ** (camera.zoom - tile.dst.z), size = 256 * scale;
    const x = tile.dst.x * size - camera.x * n + canvas.clientWidth / 2, y = tile.dst.y * size - camera.y * n + canvas.clientHeight / 2;
    ctx.globalAlpha = opacity;
    for (const item of tile.items) {
      const texture = textures.get(item.key); if (!texture) continue;
      const ratio = 2 ** (item.dst.z - item.src.z), sw = texture.width / ratio, sh = texture.height / ratio;
      ctx.drawImage(texture, (item.dst.x % ratio) * sw, (item.dst.y % ratio) * sh, sw, sh, x, y, size, size); calls++;
    }
    ctx.strokeStyle = '#71858a'; ctx.strokeRect(x, y, size, size);
  }
  function draw() {
    queued = false; if (dead) return;
    const ratio = devicePixelRatio || 1; canvas.width = Math.round(canvas.clientWidth * ratio); canvas.height = Math.round(canvas.clientHeight * ratio); ctx.scale(ratio, ratio);
    ctx.fillStyle = '#12212a'; ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight); calls = 0;
    for (const tile of base) drawTile(tile, 1);
    if (!style.hidden) for (const tile of plan.tiles) {
      const key = `${tile.dst.z}/${tile.dst.x}/${tile.dst.y}`;
      if (tile.items.every(i => textures.has(i.key))) completed.set(key, tile);
      const ready = completed.get(key); if (ready) drawTile(ready, style.opacity);
    }
    ctx.globalAlpha = 1;
    if (fields) for (let i = 0; i < fields.mx.length; i++) if (fieldVisible(i)) {
      const pos = point(unproject(fields.mx[i] * 256, fields.my[i] * 256, 0));
      ctx.beginPath(); ctx.arc(pos.x, pos.y, 7, 0, 7); ctx.fillStyle = fields.status[i] === 0 ? '#36a35d' : '#c23b2a'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.stroke();
    }
    for (const batch of spaceTiles.values()) for (let i = 0; i < batch.from.length; i++) {
      const code = batch.style[i], isE = [4, 6, 7].includes(code);
      if (!(spaceFilter.classMask & (isE ? 2 : 1)) || !(spaceFilter.regionMask & 1 << batch.rg[i]) || spaceFilter.day < batch.from[i] || spaceFilter.day >= batch.to[i]) continue;
      ctx.strokeStyle = isE ? '#ce58bf' : '#168fff'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath();
      for (let j = batch.starts[i]; j < batch.starts[i + 1]; j++) { const p = point(unproject(batch.positions[j * 2] * 256, batch.positions[j * 2 + 1] * 256, 0)); if (j === batch.starts[i]) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (pin) { const p = point(pin); ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, 7); ctx.strokeStyle = '#168fff'; ctx.lineWidth = 3; ctx.stroke(); }
  }
  const schedule = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };
  const setCamera = value => { camera = { ...value, x: ((value.x % 1) + 1) % 1, y: Math.max(0, Math.min(1, value.y)), zoom: Math.max(minZoom, Math.min(maxZoom, value.zoom)) }; schedule(); emit('move'); emit('moveend'); };
  const handlers = {
    pointerdown(e) { if (e.button !== 0) return; drag = { x: e.clientX, y: e.clientY, camera, moved: false }; canvas.parentElement.focus(); canvas.setPointerCapture(e.pointerId); },
    pointermove(e) { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) < 4) return; drag.moved = true; const n = 256 * 2 ** drag.camera.zoom; setCamera({ ...drag.camera, x: drag.camera.x - dx / n, y: drag.camera.y - dy / n }); },
    pointerup(e) { if (!drag) return; if (!drag.moved) { const rect = canvas.getBoundingClientRect(); emit('click', { ...inverse({ x: e.clientX - rect.left, y: e.clientY - rect.top }), clientX: e.clientX, clientY: e.clientY, picked: api.pick(e.clientX, e.clientY) }); } drag = null; },
    pointercancel() { drag = null; },
    wheel(e) { e.preventDefault(); setCamera({ ...camera, zoom: camera.zoom + Math.sign(-e.deltaY) }); }
  };
  for (const [type, handler] of Object.entries(handlers)) canvas.addEventListener(type, handler, { passive: false });
  const observer = new ResizeObserver(schedule); observer.observe(canvas);
  const api = {
    getCamera: () => camera, setCamera,
    fitBounds([w, s, e, n], { padding = 0, maxZoom: limit = maxZoom } = {}) { const a = project(n, w, 0), b = project(s, e, 0); setCamera({ x: (a[0] + b[0]) / 512, y: (a[1] + b[1]) / 512, zoom: Math.min(limit, Math.log2(Math.min((canvas.clientWidth - padding * 2) / (b[0] - a[0]), (canvas.clientHeight - padding * 2) / (b[1] - a[1])))) }); },
    flyTo({ lng, lat }, zoom) { const [x, y] = project(lat, lng, 0); setCamera({ x: x / 256, y: y / 256, zoom }); },
    project: point, unproject: inverse,
    visibleTiles(tileSize = 256) { const c = geographic(); const z = Math.max(0, Math.floor(c.zoom) - (tileSize === 512 ? 1 : 0)); return visibleTiles({ ...c, zoom: z }, canvas.clientWidth / 2 ** (c.zoom - z), canvas.clientHeight / 2 ** (c.zoom - z)).sort((a, b) => Math.hypot(a.screenX - canvas.clientWidth / 2, a.screenY - canvas.clientHeight / 2) - Math.hypot(b.screenX - canvas.clientWidth / 2, b.screenY - canvas.clientHeight / 2)).map(({ z, x, y }) => ({ z, x, y })); },
    hasTexture: key => textures.has(key),
    upload(key, bitmap) { // Keep a canvas copy so ownership is closed exactly as C7 requires.
      const copy = new OffscreenCanvas(bitmap.width, bitmap.height); copy.getContext('2d').drawImage(bitmap, 0, 0); bitmap.close(); textures.set(key, copy);
      const needed = new Set([...plan.tiles, ...base, ...completed.values()].flatMap(t => t.items.map(i => i.key)));
      for (const k of textures.keys()) if (textures.size > 160 && !needed.has(k)) { textures.delete(k); emit('evict', { key: k }); }
      schedule();
    },
    setChartPlan(value) { plan = value; const wanted = new Set(plan.tiles.map(t => `${t.dst.z}/${t.dst.x}/${t.dst.y}`)); for (const k of completed.keys()) if (!wanted.has(k)) completed.delete(k); schedule(); },
    setBasemapPlan(value) { base = value; schedule(); },
    setChartStyle(value) { style = value; schedule(); },
    setClipRing() {}, setAirfields(value) { fields = value; schedule(); },
    setAirfieldFilter(value) { fieldFilter = value; schedule(); }, setAirspaceTile(tileId, batch) { if (batch) spaceTiles.set(tileId, batch); else spaceTiles.delete(tileId); schedule(); }, setAirspaceFilter(value) { spaceFilter = value; schedule(); }, setPin(value) { pin = value; schedule(); },
    pick(clientX, clientY) { if (!fields) return null; const rect = canvas.getBoundingClientRect(); for (let i = 0; i < fields.mx.length; i++) if (fieldVisible(i)) { const p = point(unproject(fields.mx[i] * 256, fields.my[i] * 256, 0)); if (Math.hypot(p.x + rect.left - clientX, p.y + rect.top - clientY) < 14) return { kind: 'airfield', index: i }; } return null; },
    on(type, cb) { if (!events.has(type)) events.set(type, new Set()); events.get(type).add(cb); }, off(type, cb) { events.get(type)?.delete(cb); },
    stats() { return { textures: textures.size, textureBytes: textures.size * 262144, drawCalls: calls, frameMs: 0 }; }, resize: schedule,
    destroy() { dead = true; observer.disconnect(); textures.clear(); completed.clear(); events.clear(); for (const [type, handler] of Object.entries(handlers)) canvas.removeEventListener(type, handler); }
  };
  schedule(); return api;
}
