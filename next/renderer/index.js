import { Camera, clamp, mercator, wrap } from './camera.js';
import { TexturePool } from './texture-pool.js';
import * as shader from './shaders.js';

export class RendererUnsupportedError extends Error {
  constructor(message = 'This viewer requires WebGL2.') { super(message); this.name = 'RendererUnsupportedError'; }
}
const tileId = t => `${t.z}/${t.x}/${t.y}`;
const pathOf = key => key.slice(0, key.lastIndexOf('/', key.lastIndexOf('/', key.lastIndexOf('/') - 1) - 1));
const easing = t => 1 - (1 - t) ** 3;
export function createRenderer(canvas, options = {}) { return new Renderer(canvas, options); }

class Renderer {
  constructor(canvas, options) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', { alpha: false, stencil: true, antialias: false, preserveDrawingBuffer: !!options.preserveDrawingBuffer });
    if (!this.gl) throw new RendererUnsupportedError();
    this.coarse = options.coarsePointer ?? matchMedia('(pointer: coarse)').matches;
    const ios = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    this.budget = options.maxTextureBytes ?? ((ios || (this.coarse && Math.min(screen.width, screen.height) <= 820)) ? 24 : 64) * 1024 * 1024;
    if (!Number.isFinite(this.budget) || this.budget < 256 * 256 * 4) throw new RangeError('maxTextureBytes must fit at least one 256px RGBA tile');
    this.camera = new Camera({ minZoom: options.minZoom ?? 4, maxZoom: options.maxZoom ?? 14, center: [0, 0], zoom: options.minZoom ?? 4 });
    this.events = new Map(); this.chart = new Map(); this.base = new Map(); this.rings = new Map(); this.airspace = new Map(); this.airspaceDraws = [];
    this.queue = []; this.queued = new Map(); this.style = { opacity: 1, hidden: false };
    this.fieldFilter = { year: 0, statusMask: 7 }; this.spaceFilter = { day: 0, classMask: 3, regionMask: 7 };
    this.frame = 0; this.animation = null; this.lost = false; this.dead = false; this.drawCalls = 0; this.frameMs = 0; this.renderEvent = { frameMs: 0 };
    this.chartsVisible = []; this.baseVisible = []; this.chartDraws = []; this.baseDraws = []; this.viewDirty = true; this.planDirty = true;
    this._drawBound = t => this._frame(t); this._emitBound = (e, p) => this._emit(e, p);
    this._initGL();
    this.abort = new AbortController();
    const listen = (name, fn) => canvas.addEventListener(name, fn, { signal: this.abort.signal });
    listen('webglcontextlost', e => {
      e.preventDefault(); this.lost = true; this.pool.clear(); cancelAnimationFrame(this.frame); this.frame = 0; this.animation = null;
      for (const q of this.queue) q.bitmap.close(); this.queue.length = 0; this.queued.clear();
      this._emit('contextlost');
    });
    listen('webglcontextrestored', () => {
      this.lost = false; this._initGL();
      for (const cell of this.chart.values()) cell.completed = null;
      for (const cell of this.base.values()) cell.completed = null;
      for (const r of this.rings.values()) this._ringGPU(r);
      if (this.fields) this._fieldsGPU();
      for (const b of this.airspace.values()) this._lineGPU(b);
      this.planDirty = true; this._emit('contextrestored'); this._invalidate();
    });
    canvas.style.touchAction = 'none'; if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
    this._input(listen);
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(canvas);
    window.addEventListener('resize', () => this.resize(), { signal: this.abort.signal }); this.resize();
  }
  on(event, fn) { const listeners = this.events.get(event) ?? []; if (!listeners.includes(fn)) this.events.set(event, [...listeners, fn]); return this; }
  off(event, fn) { const listeners = this.events.get(event); if (listeners) this.events.set(event, listeners.filter(listener => listener !== fn)); return this; }
  _emit(event, payload) { const listeners = this.events.get(event); if (listeners) for (let i = 0; i < listeners.length; i++) listeners[i](payload); }
  getCamera() { return { x: this.camera.x, y: this.camera.y, zoom: this.camera.zoom }; }
  setCamera({ x = this.camera.x, y = this.camera.y, zoom = this.camera.zoom }, { animate = false } = {}) {
    if (![x, y, zoom].every(Number.isFinite)) throw new TypeError('Invalid camera');
    const target = { x, y: clamp(y, 0, 1), zoom: clamp(zoom, this.camera.minZoom, this.camera.maxZoom) };
    if (animate) this._animate(target, 250);
    else { this.animation = null; Object.assign(this.camera, target); this._moved(); this._emit('moveend', this.getCamera()); }
  }
  fitBounds(bounds, { padding = 0, maxZoom = this.camera.maxZoom } = {}) {
    const view = this.camera.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], padding);
    const p = mercator(...view.center); this.setCamera({ x: p[0], y: p[1], zoom: Math.min(maxZoom, view.zoom) });
  }
  flyTo({ lng, lat }, zoom) {
    const p = mercator(lng, lat); p[0] += Math.round(this.camera.x - p[0]);
    this._animate({ x: p[0], y: p[1], zoom: clamp(zoom, this.camera.minZoom, this.camera.maxZoom) }, 1500);
  }
  project({ lng, lat }) { const p = this.camera.project([lng, lat]); return { x: p[0], y: p[1] }; }
  unproject({ x, y }) { const p = this.camera.unproject([x, y]); return { lng: p[0], lat: p[1] }; }
  visibleTiles(tileSize = 256) {
    this._view();
    // Public coordinates are canonical; duplicate world copies are drawn internally.
    const source = tileSize === 512 ? this.baseVisible : this.chartsVisible, seen = new Set(), out = [];
    for (const t of source) if (!seen.has(t.key)) { seen.add(t.key); out.push({ z: t.z, x: t.x, y: t.y }); }
    return out;
  }
  hasTexture(key) { return this.pool.entries.has(key); }
  upload(key, bitmap) {
    if (this.dead || this.lost) { bitmap.close(); return; }
    const size = bitmap.width;
    if ((size !== 256 && size !== 512) || bitmap.height !== size) { bitmap.close(); throw new RangeError('Tiles must be 256 or 512 pixels square'); }
    if (this.hasTexture(key) || this.queued.has(key)) { bitmap.close(); return; }
    const q = { key, bitmap, size }; this.queue.push(q); this.queued.set(key, q); this._invalidate();
  }
  _makePlan(plans, previous) {
    const next = new Map();
    for (const plan of plans) {
      const id = tileId(plan.dst), cell = previous.get(id);
      next.set(id, { dst: plan.dst, items: plan.items, completed: cell?.completed ?? this._previousAncestor(plan.dst, previous) });
    }
    // Keep visible completed ancestors/children through a zoom transition.
    // Off-screen obsolete cells can be dropped; the pool caches their textures.
    for (const [id, cell] of previous) if (!next.has(id) && this._onScreen(cell.dst) && cell.completed?.some(item => this.hasTexture(item.key))) next.set(id, { dst: cell.dst, items: null, completed: cell.completed });
    return next;
  }
  _previousAncestor(dst, cells) {
    for (let z = dst.z - 1; z >= 0; z--) {
      const k = 2 ** (dst.z - z), c = cells.get(`${z}/${Math.floor(dst.x / k)}/${Math.floor(dst.y / k)}`);
      if (c?.completed) return c.completed;
    }
    return null;
  }
  setChartPlan(plan) { this.planId = plan.id; this.chart = this._makePlan(plan.tiles, this.chart); this.planDirty = true; this._invalidate(); }
  setBasemapPlan(tilePlans) { this.base = this._makePlan(tilePlans, this.base); this.planDirty = true; this._invalidate(); }
  setChartStyle(style) { Object.assign(this.style, style); this.style.opacity = clamp(this.style.opacity, 0, 1); this._invalidate(); }
  setClipRing(id, ring) {
    const old = this.rings.get(id); if (old && !this.lost) this.gl.deleteBuffer(old.buffer);
    if (!ring) this.rings.delete(id);
    else {
      const data = new Float32Array(ring.length * 2); let previous;
      for (let i = 0; i < ring.length; i++) {
        const p = mercator(...ring[i]); if (previous !== undefined) p[0] += Math.round(previous - p[0]);
        data[i * 2] = p[0]; data[i * 2 + 1] = p[1]; previous = p[0];
      }
      const r = { data, count: ring.length, x: data[0] }; this.rings.set(id, r); if (!this.lost) this._ringGPU(r);
    }
    this._invalidate();
  }
  setAirfields(arrays) {
    if (this.fieldVAO && !this.lost) { this.gl.deleteVertexArray(this.fieldVAO); this.gl.deleteBuffer(this.fieldBuffer); }
    this.fields = arrays; this.grid = new Map(); this.fieldVAO = null;
    if (arrays) {
      const n = arrays.mx.length, data = new Float32Array(n * 5);
      for (let i = 0; i < n; i++) {
        data.set([arrays.mx[i], arrays.my[i], arrays.start[i], arrays.end[i], arrays.status[i]], i * 5);
        const k = `${Math.floor(wrap(arrays.mx[i]) * 256)}/${clamp(Math.floor(arrays.my[i] * 256), 0, 255)}`;
        if (!this.grid.has(k)) this.grid.set(k, []); this.grid.get(k).push(i);
      }
      this.fieldData = data; if (!this.lost) this._fieldsGPU();
    }
    this._invalidate();
  }
  setAirfieldFilter({ year = this.fieldFilter.year, statusMask = this.fieldFilter.statusMask }) {
    this.fieldFilter.year = year ?? 0; this.fieldFilter.statusMask = statusMask; this._invalidate();
  }
  setAirspaceTile(tileId, batch) {
    const old = this.airspace.get(tileId);
    if (old && !this.lost) for (const group of old.groups) { this.gl.deleteBuffer(group.buffer); this.gl.deleteVertexArray(group.vao); }
    if (!batch) this.airspace.delete(tileId);
    else { const b = this._buildLines(batch); const [z, x, y] = tileId.split('/').map(Number); b.x = x / 2 ** z; b.y = y / 2 ** z; b.size = 1 / 2 ** z; this.airspace.set(tileId, b); if (!this.lost) this._lineGPU(b); }
    this.airspaceDraws = Array.from(this.airspace.values()); this._invalidate();
  }
  setAirspaceFilter(filter) { Object.assign(this.spaceFilter, filter); this._invalidate(); }
  setPin(lngLat) { this.pin = lngLat ? mercator(lngLat.lng, lngLat.lat) : null; this._invalidate(); }
  _fieldOn(i) {
    const a = this.fields, f = this.fieldFilter, s = a.status[i];
    return !!(f.statusMask & (1 << s)) && (!f.year || ((!a.start[i] || f.year >= a.start[i]) && (s === 0 || !a.end[i] || f.year <= a.end[i])));
  }
  _radius() { const z = this.camera.zoom; return (z >= 11 ? 8 : z >= 9 ? 7 : z >= 7 ? 5.5 : 4) + (this.coarse ? 1 : 0); }
  pick(clientX, clientY) {
    if (!this.fields) return null;
    const rect = this.canvas.getBoundingClientRect(), x = clientX - rect.left, y = clientY - rect.top;
    const c = this.camera, radius = this._radius() + 1 + (this.coarse ? 14 : 4), r = radius / c.world;
    const mx = c.x + (x - c.width / 2) / c.world, my = c.y + (y - c.height / 2) / c.world;
    let best = -1, bestD = radius * radius;
    const left = Math.floor((mx - r) * 256), right = Math.min(left + 256, Math.floor((mx + r) * 256));
    for (let gy = Math.max(0, Math.floor((my - r) * 256)); gy <= Math.min(255, Math.floor((my + r) * 256)); gy++) {
      for (let gx = left; gx <= right; gx++) {
        const list = this.grid.get(`${((gx % 256) + 256) % 256}/${gy}`); if (!list) continue;
        for (const i of list) if (this._fieldOn(i)) {
          const fx = this.fields.mx[i] + Math.round(mx - this.fields.mx[i]);
          const dx = (fx - mx) * c.world, dy = (this.fields.my[i] - my) * c.world, d = dx * dx + dy * dy;
          if (d <= bestD) { bestD = d; best = i; }
        }
      }
    }
    return best < 0 ? null : { kind: 'airfield', index: best };
  }
  stats() { return { textures: this.pool.entries.size, textureBytes: this.pool.bytes, drawCalls: this.drawCalls, frameMs: this.frameMs }; }
  resize() {
    if (this.dead) return;
    const r = this.canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    const width = Math.max(1, Math.round(r.width * dpr)), height = Math.max(1, Math.round(r.height * dpr));
    if (this.dpr === dpr && this.camera.width === Math.max(1, r.width) && this.camera.height === Math.max(1, r.height) && this.canvas.width === width && this.canvas.height === height) return;
    this.dpr = dpr;
    this.camera.width = Math.max(1, r.width); this.camera.height = Math.max(1, r.height);
    this.canvas.width = width; this.canvas.height = height;
    this._moved();
  }
  destroy() {
    this.dead = true; cancelAnimationFrame(this.frame); clearTimeout(this.wheelTimer);
    this.observer.disconnect(); this.abort.abort(); this.pool.clear();
    for (const q of this.queue) q.bitmap.close(); this.queue.length = 0; this.queued.clear();
    const gl = this.gl;
    for (const p of Object.values(this.programs)) gl.deleteProgram(p.program);
    for (const r of this.rings.values()) gl.deleteBuffer(r.buffer);
    for (const b of this.airspace.values()) for (const g of b.groups) { gl.deleteBuffer(g.buffer); gl.deleteVertexArray(g.vao); }
    gl.deleteBuffer(this.fieldBuffer); gl.deleteVertexArray(this.fieldVAO); gl.deleteVertexArray(this.emptyVAO);
    this.events.clear();
  }
  _initGL() {
    const gl = this.gl; this.pool = new TexturePool(gl, this.budget, this._emitBound);
    this.programs = {};
    for (const name of ['tile', 'ring', 'field', 'line', 'pin']) {
      const p = gl.createProgram();
      for (const [type, source] of [[gl.VERTEX_SHADER, shader[`${name}Vertex`]], [gl.FRAGMENT_SHADER, shader[`${name}Fragment`]]]) {
        const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s); gl.deleteShader(s);
      }
      gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {};
      for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const n = gl.getActiveUniform(p, i).name; u[n] = gl.getUniformLocation(p, n); }
      this.programs[name] = { program: p, u };
    }
    this.emptyVAO = gl.createVertexArray(); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.activeTexture(gl.TEXTURE0);
  }
  _ringGPU(r) { const gl = this.gl; r.buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, r.buffer); gl.bufferData(gl.ARRAY_BUFFER, r.data, gl.STATIC_DRAW); }
  _fieldsGPU() {
    const gl = this.gl; this.fieldVAO = gl.createVertexArray(); gl.bindVertexArray(this.fieldVAO);
    this.fieldBuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.fieldBuffer); gl.bufferData(gl.ARRAY_BUFFER, this.fieldData, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 20, 0); gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 20, 8); gl.vertexAttribDivisor(1, 1);
  }
  _buildLines(batch) {
    const groups = [[], []], pos = batch.positions;
    for (let l = 0; l < batch.style.length; l++) {
      const start = batch.starts[l], end = batch.starts[l + 1], code = batch.style[l], data = groups[code === 6 ? 0 : 1];
      let distance = 0;
      const vertex = (i, side, d) => {
        const prev = Math.max(start, i - 1), next = Math.min(end - 1, i + 1);
        data.push(pos[i * 2], pos[i * 2 + 1], pos[prev * 2], pos[prev * 2 + 1], pos[next * 2], pos[next * 2 + 1], side, d,
          batch.from[l], batch.to[l], code === 4 || code >= 6 ? 2 : 1, 1 << batch.rg[l], code >= 6 ? 6 : code, code === 6 ? 700 : 1200);
      };
      for (let i = start; i < end - 1; i++) {
        const nextD = distance + Math.hypot(pos[(i + 1) * 2] - pos[i * 2], pos[(i + 1) * 2 + 1] - pos[i * 2 + 1]);
        vertex(i, -1, distance); vertex(i, 1, distance); vertex(i + 1, -1, nextD);
        vertex(i + 1, -1, nextD); vertex(i, 1, distance); vertex(i + 1, 1, nextD); distance = nextD;
      }
    }
    return { groups: groups.map(data => ({ data: new Float32Array(data), count: data.length / 14 })) };
  }
  _lineGPU(b) {
    const gl = this.gl;
    for (const group of b.groups) {
      if (!group.count) continue;
      group.vao = gl.createVertexArray(); gl.bindVertexArray(group.vao); group.buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, group.buffer); gl.bufferData(gl.ARRAY_BUFFER, group.data, gl.STATIC_DRAW);
      let offset = 0;
      for (let a = 0; a < 6; a++) { const size = a === 4 ? 4 : 2; gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, size, gl.FLOAT, false, 56, offset * 4); offset += size; }
    }
  }
  _view() {
    if (!this.viewDirty) return;
    this.planDirty = true;
    this.chartsVisible = this.camera.visibleTiles(256); this.baseVisible = this.camera.visibleTiles(512); this.viewDirty = false;
    this.firstWorld = Math.floor(this.camera.x - this.camera.width / this.camera.world / 2);
    this.lastWorld = Math.floor(this.camera.x + this.camera.width / this.camera.world / 2);
  }
  _resolve(item) {
    const path = pathOf(item.key), src = item.src;
    for (let z = src.z; z >= 0; z--) {
      const scale = 2 ** (src.z - z), x = Math.floor(src.x / scale), y = Math.floor(src.y / scale), key = z === src.z ? item.key : `${path}/${z}/${x}/${y}`;
      if (this.pool.entries.has(key)) return { key, src: { z, x, y }, clip: item.clip };
    }
    return null;
  }
  _onScreen(dst) {
    const c = this.camera, n = 2 ** dst.z, x = dst.x / n + .5 / n;
    const nearest = x + Math.round(c.x - x), half = .5 / n;
    return Math.abs(nearest - c.x) < c.width / c.world / 2 + half && Math.abs((dst.y + .5) / n - c.y) < c.height / c.world / 2 + half;
  }
  _backfill(tile, cells, draws) {
    const exact = cells.get(tile.key);
    if (exact?.completed) { draws.push({ tile, items: exact.completed }); return; }
    for (let z = tile.z - 1; z >= 0; z--) {
      const k = 2 ** (tile.z - z), cell = cells.get(`${z}/${Math.floor(tile.x / k)}/${Math.floor(tile.y / k)}`);
      if (cell?.completed && cell.completed.every(item => this.hasTexture(item.key))) { draws.push({ tile, items: cell.completed }); return; }
    }
    // While zooming out, old child tiles form the backfill mosaic until the
    // destination plan is ready. Prefer the coarsest completed children.
    for (const cell of cells.values()) {
      const dst = cell.dst, k = 2 ** (dst.z - tile.z);
      if (dst.z <= tile.z || !cell.completed || Math.floor(dst.x / k) !== tile.x || Math.floor(dst.y / k) !== tile.y) continue;
      let covered = false;
      for (let z = tile.z + 1; z < dst.z; z++) {
        const f = 2 ** (dst.z - z), parent = cells.get(`${z}/${Math.floor(dst.x / f)}/${Math.floor(dst.y / f)}`);
        if (parent?.completed) { covered = true; break; }
      }
      if (!covered) draws.push({ tile: { ...dst, worldX: dst.x + Math.floor(tile.worldX / 2 ** tile.z) * 2 ** dst.z }, items: cell.completed });
    }
  }
  _plans() {
    if (!this.planDirty && !this.viewDirty) return;
    this._view(); this.pool.pinned.clear(); this.chartDraws.length = 0; this.baseDraws.length = 0;
    for (const cells of [this.chart, this.base]) for (const cell of cells.values()) if (cell.items) {
      for (const item of cell.items) this.pool.pinned.add(item.key);
    }
    for (const [cells, tiles, draws] of [[this.chart, this.chartsVisible, this.chartDraws], [this.base, this.baseVisible, this.baseDraws]]) {
      for (const tile of tiles) {
        const cell = cells.get(tile.key);
        if (cell?.items) {
          const resolved = []; let ready = true;
          for (const item of cell.items) { const r = this._resolve(item); if (!r) { ready = false; break; } resolved.push(r); }
          if (ready) cell.completed = resolved;
        }
        if (tile.onScreen) this._backfill(tile, cells, draws);
        // Partial replacement ancestors are pinned until the complete swap.
        if (cell?.items) for (const item of cell.items) { const r = this._resolve(item); if (r) this.pool.pinned.add(r.key); }
      }
      for (const draw of draws) for (const item of draw.items) this.pool.pinned.add(item.key);
    }
    this.planDirty = false;
  }
  _use(name) {
    const gl = this.gl, p = this.programs[name], c = this.camera;
    gl.useProgram(p.program); gl.uniform2f(p.u.center, c.x, c.y); gl.uniform2f(p.u.viewport, c.width, c.height); gl.uniform1f(p.u.world, c.world); return p.u;
  }
  _drawTile(tile, item, opacity) {
    const gl = this.gl, e = this.pool.get(item.key); if (!e) return;
    const n = 2 ** tile.z, srcN = 2 ** item.src.z, k = n / srcN;
    const u = this._use('tile'); gl.bindVertexArray(this.emptyVAO);
    gl.uniform3f(u.tile, tile.worldX / n, tile.y / n, 1 / n);
    gl.uniform4f(u.uvRect, (tile.x - item.src.x * k) / k, (tile.y - item.src.y * k) / k, 1 / k, 1 / k);
    gl.uniform1f(u.layer, e.layer); gl.uniform1f(u.opacity, opacity); gl.bindTexture(gl.TEXTURE_2D_ARRAY, e.page.texture);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); this.drawCalls++;
  }
  _clip(id, worldOffset) {
    const r = this.rings.get(id); if (!r) return false;
    const gl = this.gl; gl.enable(gl.STENCIL_TEST); gl.stencilMask(255); gl.clearStencil(0); gl.clear(gl.STENCIL_BUFFER_BIT);
    gl.colorMask(false, false, false, false); gl.stencilFunc(gl.ALWAYS, 0, 255); gl.stencilOp(gl.KEEP, gl.KEEP, gl.INVERT);
    const u = this._use('ring'); gl.uniform1f(u.wrap, worldOffset); gl.bindVertexArray(this.emptyVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, r.buffer); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_FAN, 0, r.count); this.drawCalls++; gl.disableVertexAttribArray(0);
    gl.colorMask(true, true, true, true); gl.stencilFunc(gl.NOTEQUAL, 0, 255); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP); return true;
  }
  _drawPlans(draws, opacity) {
    const gl = this.gl;
    for (let d = 0; d < draws.length; d++) {
      const draw = draws[d], t = draw.tile;
      for (let i = 0; i < draw.items.length; i++) {
        const item = draw.items[i];
        const r = this.rings.get(item.clip);
        const clipped = item.clip ? this._clip(item.clip, Math.round(t.worldX / 2 ** t.z - (r?.x ?? 0))) : false;
        this._drawTile(t, item, opacity); if (clipped) gl.disable(gl.STENCIL_TEST);
      }
    }
  }
  _drawOverlays() {
    const gl = this.gl;
    if (this.airspace.size) {
      gl.enable(gl.SCISSOR_TEST);
      const u = this._use('line'); gl.uniform1f(u.zoom, this.camera.zoom); gl.uniform1f(u.day, this.spaceFilter.day);
      gl.uniform1ui(u.classMask, this.spaceFilter.classMask); gl.uniform1ui(u.regionMask, this.spaceFilter.regionMask);
      for (let w = this.firstWorld; w <= this.lastWorld; w++) {
        gl.uniform1f(u.wrap, w);
        for (let a = 0; a < this.airspaceDraws.length; a++) {
          const b = this.airspaceDraws[a], c = this.camera, left = ((b.x + w - c.x) * c.world + c.width / 2) * this.dpr;
          const top = ((b.y - c.y) * c.world + c.height / 2) * this.dpr, size = b.size * c.world * this.dpr;
          const x0 = Math.max(0, Math.round(left)), x1 = Math.min(this.canvas.width, Math.round(left + size));
          const y0 = Math.max(0, Math.round(top)), y1 = Math.min(this.canvas.height, Math.round(top + size));
          if (x1 <= x0 || y1 <= y0) continue;
          gl.scissor(x0, this.canvas.height - y1, x1 - x0, y1 - y0);
          for (let i = 0; i < b.groups.length; i++) {
            const group = b.groups[i]; if (!group.count) continue; gl.bindVertexArray(group.vao); gl.uniform1f(u.floorColor, i === 0 ? 700 : 1200);
            // Floors are drawn once; class boundaries draw casing then color.
            gl.uniform1i(u.pass, 0); gl.drawArrays(gl.TRIANGLES, 0, group.count); this.drawCalls++;
            gl.uniform1i(u.pass, 1);
            // Fragment shader suppresses floor geometry on this second pass.
            gl.drawArrays(gl.TRIANGLES, 0, group.count); this.drawCalls++;
          }
        }
      }
      gl.disable(gl.SCISSOR_TEST);
    }
    if (this.fields) {
      const u = this._use('field'); gl.bindVertexArray(this.fieldVAO); gl.uniform1f(u.radius, this._radius()); gl.uniform1f(u.year, this.fieldFilter.year); gl.uniform1ui(u.statusMask, this.fieldFilter.statusMask);
      for (let w = this.firstWorld; w <= this.lastWorld; w++) { gl.uniform1f(u.wrap, w); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.fields.mx.length); this.drawCalls++; }
    }
    if (this.pin) {
      const u = this._use('pin'); gl.bindVertexArray(this.emptyVAO); gl.uniform2f(u.position, this.pin[0] + Math.round(this.camera.x - this.pin[0]), this.pin[1]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); this.drawCalls++;
    }
  }
  _frame(time) {
    this.frame = 0; if (this.lost || this.dead) return;
    const start = performance.now();
    if (this.animation) {
      const a = this.animation, t = clamp((time - a.time) / a.duration, 0, 1), k = easing(t);
      if (a.anchor) this.camera.zoomAt(a.from.zoom + (a.to.zoom - a.from.zoom) * k, a.anchor[0], a.anchor[1]);
      else { this.camera.x = a.from.x + (a.to.x - a.from.x) * k; this.camera.y = a.from.y + (a.to.y - a.from.y) * k; this.camera.zoom = a.from.zoom + (a.to.zoom - a.from.zoom) * k; }
      this.viewDirty = true; this.planDirty = true; this._emit('move', this.getCamera());
      if (t === 1) { this.animation = null; this._emit('moveend', this.getCamera()); }
    }
    this._plans();
    let uploaded = 0, blocked = false;
    while (this.queue.length && uploaded < 4 && performance.now() - start < 4) {
      const q = this.queue[0], entry = this.pool.put(q.key, q.bitmap, q.size);
      if (!entry) { blocked = true; this._emit('texturepressure', { key: q.key, maxTextureBytes: this.budget }); break; }
      q.bitmap.close(); this.queue.shift(); this.queued.delete(q.key); uploaded++; this.planDirty = true; this._plans();
    }
    const gl = this.gl; gl.viewport(0, 0, this.canvas.width, this.canvas.height); gl.clearColor(.035, .047, .063, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT); this.drawCalls = 0;
    this._drawPlans(this.baseDraws, 1);
    if (!this.style.hidden) this._drawPlans(this.chartDraws, this.style.opacity);
    this._drawOverlays(); this.frameMs = performance.now() - start;
    this.renderEvent.frameMs = this.frameMs; this._emit('render', this.renderEvent);
    if (this.animation || (this.queue.length && !blocked)) this._invalidate();
  }
  _invalidate() { if (!this.frame && !this.lost && !this.dead) this.frame = requestAnimationFrame(this._drawBound); }
  _moved() { this.viewDirty = true; this.planDirty = true; this._emit('move', this.getCamera()); this._invalidate(); }
  _animate(to, duration, anchor) { this.animation = { from: this.getCamera(), to, duration, anchor, time: performance.now() }; this._invalidate(); }
  _zoom(zoom, x = this.camera.width / 2, y = this.camera.height / 2) {
    this._animate({ zoom: clamp(Math.round(zoom), this.camera.minZoom, this.camera.maxZoom) }, 220, [x, y]);
  }
  _input(listen) {
    const pointers = new Map(); let lastTime = 0, vx = 0, vy = 0, moved = false, downX = 0, downY = 0, tapTime = 0, tapX = 0, tapY = 0;
    const local = e => { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    listen('pointerdown', e => {
      if (e.button !== 0) return; this.canvas.focus({ preventScroll: true }); this.canvas.setPointerCapture(e.pointerId); this.animation = null;
      pointers.set(e.pointerId, local(e)); downX = e.clientX; downY = e.clientY; lastTime = e.timeStamp; vx = vy = 0; moved = false;
    });
    listen('pointermove', e => {
      const old = pointers.get(e.pointerId); if (!old) return;
      const p = local(e), dt = Math.max(1, e.timeStamp - lastTime);
      if (pointers.size === 2) {
        const other = Array.from(pointers).find(([id]) => id !== e.pointerId)[1];
        const d0 = Math.hypot(old[0] - other[0], old[1] - other[1]), d1 = Math.hypot(p[0] - other[0], p[1] - other[1]);
        const cx = (old[0] + other[0]) / 2, cy = (old[1] + other[1]) / 2;
        if (d0 > 0 && d1 > 0) this.camera.zoomAt(this.camera.zoom + Math.log2(d1 / d0), cx, cy);
        this.camera.pan((p[0] - old[0]) / 2, (p[1] - old[1]) / 2); moved = true;
      } else {
        const dx = p[0] - old[0], dy = p[1] - old[1]; this.camera.pan(dx, dy); vx = .6 * vx + .4 * dx / dt; vy = .6 * vy + .4 * dy / dt;
        moved ||= Math.hypot(e.clientX - downX, e.clientY - downY) > 4;
      }
      pointers.set(e.pointerId, p); lastTime = e.timeStamp; this._moved();
    });
    const finish = (e, cancelled) => {
      const p = pointers.get(e.pointerId); if (!p) return; const pinching = pointers.size > 1; pointers.delete(e.pointerId);
      if (pinching) { moved = true; vx = vy = 0; return; }
      if (pointers.size) return;
      if (!cancelled && !moved) {
        const ll = this.unproject({ x: p[0], y: p[1] });
        this._emit('click', { ...ll, clientX: e.clientX, clientY: e.clientY, picked: this.pick(e.clientX, e.clientY) });
        if (e.pointerType !== 'mouse' && e.timeStamp - tapTime < 300 && Math.hypot(e.clientX - tapX, e.clientY - tapY) < 25) { this._zoom(this.camera.zoom + 1, ...p); tapTime = 0; }
        else { tapTime = e.timeStamp; tapX = e.clientX; tapY = e.clientY; }
      } else if (Math.abs(this.camera.zoom - Math.round(this.camera.zoom)) > .001) this._zoom(this.camera.zoom);
      else if (!cancelled && e.timeStamp - lastTime < 80 && Math.hypot(vx, vy) > .05) {
        const c = this.camera, d = 180; this._animate({ x: c.x - vx * d / c.world, y: clamp(c.y - vy * d / c.world, 0, 1), zoom: c.zoom }, 500);
      } else this._emit('moveend', this.getCamera());
    };
    listen('pointerup', e => finish(e, false)); listen('pointercancel', e => finish(e, true)); listen('lostpointercapture', e => finish(e, true));
    listen('dblclick', e => { e.preventDefault(); const p = local(e); this._zoom(this.camera.zoom + 1, ...p); });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault(); this.animation = null;
      const p = local(e), delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.camera.height : 1);
      this.camera.zoomAt(this.camera.zoom - delta / 120, ...p); this._moved(); clearTimeout(this.wheelTimer);
      this.wheelTimer = setTimeout(() => this._zoom(this.camera.zoom, ...p), 60);
    }, { passive: false, signal: this.abort.signal });
    listen('keydown', e => {
      const pan = { ArrowLeft: [80, 0], ArrowRight: [-80, 0], ArrowUp: [0, 80], ArrowDown: [0, -80] }[e.key];
      if (pan) { e.preventDefault(); this.animation = null; this.camera.pan(...pan); this._moved(); this._emit('moveend', this.getCamera()); }
      else if (['+', '=', '-', '_'].includes(e.key)) { e.preventDefault(); this._zoom((this.animation?.to.zoom ?? this.camera.zoom) + (e.key === '+' || e.key === '=' ? 1 : -1)); }
    });
  }
}
