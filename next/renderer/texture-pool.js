// Lazy fixed-size array pages. Physical RGBA8 storage, not just resident images,
// counts toward the budget. Basemap and chart pages never share an array.
export class TexturePool {
  constructor(gl, budget, emit) {
    this.gl = gl; this.budget = budget; this.emit = emit;
    this.pages = []; this.entries = new Map(); this.pinned = new Set();
    this.bytes = 0; this.clock = 0; this.uploads = 0;
  }
  get(key) { const e = this.entries.get(key); if (e) e.used = ++this.clock; return e; }
  put(key, bitmap, size) {
    if (this.entries.has(key)) return this.get(key);
    const gl = this.gl, bytes = size * size * 4;
    let page, layer;
    for (const p of this.pages) if (p.size === size && p.free.length) { page = p; layer = p.free.pop(); break; }
    if (!page && this.bytes + bytes > this.budget && !Array.from(this.entries.values()).some(e => e.page.size === size && !this.pinned.has(e.key))) {
      // Repurpose pages when demand changes between chart and basemap sizes.
      // A page can be retired only if every resident layer is unpinned.
      while (this.bytes + bytes > this.budget) {
        let victimPage, oldest = Infinity;
        for (const p of this.pages) {
          let pinned = false, newest = 0;
          for (const e of this.entries.values()) if (e.page === p) { pinned ||= this.pinned.has(e.key); newest = Math.max(newest, e.used); }
          if (!pinned && newest < oldest) { victimPage = p; oldest = newest; }
        }
        if (!victimPage) break;
        for (const [k, e] of this.entries) if (e.page === victimPage) { this.entries.delete(k); this.emit('evict', { key: k }); }
        gl.deleteTexture(victimPage.texture); this.pages.splice(this.pages.indexOf(victimPage), 1);
        this.bytes -= victimPage.size * victimPage.size * 4 * victimPage.count;
      }
    }
    if (!page && this.bytes + bytes <= this.budget) {
      const count = Math.min(8, gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS), Math.floor((this.budget - this.bytes) / bytes));
      page = { size, free: [], texture: gl.createTexture(), count };
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, page.texture);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA8, size, size, count);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      for (let i = count - 1; i > 0; i--) page.free.push(i);
      layer = 0; this.pages.push(page); this.bytes += count * bytes;
    }
    if (!page) {
      let victim;
      for (const e of this.entries.values()) if (e.page.size === size && !this.pinned.has(e.key) && (!victim || e.used < victim.used)) victim = e;
      if (!victim) return null;
      page = victim.page; layer = victim.layer;
      this.entries.delete(victim.key); this.emit('evict', { key: victim.key });
    }
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, page.texture);
    // Decoded bitmaps arrive premultiplied (the data plane asks for it); this
    // flag only matters for the Image fallback source. The tile shader blends
    // with ONE, ONE_MINUS_SRC_ALPHA so filtered edges never darken.
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, size, size, 1, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    const e = { key, page, layer, used: ++this.clock };
    this.entries.set(key, e); this.uploads++; return e;
  }
  // Every resident key is reported evicted so the data plane re-requests it.
  clear() { for (const p of this.pages) this.gl.deleteTexture(p.texture); this.pages.length = 0; const keys = [...this.entries.keys()]; this.entries.clear(); this.pinned.clear(); this.bytes = 0; for (const key of keys) this.emit('evict', { key }); }
}
