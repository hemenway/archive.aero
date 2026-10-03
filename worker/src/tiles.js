// PMTiles v3 tile serving. All storage reads use index.js's existing block path.
// /t/ is reserved: enable this only after confirming no bucket key starts t/.
export const IMMUTABLE = 'public, max-age=31536000, immutable';
export const DIRECTORY_CACHE_BYTES = 16 * 1024 * 1024;
const MAX_DIRECTORY_BYTES = 8 * 1024 * 1024;
const cached = new Map();
let cacheBytes = 0;
export function clearDirectoryCache() { cached.clear(); cacheBytes = 0; }
export function directoryCacheStats() { return { bytes: cacheBytes, entries: cached.size, limit: DIRECTORY_CACHE_BYTES }; }
function recall(key) {
  const entry = cached.get(key);
  if (entry) { cached.delete(key); cached.set(key, entry); }
  return entry?.value;
}
function remember(key, value, size) {
  size += key.length * 2 + 128;
  if (size > DIRECTORY_CACHE_BYTES) return value;
  const old = cached.get(key); if (old) { cacheBytes -= old.size; cached.delete(key); }
  while (cacheBytes + size > DIRECTORY_CACHE_BYTES) {
    const first = cached.keys().next().value; cacheBytes -= cached.get(first).size; cached.delete(first);
  }
  cached.set(key, { value, size }); cacheBytes += size; return value;
}
export function tileId(z, x, y) {
  let id = (4 ** z - 1) / 3;
  for (let s = 2 ** (z - 1); s >= 1; s /= 2) {
    const rx = (Math.floor(x / s) % 2 + 2) % 2, ry = (Math.floor(y / s) % 2 + 2) % 2;
    id += s * s * ((3 * rx) ^ ry);
    if (ry === 0) { if (rx === 1) { x = s - 1 - x; y = s - 1 - y; } [x, y] = [y, x]; }
  }
  return id;
}
export function parseHeader(bytes, size) {
  if (bytes.length !== 127 || new TextDecoder().decode(bytes.subarray(0, 7)) !== 'PMTiles' || bytes[7] !== 3) throw Error('invalid PMTiles v3 header');
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u64 = off => { const n = Number(v.getBigUint64(off, true)); if (!Number.isSafeInteger(n)) throw Error('unsafe archive offset'); return n; };
  const h = { root: u64(8), rootLength: u64(16), metadata: u64(24), metadataLength: u64(32),
    leaves: u64(40), leavesLength: u64(48), tiles: u64(56), tilesLength: u64(64),
    compression: bytes[97], tileCompression: bytes[98], type: bytes[99], min: bytes[100], max: bytes[101] };
  for (const [off, len] of [[h.root,h.rootLength],[h.metadata,h.metadataLength],[h.leaves,h.leavesLength],[h.tiles,h.tilesLength]]) {
    if (off + len > size || !Number.isSafeInteger(off + len)) throw Error('archive section outside object');
  }
  if (![1,2].includes(h.compression) || ![1,2].includes(h.tileCompression)) throw Error('unsupported compression');
  return h;
}
async function unpack(bytes, compression) {
  if (bytes.length > MAX_DIRECTORY_BYTES) throw Error('compressed directory too large');
  if (compression === 1) return bytes;
  const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();
  const chunks = []; let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      total += value.length; if (total > MAX_DIRECTORY_BYTES) throw Error('decompressed directory too large'); chunks.push(value);
    }
  } finally { await reader.cancel(); }
  const result = new Uint8Array(total); let off = 0;
  for (const chunk of chunks) { result.set(chunk, off); off += chunk.length; }
  return result;
}
export function parseDirectory(bytes) {
  let pos = 0;
  function variable() {
    let result = 0;
    for (let shift = 0; shift < 56; shift += 7) {
      if (pos >= bytes.length) throw Error('truncated directory');
      const b = bytes[pos++]; result += (b & 127) * 2 ** shift;
      if (!Number.isSafeInteger(result)) throw Error('unsafe varint');
      if (!(b & 128)) return result;
    }
    throw Error('oversized varint');
  }
  const count = variable();
  if (count > bytes.length / 4 || count * 32 > MAX_DIRECTORY_BYTES) throw Error('oversized parsed directory');
  // 4 doubles per entry: tile id, offset, length, run. Precise through z24.
  const entries = new Float64Array(count * 4); let tid = 0;
  for (let i = 0; i < count; i++) { tid += variable(); if (!Number.isSafeInteger(tid)) throw Error('unsafe tile id'); entries[i*4] = tid; }
  for (let i = 0; i < count; i++) entries[i*4+3] = variable();
  for (let i = 0; i < count; i++) entries[i*4+2] = variable();
  for (let i = 0; i < count; i++) {
    const off = variable(); entries[i*4+1] = off === 0 && i > 0 ? entries[(i-1)*4+1]+entries[(i-1)*4+2] : off-1;
    if (entries[i*4+1] < 0 || entries[i*4+2] === 0) throw Error('invalid entry');
  }
  if (pos !== bytes.length) throw Error('trailing directory data');
  return entries;
}
function find(entries, id) {
  let lo = 0, hi = entries.length/4-1;
  while (lo <= hi) { const mid = Math.floor((lo+hi)/2); if (entries[mid*4] <= id) lo = mid+1; else hi = mid-1; }
  if (hi < 0) return null;
  const e = entries.subarray(hi*4, hi*4+4);
  return e[3] === 0 || id-e[0] < e[3] ? e : null;
}
const TYPES = { 1:'application/vnd.mapbox-vector-tile', 2:'image/png', 3:'image/jpeg', 4:'image/webp', 5:'image/avif' };
export async function handleTiles(request, env, ctx, { getBlock, blockBytes }) {
  const started = Date.now(); const url = new URL(request.url);
  const headers = new Headers({ 'access-control-allow-origin':'*',
    'access-control-allow-methods':'GET, HEAD, OPTIONS',
    'access-control-expose-headers':'Content-Length, Content-Encoding, ETag, X-Cache' });
  function response(status, body = null, extra = {}) {
    const h = new Headers(headers); for (const [k,v] of Object.entries(extra)) h.set(k,v);
    if (status === 200 || status === 204) h.set('cache-control', IMMUTABLE);
    else h.set('cache-control', status === 404 ? 'public, max-age=60' : 'no-store');
    return new Response(request.method === 'HEAD' ? null : body, { status, headers:h, encodeBody:'manual' });
  }
  let parts;
  try { parts = decodeURIComponent(url.pathname).slice(3).split('/'); } catch { return response(400); }
  const metadata = parts.at(-1) === 'metadata';
  let z, x, y;
  if (!metadata) {
    const coords = parts.splice(-3);
    if (coords.length !== 3 || coords.some(n => !/^(0|[1-9]\d*)$/.test(n))) return response(400);
    [z,x,y] = coords.map(Number);
    if (z > 24 || x >= 2**z || y >= 2**z || ![z,x,y].every(Number.isSafeInteger)) return response(400);
  } else parts.pop();
  const path = parts.join('/');
  if (!/^(?:[A-Za-z0-9_-]+\/)+[A-Za-z0-9_.-]+\.[a-f0-9]{12}$/.test(path) || parts.some(p => p === '.' || p === '..')) return response(404);
  const key = path+'.pmtiles'; const objectUrl = new URL('/'+key, url.origin);
  const identity = objectUrl.href; let source = 'HIT';
  // Reuse this request's blocks while waitUntil fills the edge entry. This
  // closes the settlement-to-cache-put gap without sharing request I/O.
  const localBlocks = new Map();
  function keepBlock(index, block) {
    if (localBlocks.size >= 2 && !localBlocks.has(index)) localBlocks.delete(localBlocks.keys().next().value);
    localBlocks.set(index, block); return block;
  }
  async function read(offset, length) {
    if (!Number.isSafeInteger(offset+length) || offset < 0 || length < 0) throw Error('invalid read');
    const out = new Uint8Array(length); let written = 0;
    while (written < length) {
      const index = Math.floor((offset+written)/blockBytes);
      const block = localBlocks.get(index) || keepBlock(index, await getBlock(env, ctx, objectUrl, key, index));
      if (!block) return null;
      if (block.source === 'MISS' || (source !== 'MISS' && block.source === 'COALESCE')) source = block.source;
      const from = (offset+written)%blockBytes, n = Math.min(length-written, block.buffer.byteLength-from);
      if (n <= 0) throw Error('short block');
      out.set(new Uint8Array(block.buffer, from, n), written); written += n;
    }
    return out;
  }
  async function dir(h, offset, length) {
    const id = identity+':dir:'+offset+':'+length;
    const hit = recall(id); if (hit) return hit;
    if (length > MAX_DIRECTORY_BYTES) throw Error('directory too large');
    const bytes = await read(offset,length); if (!bytes) throw Error('archive disappeared');
    const entries = parseDirectory(await unpack(bytes,h.compression));
    return remember(id, entries, entries.byteLength);
  }
  try {
    let h = recall(identity+':header');
    if (!h) {
      const first = await getBlock(env, ctx, objectUrl, key, 0);
      if (!first) return response(404);
      keepBlock(0,first); source = first.source; h = parseHeader(new Uint8Array(first.buffer).subarray(0,127),first.total);
      remember(identity+':header',h,512);
    }
    if (metadata) {
      const bytes = await read(h.metadata,h.metadataLength);
      const body = await unpack(bytes,h.compression); JSON.parse(new TextDecoder().decode(body));
      return response(200,body,{'content-type':'application/json','content-length':String(body.length),'x-cache':source});
    }
    if (z < h.min || z > h.max) return response(204);
    const id = tileId(z,x,y); let offset = h.root, length = h.rootLength;
    for (let depth = 0; depth < 4; depth++) {
      const e = find(await dir(h,offset,length),id);
      if (!e) return response(204);
      if (!e[3]) {
        if (e[1]+e[2] > h.leavesLength) throw Error('leaf outside section');
        offset = h.leaves+e[1]; length = e[2]; continue;
      }
      if (e[1]+e[2] > h.tilesLength || !TYPES[h.type]) throw Error('invalid tile');
      const extra = {'content-type':TYPES[h.type],'content-length':String(e[2]),'x-cache':source};
      if (h.tileCompression === 2) extra['content-encoding'] = 'gzip';
      const body = request.method === 'HEAD' ? null : await read(h.tiles+e[1],e[2]);
      extra['x-cache'] = source;
      if (env.TILES && request.method === 'GET' && Math.random() < Number(env.SAMPLE_RATE || '0.05')) {
        try { env.TILES.writeDataPoint({indexes:[key],blobs:[key,request.cf?.country||'XX',source,
          (request.headers.get('user-agent')||'').slice(0,256),request.headers.get('referer')||''],
          doubles:[e[2],Date.now()-started,h.tiles+e[1]]}); } catch {}
      }
      return response(200,body,extra);
    }
    throw Error('directory nesting exceeds v3 limit');
  } catch (err) {
    console.error(`Tile read failed for ${key}: ${String(err?.message || err)}`);
    return response(503,null,{'retry-after':'1'});
  }
}
