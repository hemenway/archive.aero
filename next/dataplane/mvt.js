// A deliberately small protobuf/MVT decoder. Geometry stays in integer tile
// coordinates: no projection, DOM, gzip library, or third-party dependency.
const utf8 = new TextDecoder('utf-8', { fatal: true });
const safeNumber = n => n <= BigInt(Number.MAX_SAFE_INTEGER) && n >= BigInt(Number.MIN_SAFE_INTEGER) ? Number(n) : n;

class Reader {
  constructor(bytes) { this.bytes = bytes; this.pos = 0; this.end = bytes.length; this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); }
  need(n) { if (!Number.isSafeInteger(n) || n < 0 || n > this.end - this.pos) throw new Error('Truncated MVT protobuf'); }
  uint64() {
    let n = 0n;
    for (let i = 0; i < 10; i++) {
      this.need(1);
      const b = this.bytes[this.pos++];
      if (i === 9 && b > 1) throw new Error('Invalid protobuf varint');
      n |= BigInt(b & 127) << BigInt(i * 7);
      if (!(b & 128)) return n;
    }
    throw new Error('Invalid protobuf varint');
  }
  uint() {
    const n = this.uint64();
    if (n > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Unsafe protobuf integer');
    return Number(n);
  }
  block() { const n = this.uint(); this.need(n); const b = this.bytes.subarray(this.pos, this.pos + n); this.pos += n; return b; }
  float(n) { this.need(n); const value = n === 4 ? this.view.getFloat32(this.pos, true) : this.view.getFloat64(this.pos, true); this.pos += n; return value; }
  skip(wire) {
    if (wire === 0) this.uint64();
    else if (wire === 1 || wire === 5) { const n = wire === 1 ? 8 : 4; this.need(n); this.pos += n; }
    else if (wire === 2) this.block();
    else throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
  tag() {
    const tag = this.uint();
    if (tag < 8 || tag > 0xffffffff) throw new Error('Invalid protobuf tag');
    return [Math.floor(tag / 8), tag & 7];
  }
}

function value(bytes) {
  const r = new Reader(bytes);
  let out = null;
  while (r.pos < r.end) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 2) out = utf8.decode(r.block());
    else if (field === 2 && wire === 5) out = r.float(4);
    else if (field === 3 && wire === 1) out = r.float(8);
    else if (field === 4 && wire === 0) out = safeNumber(BigInt.asIntN(64, r.uint64()));
    else if (field === 5 && wire === 0) out = safeNumber(r.uint64());
    else if (field === 6 && wire === 0) { const n = r.uint64(); out = safeNumber((n >> 1n) ^ -(n & 1n)); }
    else if (field === 7 && wire === 0) out = r.uint64() !== 0n;
    else r.skip(wire);
  }
  return out;
}

function packed(bytes) {
  const r = new Reader(bytes), out = [];
  while (r.pos < r.end) out.push(r.uint());
  return out;
}

function geometry(commands, type) {
  if (type === 0) return [];
  if (type < 1 || type > 3) throw new Error('Invalid MVT geometry type');
  let x = 0, y = 0, cursor = 0, path = null;
  const paths = [];
  const point = () => {
    if (cursor + 2 > commands.length) throw new Error('Truncated MVT geometry');
    const a = commands[cursor++], b = commands[cursor++];
    if (a > 0xffffffff || b > 0xffffffff) throw new Error('Invalid MVT coordinate');
    x += a % 2 ? -(a + 1) / 2 : a / 2;
    y += b % 2 ? -(b + 1) / 2 : b / 2;
    if (x < -2147483648 || x > 2147483647 || y < -2147483648 || y > 2147483647) throw new Error('MVT coordinate overflow');
    return [x, y];
  };
  while (cursor < commands.length) {
    const command = commands[cursor++], id = command % 8, count = Math.floor(command / 8);
    if (!count || command > 0xffffffff) throw new Error('Invalid MVT geometry command');
    if (id === 1) {
      if (count > (commands.length - cursor) / 2) throw new Error('Truncated MVT geometry');
      if (type !== 1 && count !== 1) throw new Error('Invalid MVT MoveTo count');
      for (let i = 0; i < count; i++) { path = point(); paths.push(path); }
    } else if (id === 2) {
      if (!path || type === 1 || count > (commands.length - cursor) / 2) throw new Error('Invalid MVT LineTo');
      for (let i = 0; i < count; i++) path.push(...point());
    } else if (id === 7) {
      if (!path || type !== 3 || count !== 1 || path.length < 6) throw new Error('Invalid MVT ClosePath');
      if (path[0] !== path[path.length - 2] || path[1] !== path[path.length - 1]) path.push(path[0], path[1]);
      path = null;
    } else throw new Error(`Unknown MVT geometry command ${id}`);
  }
  if (type === 3 && path) throw new Error('Unclosed MVT polygon');
  return paths.map(p => new Int32Array(p));
}

function feature(bytes, keys, values) {
  const r = new Reader(bytes), tags = [], commands = [];
  let id = null, type = 0;
  while (r.pos < r.end) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 0) id = safeNumber(r.uint64());
    else if (field === 2 && wire === 2) { for (const n of packed(r.block())) tags.push(n); }
    else if (field === 2 && wire === 0) tags.push(r.uint());
    else if (field === 3 && wire === 0) type = r.uint();
    else if (field === 4 && wire === 2) {
      // Avoid spread's argument limit for large, valid polygon features.
      for (const n of packed(r.block())) commands.push(n);
    } else if (field === 4 && wire === 0) commands.push(r.uint());
    else r.skip(wire);
  }
  if (tags.length % 2) throw new Error('Odd MVT feature tag count');
  const properties = Object.create(null);
  for (let i = 0; i < tags.length; i += 2) {
    if (tags[i] >= keys.length || tags[i + 1] >= values.length) throw new Error('MVT tag index out of range');
    properties[keys[tags[i]]] = values[tags[i + 1]];
  }
  return { id, type, properties, geometry: geometry(commands, type) };
}

function layer(bytes) {
  const r = new Reader(bytes), keys = [], values = [], features = [];
  let name = null, extent = 4096, version = 1;
  while (r.pos < r.end) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 2) name = utf8.decode(r.block());
    else if (field === 2 && wire === 2) features.push(r.block());
    else if (field === 3 && wire === 2) keys.push(utf8.decode(r.block()));
    else if (field === 4 && wire === 2) values.push(value(r.block()));
    else if (field === 5 && wire === 0) extent = r.uint();
    else if (field === 15 && wire === 0) version = r.uint();
    else r.skip(wire);
  }
  if (!name || !Number.isSafeInteger(extent) || extent < 1 || extent > 0x7fffffff || ![1, 2].includes(version)) throw new Error('Invalid MVT layer');
  return { name, extent, version, features: features.map(b => feature(b, keys, values)) };
}

export function decodeMVT(input) {
  const bytes = input instanceof ArrayBuffer ? new Uint8Array(input)
    : ArrayBuffer.isView(input) ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength) : null;
  if (!bytes) throw new TypeError('MVT input must be an ArrayBuffer or typed-array view');
  const r = new Reader(bytes), layers = [];
  while (r.pos < r.end) {
    const [field, wire] = r.tag();
    if (field === 3 && wire === 2) layers.push(layer(r.block()));
    else r.skip(wire);
  }
  return { layers };
}
