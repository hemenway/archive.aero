// Hand-built protobuf fixtures, independent of the decoder implementation.
export function varint(input) {
  let n = BigInt.asUintN(64, BigInt(input));
  const bytes = [];
  do { const b = Number(n & 127n); n >>= 7n; bytes.push(b | (n ? 128 : 0)); } while (n);
  return bytes;
}
export const cat = (...chunks) => chunks.flat();
export const field = (number, wire, bytes) => [...varint(number * 8 + wire), ...bytes];
export const block = (number, bytes) => field(number, 2, [...varint(bytes.length), ...bytes]);
export const text = value => [...new TextEncoder().encode(value)];
export function mvtValue(value) {
  if (typeof value === 'string') return block(1, text(value));
  if (typeof value === 'boolean') return field(7, 0, varint(value ? 1 : 0));
  return field(value < 0 ? 4 : 5, 0, varint(value));
}
export function commands(paths, polygon = true) {
  let x = 0, y = 0;
  const out = [], zig = n => n < 0 ? -n * 2 - 1 : n * 2;
  for (const points of paths) {
    out.push(9);
    const first = points[0];
    out.push(zig(first[0] - x), zig(first[1] - y)); x = first[0]; y = first[1];
    const tail = polygon && points.at(-1)[0] === x && points.at(-1)[1] === y ? points.slice(1, -1) : points.slice(1);
    if (tail.length) out.push(tail.length * 8 + 2);
    for (const point of tail) { out.push(zig(point[0] - x), zig(point[1] - y)); x = point[0]; y = point[1]; }
    if (polygon) out.push(15);
  }
  return out;
}
export function tile(layers) {
  return new Uint8Array(layers.flatMap(({ name, features, extent = 4096 }) => {
    const keys = [], values = [], feats = [];
    for (const { properties = {}, paths = [], type = 3, id = 1, rawCommands } of features) {
      const tags = [];
      for (const [k, v] of Object.entries(properties)) {
        let key = keys.indexOf(k); if (key < 0) { key = keys.length; keys.push(k); }
        const val = values.length; values.push(v); tags.push(key, val);
      }
      feats.push(block(2, cat(field(1, 0, varint(id)), block(2, tags.flatMap(varint)),
        field(3, 0, varint(type)), block(4, (rawCommands || commands(paths, type === 3)).flatMap(varint)))));
    }
    // Features precede their string/value tables, as the MVT format permits.
    const layer = cat(block(1, text(name)), ...feats, ...keys.map(k => block(3, text(k))),
      ...values.map(v => block(4, mvtValue(v))), field(5, 0, varint(extent)), field(15, 0, varint(2)));
    return block(3, layer);
  }));
}
