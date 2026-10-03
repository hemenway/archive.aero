import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeMVT } from '../mvt.js';
import { tile, field, block, varint, cat, text } from './mvt-fixture.mjs';

test('MVT hand-built polygon has signed deltas, closed rings and properties', () => {
  const bytes = tile([{ name: 'class', features: [{ properties: { cls: 'B', from: 20200101, ex: false, lo: -12 },
    paths: [[[10, 20], [30, 20], [10, 40], [10, 20]]] }] }]);
  const { layers } = decodeMVT(bytes);
  assert.equal(layers.length, 1);
  assert.equal(layers[0].extent, 4096);
  assert.equal(layers[0].version, 2);
  const f = layers[0].features[0];
  assert.equal(f.id, 1); assert.equal(f.type, 3);
  assert.deepEqual({ ...f.properties }, { cls: 'B', from: 20200101, ex: false, lo: -12 });
  assert.ok(f.geometry[0] instanceof Int32Array);
  assert.deepEqual([...f.geometry[0]], [10, 20, 30, 20, 10, 40, 10, 20]);
});

test('MVT supports multiple layers, unclosed lines, buffered coordinates and byte offsets', () => {
  const bytes = tile([{ name: 'efloor', extent: 256, features: [{ type: 2, id: 9007199254740993n,
    paths: [[[-32, 0], [100, 300]], [[10, 10], [20, 20]]], properties: { k: '700' } }] }, { name: 'empty', features: [] }]);
  const padded = new Uint8Array(bytes.length + 30); padded.set(bytes, 7);
  const { layers } = decodeMVT(padded.subarray(7, 7 + bytes.length));
  assert.equal(layers.length, 2);
  assert.equal(layers[0].features[0].id, 9007199254740993n);
  assert.deepEqual(layers[0].features[0].geometry.map(a => [...a]), [[-32, 0, 100, 300], [10, 10, 20, 20]]);
  assert.deepEqual(layers[1].features, []);
});

test('MVT all protobuf Value encodings and unknown fields', () => {
  const f32 = new Uint8Array(4), f64 = new Uint8Array(8);
  new DataView(f32.buffer).setFloat32(0, 1.25, true);
  new DataView(f64.buffer).setFloat64(0, -2.5, true);
  const vals = [block(1, text('abc')), field(2, 5, [...f32]), field(3, 1, [...f64]),
    field(4, 0, varint(-17)), field(5, 0, varint(4294967296n)), field(6, 0, varint(33)), field(7, 0, varint(1))];
  const keys = vals.map((_, i) => `v${i}`);
  const feat = cat(block(2, keys.flatMap((_, i) => cat(varint(i), varint(i)))), field(3, 0, varint(1)), block(4, [9, 0, 0]));
  const layer = cat(block(1, text('values')), block(2, feat), ...keys.map(k => block(3, text(k))), ...vals.map(v => block(4, v)));
  const bytes = new Uint8Array(cat(block(99, [1, 2, 3]), field(98, 5, [0, 0, 0, 0]), block(3, layer)));
  assert.deepEqual(Object.values(decodeMVT(bytes).layers[0].features[0].properties), ['abc', 1.25, -2.5, -17, 4294967296, -17, true]);
});

test('MVT rejects truncated bytes, invalid varints and malformed geometry', () => {
  assert.throws(() => decodeMVT(new Uint8Array([26, 10, 1])), /Truncated/);
  assert.throws(() => decodeMVT(new Uint8Array(new Array(11).fill(128))), /varint/);
  assert.throws(() => decodeMVT(new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 0])), /tag/);
  const malformed = rawCommands => tile([{ name: 'class', features: [{ rawCommands }] }]);
  assert.throws(() => decodeMVT(malformed([9, 0])), /Truncated/);
  assert.throws(() => decodeMVT(malformed([11, 0, 0])), /Unknown/);
  assert.throws(() => decodeMVT(malformed([9, 0, 0, 10, 20, 0])), /Unclosed/);
  assert.throws(() => decodeMVT(malformed([9, 0, 0, 23])), /ClosePath/);
  assert.throws(() => decodeMVT({}), /input/);
});
