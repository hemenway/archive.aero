import test from 'node:test';
import assert from 'node:assert/strict';
import {createDataPlane} from '../index.js';
import {DataCore} from '../core.js';
import {parseManifest} from '../manifest.js';

const era = (start = '2000-01-01', end = '2001-01-01') => ({
  k: `${start}_to_${end}`, h: '0123456789ab', z: [4, 4], b: null, c: null,
});
const raw = (options = {}) => ({version: 1, tileBase: 'https://fixture.test/t/', fileBase: 'https://fixture.test/', eras: [era()], ...options});
const chartTile = {z: 4, x: 2, y: 3};
const chartState = {date: '2000-06-01', chartTiles: [chartTile]};
const flush = async () => {for (let i = 0; i < 40; i++) await Promise.resolve();};
const bitmap = () => ({closed: false, close() {this.closed = true;}});

async function harness(t, manifest) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'createImageBitmap');
  const decodes = [], events = [], errors = [], fetches = [];
  Object.defineProperty(globalThis, 'createImageBitmap', {configurable: true, writable: true,
    value: blob => new Promise((resolve, reject) => decodes.push({blob, resolve, reject})),
  });
  const dp = await createDataPlane({worker: false, decodeBitmap: false, manifestUrl: 'https://fixture.test/manifest', fetch: async url => {
    if (String(url).endsWith('/manifest')) return Response.json(manifest);
    fetches.push(String(url));
    return new Response(String(url), {headers: {'content-type': 'image/png'}});
  }});
  dp.on('tile', event => events.push(event));
  dp.on('error', event => errors.push(event));
  t.after(() => {
    dp.destroy();
    if (descriptor) Object.defineProperty(globalThis, 'createImageBitmap', descriptor);
    else delete globalThis.createImageBitmap;
  });
  return {dp, decodes, events, errors, fetches};
}

test('fallback dedupes repeated demand until decode ack and eviction reuses encoded cache', async t => {
  const h = await harness(t, raw());
  h.dp.setDemand(chartState); await flush();
  h.dp.setDemand(chartState); await flush();
  h.dp.setDemand(chartState); await flush();
  assert.equal(h.decodes.length, 1);
  assert.equal(h.fetches.length, 1);
  assert.equal(h.dp.readiness(chartState.date, [chartTile]), 0);
  const first = bitmap(); h.decodes[0].resolve(first); await flush();
  assert.equal(h.events.length, 1);
  assert.equal(h.events[0].bitmap, first);
  assert.equal(h.dp.readiness(chartState.date, [chartTile]), 1);
  const key = h.events[0].key;
  h.dp.markEvicted(key);
  assert.equal(h.dp.readiness(chartState.date, [chartTile]), 0);
  h.dp.setDemand(chartState); await flush();
  assert.equal(h.decodes.length, 2);
  assert.equal(h.fetches.length, 1);
  h.decodes[1].resolve(bitmap()); await flush();
  assert.equal(h.events.length, 2);
  assert.equal(h.dp.readiness(chartState.date, [chartTile]), 1);
});

test('still-demanded fallback basemap survives a demand generation change', async t => {
  const h = await harness(t, raw({eras: [], basemap: {p: 'basemap/test.0123456789ab', z: [0, 0]}}));
  const state = {date: '2000-01-01', chartTiles: [], basemapTiles: [{z: 0, x: 0, y: 0}]};
  h.dp.setDemand(state); await flush();
  h.dp.setDemand(state); await flush();
  assert.equal(h.decodes.length, 1);
  const result = bitmap(); h.decodes[0].resolve(result); await flush();
  assert.equal(result.closed, false);
  assert.equal(h.events.length, 1);
  assert.match(h.events[0].key, /^basemap\//);
});

test('still-demanded fallback prefetch survives demand updates and buffers the next frame', async t => {
  const h = await harness(t, raw({eras: [era(), era('2001-01-01', '2002-01-01')]}));
  const state = {...chartState, scrub: {playing: true}};
  h.dp.setDemand(state); await flush();
  h.dp.setDemand(state); await flush();
  assert.equal(h.decodes.length, 2);
  const current = bitmap(), next = bitmap();
  h.decodes[0].resolve(current); h.decodes[1].resolve(next); await flush();
  assert.equal(h.events.length, 2);
  assert.equal(next.closed, false);
  assert.equal(h.dp.readiness('2001-01-01', [chartTile]), 1);
});

test('superseded fallback decode closes its bitmap and emits only the latest tile', async t => {
  const h = await harness(t, raw({eras: [], basemap: {p: 'basemap/test.0123456789ab', z: [1, 1]}}));
  const state = {date: '2000-01-01', chartTiles: [], basemapTiles: [{z: 1, x: 0, y: 0}]};
  h.dp.setDemand(state); await flush();
  h.dp.setDemand({...state, basemapTiles: [{z: 1, x: 1, y: 0}]}); await flush();
  assert.equal(h.decodes.length, 2);
  const stale = bitmap(), latest = bitmap();
  h.decodes[0].resolve(stale); h.decodes[1].resolve(latest); await flush();
  assert.equal(stale.closed, true);
  assert.equal(latest.closed, false);
  assert.equal(h.events.length, 1);
  assert.equal(h.events[0].bitmap, latest);
});

test('failed main decode releases the key so later demand retries cached bytes', async t => {
  const h = await harness(t, raw());
  h.dp.setDemand(chartState); await flush();
  h.decodes[0].reject(new Error('temporary decode failure')); await flush();
  assert.equal(h.errors.length, 1);
  assert.equal(h.dp.readiness(chartState.date, [chartTile]), 0);
  h.dp.setDemand(chartState); await flush();
  assert.equal(h.decodes.length, 2);
  assert.equal(h.fetches.length, 1);
  h.decodes[1].resolve(bitmap()); await flush();
  assert.equal(h.events.length, 1);
  assert.equal(h.dp.readiness(chartState.date, [chartTile]), 1);
});

test('transferring fallback bytes leaves the encoded LRU buffer owned by the core', async t => {
  const events = [];
  const core = new DataCore(parseManifest(raw()), async () => new Response(new Uint8Array([1, 2, 3])), {decodeBitmap: false}, (event, payload) => {
    if (event === 'encoded') events.push(payload);
  });
  t.after(() => core.destroy());
  core.setDemand(chartState); await flush();
  assert.equal(events.length, 1);
  const {key, bytes} = events[0], cached = core.scheduler.cache.get(key);
  assert.notEqual(bytes.buffer, cached.buffer);
  const transferred = structuredClone(bytes, {transfer: [bytes.buffer]});
  assert.equal(bytes.byteLength, 0);
  assert.deepEqual([...transferred], [1, 2, 3]);
  assert.deepEqual([...cached], [1, 2, 3]);
  assert.equal(core.stats().bytesCached, 3);
});
