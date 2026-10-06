import assert from 'node:assert/strict';
import { test } from 'node:test';
import worker from '../src/index.js';

const BLOCK = 1 << 20;
const data = Uint8Array.from({ length: BLOCK * 2 + 17 }, (_, i) => i % 251);

// HTTP contract tests with deterministic R2/Cache doubles. These are not a
// workerd emulator: deployed binding behavior still needs a staging smoke check.
function harness(t) {
  const entries = new Map();
  const reads = [];
  const waited = [];
  const state = { etag: '"v1"', fail: false, missing: false, cacheFail: false, heads: 0, hold: null };
  const metadata = () => ({
    size: data.length, httpEtag: state.etag,
    writeHttpMetadata(h) { h.set('content-type', 'application/octet-stream'); },
  });
  const env = { SAMPLE_RATE: '0', BUCKET: {
    async head() { state.heads++; if (state.fail) throw Error('storage offline'); return state.missing ? null : metadata(); },
    async get(key, { range } = {}) {
      reads.push({ key, range });
      // hold: a one-shot gate on the next read. A gate that never settles is a
      // leader whose request was cancelled mid-read (its I/O never completes);
      // one that resolves to 'fail' is a read that errors while joiners wait.
      const gate = state.hold;
      state.hold = null;
      if (gate && (await gate) === 'fail') throw Error('storage offline');
      if (state.fail) throw Error('storage offline');
      if (state.missing) return null;
      const offset = range?.offset ?? 0;
      if (offset >= data.length) throw Error('range not satisfiable');
      const bytes = data.slice(offset, offset + (range?.length ?? data.length));
      return { ...metadata(), body: new Response(bytes).body, arrayBuffer: async () => bytes.buffer };
    },
  } };
  t.mock.method(console, 'error', () => {});
  const previous = globalThis.caches;
  globalThis.caches = { default: {
    async match(req) { return entries.get(req.url)?.clone(); },
    async put(req, response) {
      if (state.cacheFail) throw Error('cache unavailable');
      assert.equal(response.status, 200, 'Cache API must not receive a 206');
      entries.set(req.url, new Response(await response.arrayBuffer(), response));
    },
  } };
  t.after(() => { globalThis.caches = previous; });
  async function request(headers = {}, method = 'GET', path = '/chart.pmtiles') {
    const pending = []; // waitUntil is per invocation
    const response = await worker.fetch(new Request('https://tiles.test' + path, { method, headers }), env,
      { waitUntil(p) { pending.push(p); waited.push(p); } });
    const body = new Uint8Array(await response.arrayBuffer());
    await Promise.all(pending);
    assert.equal(response.headers.get('access-control-allow-origin'), '*');
    return { response, body };
  }
  return { request, reads, entries, state, env, waited };
}

for (const [name, range, start, end] of [
  ['bounded', 'bytes=10-25', 10, 25],
  ['cross-block', `bytes=${BLOCK - 8}-${BLOCK + 8}`, BLOCK - 8, BLOCK + 8],
  ['suffix', 'bytes=-11', data.length - 11, data.length - 1],
  ['EOF clamp', `bytes=${data.length - 5}-${data.length + 20}`, data.length - 5, data.length - 1],
  ['open-ended', `bytes=${data.length - 11}-`, data.length - 11, data.length - 1],
  ['large streamed', `bytes=12-${BLOCK + 30}`, 12, BLOCK + 30],
]) {
  test(`${name} returns exact bytes and remains correct on cache hit`, async t => {
    const h = harness(t);
    for (let i = 0; i < 2; i++) {
      const { response, body } = await h.request({ range });
      assert.equal(response.status, 206);
      assert.equal(response.headers.get('content-range'), `bytes ${start}-${end}/${data.length}`);
      assert.equal(+response.headers.get('content-length'), body.length);
      assert.deepEqual(body, data.slice(start, end + 1));
      assert.equal(response.headers.get('x-cache'), i ? 'HIT' : 'MISS');
    }
  });
}

test('full GET streams bytes; HEAD never reads or caches a body', async t => {
  const h = harness(t);
  const head = await h.request({}, 'HEAD');
  assert.equal(head.response.status, 200);
  assert.equal(head.body.length, 0);
  assert.equal(+head.response.headers.get('content-length'), data.length);
  assert.equal(h.reads.length, 0);
  assert.equal(h.entries.size, 0);
  const full = await h.request();
  assert.equal(full.response.status, 200);
  assert.deepEqual(full.body, data);
});

test('concurrent cold ranges in the same block share one R2 read', async t => {
  const h = harness(t);
  const results = await Promise.all([
    h.request({ range: 'bytes=10-25' }),
    h.request({ range: 'bytes=40-55' }),
    h.request({ range: 'bytes=70-85' }),
  ]);
  for (const [i, result] of results.entries()) {
    assert.equal(result.response.status, 206);
    assert.deepEqual(result.body, data.slice(10 + i * 30, 26 + i * 30));
  }
  assert.equal(h.reads.length, 1);
});

// Shared flights live in module scope, so each test below uses its own path.
// A stalled leader never settles; its request is started and abandoned.
const FLIGHT_MS = 3000; // worker/src/index.js
const flush = async () => { for (let i = 0; i < 10; i++) await new Promise(r => setImmediate(r)); };

test('a stalled leader cannot hang later reads of its block', { timeout: 5000 }, async t => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 });
  const h = harness(t);
  const path = '/stalled-leader.pmtiles';
  h.state.hold = new Promise(() => {});
  h.request({ range: 'bytes=10-25' }, 'GET', path);
  await flush();
  // The leader's read is held open past a client disconnect.
  assert.equal(h.waited.length, 1);
  const joiner = h.request({ range: 'bytes=40-55' }, 'GET', path);
  await flush();
  assert.equal(h.reads.length, 1, 'joiner waits on the flight first');
  t.mock.timers.tick(FLIGHT_MS);
  const { response, body } = await joiner;
  assert.equal(response.status, 206);
  assert.deepEqual(body, data.slice(40, 56));
  assert.equal(response.headers.get('x-cache'), 'MISS');
  assert.equal(h.reads.length, 2);
  // The joiner's own read filled the cache, so the block is unwedged.
  const after = await h.request({ range: 'bytes=70-85' }, 'GET', path);
  assert.equal(after.response.headers.get('x-cache'), 'HIT');
});

test('an expired flight is replaced, not joined', { timeout: 5000 }, async t => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 });
  const h = harness(t);
  const path = '/expired-flight.pmtiles';
  h.state.hold = new Promise(() => {});
  h.request({ range: 'bytes=10-25' }, 'GET', path);
  await flush();
  t.mock.timers.tick(FLIGHT_MS);
  // No further timer advance: a request that joined the dead flight would hang.
  const { response, body } = await h.request({ range: 'bytes=40-55' }, 'GET', path);
  assert.equal(response.status, 206);
  assert.deepEqual(body, data.slice(40, 56));
  assert.equal(h.reads.length, 2);
});

test('no-cache never joins a flight', { timeout: 5000 }, async t => {
  const h = harness(t);
  const path = '/no-cache-join.pmtiles';
  h.state.hold = new Promise(() => {});
  h.request({ range: 'bytes=10-25' }, 'GET', path);
  await flush();
  const { response, body } = await h.request({ range: 'bytes=40-55', 'cache-control': 'no-cache' }, 'GET', path);
  assert.equal(response.status, 206);
  assert.deepEqual(body, data.slice(40, 56));
  assert.equal(response.headers.get('x-cache'), 'MISS');
  assert.equal(h.reads.length, 2);
});

test('a failed leader does not fail the requests that joined it', { timeout: 5000 }, async t => {
  const h = harness(t);
  const path = '/failed-leader.pmtiles';
  let release;
  h.state.hold = new Promise(r => { release = r; });
  const leader = h.request({ range: 'bytes=10-25' }, 'GET', path);
  await flush();
  const joiner = h.request({ range: 'bytes=40-55' }, 'GET', path);
  await flush();
  release('fail');
  assert.equal((await leader).response.status, 503);
  const { response, body } = await joiner;
  assert.equal(response.status, 206);
  assert.deepEqual(body, data.slice(40, 56));
  assert.equal(h.reads.length, 2);
});

for (const range of ['bytes=20-10', 'bytes=-0', 'bytes=0-1,4-5', 'not-a-range', `bytes=${data.length}-`, 'bytes=9007199254740992-']) {
  test(`invalid/unsatisfiable ${range} returns recoverable 416`, async t => {
    const h = harness(t);
    const { response } = await h.request({ range });
    assert.equal(response.status, 416);
    assert.equal(response.headers.get('content-range'), `bytes */${data.length}`);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.equal(h.entries.size, 0);
  });
}

test('weak/list/wildcard validators return bodyless 304 on cold and warm paths', async t => {
  const h = harness(t);
  for (const range of [undefined, 'bytes=10-25']) {
    for (const validator of ['W/"v1"', '"other", "v1"', '*']) {
      const headers = { 'if-none-match': validator, ...(range ? { range } : {}) };
      const { response, body } = await h.request(headers);
      assert.equal(response.status, 304);
      assert.equal(body.length, 0);
      assert.equal(response.headers.get('etag'), '"v1"');
    }
    await h.request(range ? { range } : {});
    assert.equal((await h.request({ 'if-none-match': '"v1"', ...(range ? { range } : {}) })).response.status, 304);
  }
  assert.equal((await h.request({ 'if-none-match': '"v1"' }, 'HEAD')).response.status, 304);
});

test('reload bypasses stale cache and refuses to splice different object versions', async t => {
  const h = harness(t);
  await h.request({ range: 'bytes=10-25' });
  h.state.etag = '"v2"';
  assert.equal((await h.request({ range: `bytes=${BLOCK - 1}-${BLOCK + 1}` })).response.status, 416);
  const fresh = await h.request({ range: 'bytes=10-25', 'cache-control': 'no-cache' });
  assert.equal(fresh.response.headers.get('etag'), '"v2"');
  assert.equal(fresh.response.headers.get('x-cache'), 'MISS');
  assert.equal((await h.request({ range: `bytes=${BLOCK - 1}-${BLOCK + 1}` })).response.status, 206);
});

test('storage failures are retryable, failed flights do not poison later reads, cache failures are harmless', async t => {
  const h = harness(t);
  h.state.fail = true;
  for (const [headers, method] of [[{ range: 'bytes=10-25' }, 'GET'], [{}, 'GET'], [{}, 'HEAD']]) {
    const { response } = await h.request(headers, method);
    assert.equal(response.status, 503);
    assert.equal(response.headers.get('retry-after'), '1');
    assert.equal(response.headers.get('cache-control'), 'no-store');
  }
  h.state.fail = false;
  h.state.cacheFail = true;
  assert.deepEqual((await h.request({ range: 'bytes=10-25' })).body, data.slice(10, 26));
});

test('preflight, unsupported methods, bad paths and missing objects', async t => {
  const h = harness(t);
  assert.equal((await h.request({}, 'OPTIONS')).response.status, 204);
  assert.equal((await h.request({}, 'POST')).response.status, 405);
  assert.equal((await h.request({}, 'GET', '/%ZZ')).response.status, 400);
  assert.equal((await h.request({}, 'GET', '/')).response.status, 404);
  assert.equal(h.reads.length, 0);
  h.state.missing = true;
  assert.equal((await h.request()).response.status, 404);
});

test('a content-hashed key is immutable; a plain key keeps the one-day lifetime', async t => {
  const h = harness(t);
  const immutable = 'public, max-age=31536000, immutable', daily = 'public, max-age=86400, stale-while-revalidate=3600';
  for (const [path, expected] of [
    ['/next/manifest.0123456789ab.json', immutable], ['/next/pins.0123456789ab/391.json', immutable],
    ['/sectionals/1950-01-01_to_1960-01-01.0123456789ab.pmtiles', immutable],
    ['/chart.pmtiles', daily], ['/sectionals/timeline_data.json', daily], ['/basemap/protomaps-20260826.pmtiles', daily],
  ]) {
    for (const [headers, method] of [[{}, 'GET'], [{ range: 'bytes=0-15' }, 'GET'], [{}, 'HEAD']]) {
      const { response } = await h.request(headers, method, path);
      assert.equal(response.headers.get('cache-control'), expected, `${method} ${path} ${headers.range || ''}`);
    }
  }
});
