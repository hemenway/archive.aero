import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import worker from '../src/index.js';
import { restoreFacilities } from '../src/facilities.js';
import FACILITIES from '../src/facilities.json' with { type: 'json' };
import { keyCandidates, routeOldPath } from '../src/routes.js';

const fixture = readFileSync(new URL('../../tests/fixtures/atc-facilities.html', import.meta.url));
const base = 'https://archive.aero/atc/facility-photos';

test('every indexed location has entries at canonical article and media URLs', () => {
  assert.equal(Object.keys(FACILITIES).length, 46);
  assert.equal(Object.values(FACILITIES).reduce((n, cities) => n + Object.keys(cities).length, 0), 303);
  for (const cities of Object.values(FACILITIES)) for (const entries of Object.values(cities)) {
    assert.ok(entries.length);
    for (const entry of entries) {
      assert.equal(routeOldPath(entry.href.slice(4)), null, entry.href);
      assert.ok(keyCandidates(entry.href).length);
      assert.ok(entry.title);
      if (entry.image) {
        assert.match(entry.image, /^\/atc\/(media|history)\//);
        assert.ok(keyCandidates(entry.image).length);
      }
    }
  }
});

test('native controls replace the old widget and script while preserving source bytes', () => {
  const original = Uint8Array.from([...fixture, 0x93, 0x94, 0xe9]);
  const result = restoreFacilities(original, new URL(base));
  const text = new TextDecoder().decode(result);
  assert.equal((text.match(/<details/g) || []).length, 46);
  assert.match(text, /<summary>New York<\/summary>/);
  assert.match(text, /state=New\+York&#38;city=/);
  assert.doesNotMatch(text, /jQuery|data-ids=/);
  assert.deepEqual([...result.slice(-3)], [0x93, 0x94, 0xe9]);
  assert.equal(restoreFacilities(new TextEncoder().encode('<body>Untouched</body>'), new URL(base)), null);
});

test('city selections show only matching entries, including history pages without images', () => {
  const mobile = new TextDecoder().decode(restoreFacilities(fixture, new URL(base + '?state=Alabama&city=Mobile')));
  assert.match(mobile, /<h2>Mobile, Alabama<\/h2>/);
  assert.equal((mobile.match(/<li>/g) || []).length, 2);
  assert.match(mobile, /href="\/atc\/mobile-fss-personnel-1975"/);
  assert.match(mobile, /<details open><summary>Alabama/);
  assert.match(mobile, /aria-current="page">Mobile/);
  assert.doesNotMatch(mobile, /href="\/atc\/anniston-fss-1969"/);
  const alaska = new TextDecoder().decode(restoreFacilities(fixture, new URL(base + '?state=Alaska&city=Alaska')));
  assert.match(alaska, /href="\/atc\/alaska-fss-managers-list"/);
});

test('invalid and hostile selections give an explicit empty state without reflecting input', () => {
  for (const query of ['state=__proto__&city=constructor', 'state=Alabama&city=%3Cscript%3Ealert(1)%3C%2Fscript%3E']) {
    const text = new TextDecoder().decode(restoreFacilities(fixture, new URL(base + '?' + query)));
    assert.match(text, /Location not found/);
    assert.doesNotMatch(text, /alert\(1\)|__proto__|constructor/);
  }
});

function harness(t) {
  const entries = new Map();
  const pending = [];
  const previous = globalThis.caches;
  globalThis.caches = { default: {
    async match(req) { return entries.get(req.url)?.clone(); },
    async put(req, res) { entries.set(req.url, new Response(await res.arrayBuffer(), res)); },
  } };
  t.after(() => { globalThis.caches = previous; });
  const env = { MODE: 'redirect', ATC_SHELL: '1', SAMPLE_RATE: '0', BUCKET: {
    async get(key) {
      assert.equal(key, 'facility-photos/index.html');
      return { size: fixture.length, httpEtag: '"original"',
        writeHttpMetadata(headers) { headers.set('content-type', 'text/html; charset=utf-8'); },
        arrayBuffer: async () => fixture,
      };
    },
  } };
  return async (url, init) => {
    const res = await worker.fetch(new Request(url, init), env, { waitUntil(p) { pending.push(p); } });
    const text = await res.text();
    await Promise.all(pending.splice(0));
    return { res, text };
  };
}

test('HTTP filters survive aliases and the retired host, and render on staging', async t => {
  const request = harness(t);
  for (const url of [base + '/', 'https://atchistory.org/facility-photos/', 'https://www.atchistory.org/facility-photos']) {
    const { res } = await request(url + '?state=New%20York&city=Albany');
    assert.equal(res.status, 301);
    assert.equal(res.headers.get('location'), base + '?state=New+York&city=Albany');
  }
  const shortlink = await request('https://archive.aero/atc/?p=379&state=Alabama&city=Anniston');
  assert.equal(shortlink.res.headers.get('location'), base + '?state=Alabama&city=Anniston');
  const { res, text } = await request('https://atc-staging.archive.aero/facility-photos?state=Alabama&city=Anniston');
  assert.equal(res.status, 200);
  assert.match(text, /href="\/anniston-fss-1969"/);
  assert.equal(res.headers.get('x-robots-tag'), 'noindex, nofollow');
});

test('GET, HEAD and cache validators distinguish each selected city and pre-fix HTML', async t => {
  const request = harness(t);
  const url = base + '?state=Alabama&city=Anniston';
  const first = await request(url, { headers: { 'if-none-match': '"original-shell"' } });
  assert.equal(first.res.status, 200);
  assert.match(first.text, /Anniston, Alabama/);
  assert.equal(+first.res.headers.get('content-length'), Buffer.byteLength(first.text));
  const hit = await request(url);
  assert.equal(hit.res.headers.get('x-cache'), 'HIT');
  assert.equal(hit.text, first.text);
  const etag = first.res.headers.get('etag');
  assert.equal((await request(url, { headers: { 'if-none-match': etag } })).res.status, 304);
  const head = await request(url, { method: 'HEAD' });
  assert.equal(head.text, '');
  assert.equal(head.res.headers.get('etag'), etag);
  assert.equal(head.res.headers.get('content-length'), first.res.headers.get('content-length'));
  const dothan = await request(base + '?state=Alabama&city=Dothan', { headers: { 'if-none-match': etag } });
  assert.equal(dothan.res.status, 200);
  assert.match(dothan.text, /Dothan, Alabama/);
  assert.notEqual(dothan.res.headers.get('etag'), etag);
});
