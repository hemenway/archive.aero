import assert from 'node:assert/strict';
import { test } from 'node:test';
import { handle } from '../worker.mjs';
import { index, shell } from './fixtures.mjs';

const ORIGIN = 'https://beta.archive.aero';
const config = { version: 1, base: '/next/', site: 'https://archive.aero', charts: 'https://data-beta.archive.aero/next/charts.0123456789ab/', image: 'https://archive.aero/social-card.jpg', csp: "default-src 'self'", robots: 'noindex, nofollow' };

function harness({ withConfig = true } = {}) {
  const assetCalls = [], indexCalls = [];
  const env = { ASSETS: { async fetch(request) {
    const url = new URL(request.url); assetCalls.push(url.pathname);
    if (url.pathname === '/next/chart-pages.json') return withConfig ? Response.json(config) : new Response('', { status: 404 });
    if (url.pathname === '/next/') return new Response(shell, { headers: { 'content-type': 'text/html; charset=utf-8', etag: '"abc"', 'content-length': String(shell.length), 'x-asset': 'yes' } });
    if (url.pathname === '/404.html') return new Response('<title>Not found — beta</title>', { headers: { 'content-type': 'text/html' } });
    return new Response('asset:' + url.pathname, { status: 200 });
  } } };
  const fetcher = async (url) => { indexCalls.push(url); return url.endsWith('/boston.json') ? Response.json(index) : new Response('', { status: 404 }); };
  const run = (path, init) => handle(new Request(ORIGIN + path, init), env, {}, { fetcher });
  return { run, assetCalls, indexCalls };
}

test('renders an edition page with the chart in head, body and headers', async () => {
  const h = harness();
  const response = await h.run('/next/sectionals/chart/boston/1953-12-01');
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-type'), 'text/html; charset=utf-8');
  assert.equal(response.headers.get('cache-control'), 'no-cache');
  assert.equal(response.headers.get('content-security-policy'), config.csp);
  assert.equal(response.headers.get('x-robots-tag'), 'noindex, nofollow');
  assert.equal(response.headers.get('etag'), null); assert.equal(response.headers.get('content-length'), null);
  assert.equal(response.headers.get('x-asset'), 'yes', 'the shell response headers come through');
  const html = await response.text();
  assert.match(html, /<title>Boston sectional chart, December 1953 \(edition 32\) · archive.aero<\/title>/);
  assert.match(html, /<link rel="canonical" href="https:\/\/beta.archive.aero\/next\/sectionals\/chart\/boston\/1953-12-01">/);
  assert.match(html, /<h1 class="cv-title">Boston<\/h1>/);
  assert.match(html, /href="\/next\/sectionals\/chart\/boston\/1953-06-01"/);
  assert.match(html, /href="https:\/\/archive.aero\/sources"/);
  assert.deepEqual(h.indexCalls, [config.charts + 'boston.json']);
  const head = await h.run('/next/sectionals/chart/boston/1953-12-01', { method: 'HEAD' });
  assert.equal(head.status, 200); assert.equal(await head.text(), '');
});

test('other spellings redirect once to the canonical address, with the query kept', async () => {
  const h = harness();
  for (const [path, target] of [
    ['/next/sectionals/chart/Boston/1953-12-01', '/next/sectionals/chart/boston/1953-12-01'],
    ['/next/sectionals/chart/boston/1953-12-01/', '/next/sectionals/chart/boston/1953-12-01'],
    ['/next/sectionals/chart/boston/1953-12-01-north?lat=42&lng=-71', '/next/sectionals/chart/boston/1953-12-01?lat=42&lng=-71'],
    ['/next/sectionals/chart/boston/1953-12-01.pmtiles', '/next/sectionals/chart/boston/1953-12-01']
  ]) {
    const response = await h.run(path);
    assert.equal(response.status, 301, path); assert.equal(response.headers.get('location'), ORIGIN + target, path);
  }
});

test('unknown locations, dates without a tiled chart and malformed addresses are 404 pages', async () => {
  const h = harness();
  for (const path of ['/next/sectionals/chart/atlantis/1953-12-01', '/next/sectionals/chart/boston/1954-06-01', '/next/sectionals/chart/boston/1999-01-01', '/next/sectionals/chart/boston', '/next/sectionals/chart/boston/1953-02-30']) {
    const response = await h.run(path);
    assert.equal(response.status, 404, path);
    assert.equal(response.headers.get('x-robots-tag'), 'noindex');
    assert.match(await response.text(), /Not found — beta/, 'the host 404 page is used');
  }
  const posted = await h.run('/next/sectionals/chart/boston/1953-12-01', { method: 'POST' });
  assert.equal(posted.status, 405);
});

test('everything outside the edition prefix, and a host without the config file, pass to the assets', async () => {
  const h = harness();
  for (const path of ['/next/app.ABC.js', '/', '/about', '/other/chart/boston/1953-12-01?x=1']) {
    const response = await h.run(path);
    assert.equal(await response.text(), 'asset:' + path.split('?')[0], path);
  }
  const bare = harness({ withConfig: false });
  const response = await bare.run('/next/sectionals/chart/boston/1953-12-01');
  assert.equal(await response.text(), 'asset:/next/sectionals/chart/boston/1953-12-01');
  assert.deepEqual(bare.indexCalls, []);
});

test('a failing chart index is a 502, never a cached page', async () => {
  const h = harness();
  const response = await handle(new Request(ORIGIN + '/next/sectionals/chart/boston/1953-12-01'), { ASSETS: { fetch: (r) => new URL(r.url).pathname === '/next/chart-pages.json' ? Response.json(config) : new Response(shell) } }, {}, { fetcher: async () => new Response('', { status: 503 }) });
  assert.equal(response.status, 502); assert.equal(response.headers.get('cache-control'), 'no-store');
});
