import assert from 'node:assert/strict';
import { test } from 'node:test';
import { handle } from '../worker.mjs';
import { airport, index, shell } from './fixtures.mjs';

const ORIGIN = 'https://beta.archive.aero';
const config = { version: 1, base: '/next/', site: 'https://archive.aero', charts: 'https://data-beta.archive.aero/next/charts.0123456789ab/', airports: 'https://data-beta.archive.aero/next/airports.0123456789ab/', files: 'https://data-beta.archive.aero/', image: 'https://archive.aero/social-card.jpg', csp: "default-src 'self'", robots: 'noindex, nofollow' };

// base: where the shell lives; fallthrough: ORIGIN_FALLTHROUGH="1" (the apex deployment);
// held: asset paths the ASSETS binding answers besides the shell's own (everything else is its 404).
function harness({ withConfig = true, base = '/next/', fallthrough = false, held = null, chartIndex = index } = {}) {
  const assetCalls = [], indexCalls = [], originCalls = [];
  const cfg = { ...config, base };
  const env = { ORIGIN_FALLTHROUGH: fallthrough ? '1' : undefined, ASSETS: { async fetch(request) {
    const url = new URL(request.url); assetCalls.push(url.pathname);
    if (url.pathname === base + 'chart-pages.json') return withConfig ? Response.json(cfg) : new Response('', { status: 404 });
    if (url.pathname === base) return new Response(shell, { headers: { 'content-type': 'text/html; charset=utf-8', etag: '"abc"', 'content-length': String(shell.length), 'x-asset': 'yes' } });
    if (url.pathname === '/404.html') return new Response('<title>Not found — beta</title>', { headers: { 'content-type': 'text/html' } });
    if (held && !held.includes(url.pathname)) return new Response(null, { status: 404 });
    return new Response('asset:' + url.pathname, { status: 200 });
  } } };
  const fetcher = async (input, init) => {
    if (input instanceof Request) { originCalls.push(input.method + ' ' + new URL(input.url).pathname); return new Response('origin:' + new URL(input.url).pathname, { status: 200, headers: { 'x-origin': 'pages' } }); }
    indexCalls.push(input);
    if (input.includes('/airports.')) return input.endsWith('/kbos.json') ? Response.json(airport) : new Response('', { status: 404 });
    return input.endsWith('/boston.json') ? Response.json(chartIndex) : new Response('', { status: 404 });
  };
  const run = (path, init) => handle(new Request(ORIGIN + path, init), env, {}, { fetcher });
  return { run, assetCalls, indexCalls, originCalls };
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
  assert.match(html, /<meta property="og:image" content="https:\/\/archive.aero\/social-card.jpg">/, 'no card of its own: the site card');
  assert.match(html, /og:image:alt" content="A 1950 FAA sectional chart of New York City/);
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

test('a chart with a pre-rendered card of its own carries it as og:image, resolved against the data host', async () => {
  const own = { ...index, charts: index.charts.map(c => c.d === '1953-12-01' ? { ...c, og: 'sectionals/chart/boston/1953-12-01.jpg' } : c) };
  const h = harness({ chartIndex: own });
  const html = await (await h.run('/next/sectionals/chart/boston/1953-12-01')).text();
  assert.match(html, /<meta property="og:image" content="https:\/\/data-beta.archive.aero\/sectionals\/chart\/boston\/1953-12-01.jpg">/);
  assert.match(html, /<meta name="twitter:image" content="https:\/\/data-beta.archive.aero\/sectionals\/chart\/boston\/1953-12-01.jpg">/);
  assert.match(html, /og:image:alt" content="The Boston sectional chart of December 1953, as shown in the archive.aero chart viewer"/);
  assert.match(html, /"image":"https:\/\/data-beta.archive.aero\/sectionals\/chart\/boston\/1953-12-01.jpg"/, 'JSON-LD names the same image');
});

test('at the apex with ORIGIN_FALLTHROUGH, a path the assets lack goes to the zone origin', async () => {
  const h = harness({ base: '/', fallthrough: true, held: ['/app.ABC.js', '/sw.js'] });
  // The shell and its files come from the assets.
  assert.equal(await (await h.run('/')).text(), shell);
  assert.equal(await (await h.run('/app.ABC.js')).text(), 'asset:/app.ABC.js');
  // Pages the Worker does not hold are the origin's: /about, /atc/, the legacy viewer and its files.
  for (const path of ['/about', '/atc/', '/classic/', '/classic/styles.css', '/assets/viewer.0123456789abcdef.js', '/robots.txt']) {
    const response = await h.run(path);
    assert.equal(response.headers.get('x-origin'), 'pages', path);
    assert.equal(await response.text(), 'origin:' + path, path);
  }
  const posted = await h.run('/contribute', { method: 'POST' });
  assert.equal(await posted.text(), 'origin:/contribute', 'a non-GET never touches the assets');
  assert.deepEqual(h.originCalls, ['GET /about', 'GET /atc/', 'GET /classic/', 'GET /classic/styles.css', 'GET /assets/viewer.0123456789abcdef.js', 'GET /robots.txt', 'POST /contribute']);
  // The edition pages themselves render at the apex: config read from /chart-pages.json, canonical at the apex.
  const page = await h.run('/sectionals/chart/boston/1953-12-01');
  assert.equal(page.status, 200);
  assert.match(await page.text(), /<link rel="canonical" href="https:\/\/beta.archive.aero\/sectionals\/chart\/boston\/1953-12-01">/);
  assert.equal((await h.run('/sectionals/chart/Boston/1953-12-01')).headers.get('location'), ORIGIN + '/sectionals/chart/boston/1953-12-01');
  assert.equal((await h.run('/sectionals/chart/boston/1954-06-01')).status, 404, 'thin-content guard still a 404, not an origin fetch');
});

test('without ORIGIN_FALLTHROUGH a missing asset stays the assets 404', async () => {
  const h = harness({ held: ['/next/app.ABC.js'] });
  assert.equal((await h.run('/about')).status, 404);
  assert.deepEqual(h.originCalls, []);
});

test('a retired host answers everything 410 Gone, noindex, cached an hour', async () => {
  const h = harness();
  const env = { RETIRED: '1', ASSETS: { fetch: () => { throw new Error('assets must not be consulted'); } } };
  for (const path of ['/', '/next/', '/next/sectionals/chart/boston/1953-12-01', '/next/app.ABC.js', '/next/sw.js']) {
    const response = await handle(new Request(ORIGIN + path), env, {});
    assert.equal(response.status, 410, path);
    assert.equal(response.headers.get('cache-control'), 'public, max-age=3600');
    assert.equal(response.headers.get('x-robots-tag'), 'noindex');
    assert.match(await response.text(), /archive\.aero\//);
  }
  const head = await handle(new Request(ORIGIN + '/next/', { method: 'HEAD' }), env, {});
  assert.equal(head.status, 410); assert.equal(await head.text(), '');
  assert.equal((await handle(new Request(ORIGIN + '/next/', { method: 'POST' }), env, {})).status, 410);
  // Not retired: the harness still serves.
  assert.equal((await h.run('/next/sectionals/chart/boston/1953-12-01')).status, 200);
});

test('airport pages render at /<base>airports/<ident>: head, aside view, card of the newest edition; aliases 301; unknown 404', async () => {
  const h = harness();
  const response = await h.run('/next/airports/kbos');
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-security-policy'), config.csp);
  const html = await response.text();
  assert.match(html, /<title>Boston Logan International Airport on aeronautical charts, 1933–1970 · archive.aero<\/title>/);
  assert.match(html, /<link rel="canonical" href="https:\/\/beta.archive.aero\/next\/airports\/kbos">/);
  assert.match(html, /data-airport="kbos" data-lat="42.3643" data-lng="-71.0052" data-zoom="10" data-date="1960-01-01"/);
  assert.match(html, /href="\/next\/sectionals\/chart\/boston\/1953-12-01" data-ap-chart="boston\/1953-12-01"/);
  assert.match(html, /<meta property="og:image" content="https:\/\/data-beta.archive.aero\/sectionals\/chart\/boston\/1953-12-01.jpg">/);
  assert.deepEqual(h.indexCalls, [config.airports + 'kbos.json']);
  for (const [path, target] of [['/next/airports/KBOS', '/next/airports/kbos'], ['/next/airports/kbos/?x=1', '/next/airports/kbos?x=1']]) {
    const r = await h.run(path); assert.equal(r.status, 301, path); assert.equal(r.headers.get('location'), ORIGIN + target, path);
  }
  for (const path of ['/next/airports/kxyz', '/next/airports/k', '/next/airports/kbos/1953']) {
    const r = await h.run(path); assert.equal(r.status, 404, path); assert.equal(r.headers.get('x-robots-tag'), 'noindex');
  }
  assert.equal((await h.run('/next/airports/kbos', { method: 'POST' })).status, 405);
});

test('without airport records in the config the address passes to the assets', async () => {
  const assetCalls = [];
  const env = { ASSETS: { async fetch(request) { const p = new URL(request.url).pathname; assetCalls.push(p); return p === '/chart-pages.json' ? Response.json({ ...config, base: '/', airports: null }) : new Response('asset:' + p); } } };
  const response = await handle(new Request(ORIGIN + '/airports/kbos'), env, {}, { fetcher: async () => { throw new Error('no record fetch'); } });
  assert.equal(await response.text(), 'asset:/airports/kbos');
});

test('server-rendered chart and airport records use the tiles service binding', async () => {
  const calls = [];
  const env = {
    ORIGIN_FALLTHROUGH: '1',
    ASSETS: { async fetch(request) {
      const path = new URL(request.url).pathname;
      if (path === '/chart-pages.json') return Response.json({ ...config, base: '/' });
      return path === '/' ? new Response(shell) : new Response('', { status: 404 });
    } },
    DATA: { async fetch(input) {
      calls.push(input);
      return Response.json(input.includes('/airports.') ? airport : index);
    } }
  };
  const fetcher = async input => {
    assert.ok(input instanceof Request, 'record reads must not use public fetch');
    return new Response('origin');
  };
  for (const path of ['/airports/kbos', '/sectionals/chart/boston/1953-12-01']) {
    assert.equal((await handle(new Request(ORIGIN + path), env, {}, { fetcher })).status, 200);
  }
  assert.equal(calls.length, 2);
  assert.equal(await (await handle(new Request(ORIGIN + '/about'), env, {}, { fetcher })).text(), 'origin');
});
