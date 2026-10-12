import { readFile } from 'node:fs/promises';
import { test, expect } from '@playwright/test';
import { installFixtures } from './fixtures.mjs';

const view = '/?date=1960-01-01&lat=32.7767&lng=-96.7970&zoom=10';
const parser = '/vendor/papaparse/5.4.1/papaparse.min.js';

test('normal boot uses local deferred libraries and never fetches PapaParse or CSV', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  const requests = [];
  page.on('request', request => requests.push(new URL(request.url())));
  await page.goto(view);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  expect(requests.some(u => u.hostname === 'unpkg.com')).toBe(false);
  expect(requests.some(u => u.hostname === 'get.geojs.io')).toBe(false);
  expect(requests.some(u => u.pathname === parser || u.pathname === '/dates.csv')).toBe(false);
  expect(await page.evaluate(() => typeof window.Papa)).toBe('undefined');
  const vendors = await page.locator('script[src^="/vendor/"]').evaluateAll(scripts =>
    scripts.map(s => ({ defer: s.defer, integrity: s.integrity, async: s.async })));
  expect(vendors).toHaveLength(3);
  expect(vendors.every(s => s.defer && !s.async && s.integrity.startsWith('sha384-'))).toBe(true);
  await expect(page.locator('script[data-viewer-entry]')).toHaveAttribute('type', 'module');
  await expect(page.locator('script[data-viewer-entry]')).toHaveAttribute('src', /^\/assets\/boot\.[a-f0-9]{16}\.js$/);
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

for (const missingDecompression of [false, true]) {
  test(`CSV fallback loads PapaParse once (${missingDecompression ? 'no DecompressionStream' : 'bundle outage'})`, async ({ page }) => {
    const diagnostics = await installFixtures(page, { bundleFails: !missingDecompression });
    if (missingDecompression) await page.addInitScript(() => { window.DecompressionStream = undefined; });
    const requests = [];
    page.on('request', request => requests.push(new URL(request.url()).pathname));
    await page.goto(view);
    await expect(page.locator('#loadingSplash')).toBeHidden();
    await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
    expect(requests.filter(path => path === parser)).toHaveLength(1);
    expect(requests.indexOf(parser)).toBeLessThan(requests.indexOf('/dates.csv'));
    expect(diagnostics).toEqual({ errors: [], unexpected: [] });
  });
}

test('a failed lazy parser shows the data error instead of freezing the splash', async ({ page }) => {
  const diagnostics = await installFixtures(page, { bundleFails: true });
  await page.route('**' + parser, route => route.fulfill({ status: 503, body: '' }));
  await page.goto(view);
  await expect(page.locator('#loadingStatus')).toContainText('Failed to load chart data');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('a missing required vendor shows a useful library error', async ({ page }) => {
  await installFixtures(page);
  await page.route('**/vendor/leaflet/1.9.4/leaflet.js', route => route.fulfill({ status: 503, body: '' }));
  await page.goto(view);
  await expect(page.locator('#loadingStatus')).toContainText('Failed to load map libraries');
});

test('a missing application module is caught by the small bootstrap', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await page.route('**/assets/viewer.*.js', route => route.fulfill({ status: 503, body: '' }));
  await page.goto(view);
  await expect(page.locator('#loadingStatus')).toContainText('Failed to load the chart viewer');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('slow vendor responses do not block parsing or race application initialization', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/vendor/leaflet/1.9.4/leaflet.js', async route => { await held; await route.continue(); });
  await page.goto(view, { waitUntil: 'commit' });
  try {
    await expect(page.locator('#timeSelect')).toBeAttached();
    await expect(page.locator('#loadingSplash')).toBeVisible();
    await expect(page.locator('#loadingStatus')).toHaveText('Initializing...');
  } finally { release(); }
  await expect(page.locator('#loadingSplash')).toBeHidden();
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('a missing bootstrap still displays an actionable HTML-level error', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await page.route('**/assets/boot.*.js', route => route.fulfill({ status: 503, body: '' }));
  await page.goto(view);
  await expect(page.locator('#loadingStatus')).toContainText('Failed to load the chart viewer');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('fingerprinted assets have JavaScript MIME type and support conditional reuse', async ({ request }) => {
  const html = await (await request.get('/')).text();
  const entry = html.match(/data-viewer-entry src="([^"]+)"/)[1];
  const first = await request.get(entry);
  expect(first.headers()['content-type']).toContain('javascript');
  expect(first.headers()['cache-control']).toContain('max-age=');
  const validated = await request.get(entry, { headers: { 'if-none-match': first.headers().etag } });
  expect(validated.status()).toBe(304);
});

test('cache budgets: iOS, Android and small-memory devices get the reduced sizes', async () => {
  // The shipped DeviceInfo block, run against stand-in navigators.
  const source = await readFile(new URL('../../src/viewer.js', import.meta.url), 'utf8');
  const start = source.indexOf('const DeviceInfo = (() => {');
  const block = source.slice(start, source.indexOf('})();', start) + 5);
  expect(start).toBeGreaterThanOrEqual(0);
  const classify = nav => new Function('navigator', `${block}; return DeviceInfo;`)({ platform: '', maxTouchPoints: 0, ...nav });
  const iphone = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1';
  const pixel = 'Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36';
  const mac = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15';
  const chrome = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36';
  expect(classify({ userAgent: iphone })).toMatchObject({ isIOS: true, lowMemory: true });
  expect(classify({ userAgent: mac, platform: 'MacIntel', maxTouchPoints: 5 })).toMatchObject({ isIOS: true, lowMemory: true });
  expect(classify({ userAgent: pixel, deviceMemory: 8 })).toMatchObject({ isIOS: false, lowMemory: true });
  expect(classify({ userAgent: chrome, deviceMemory: 4 })).toMatchObject({ isIOS: false, lowMemory: true });
  expect(classify({ userAgent: chrome, deviceMemory: 8 })).toMatchObject({ isIOS: false, lowMemory: false });
  expect(classify({ userAgent: mac, platform: 'MacIntel' })).toMatchObject({ isIOS: false, lowMemory: false });
});
