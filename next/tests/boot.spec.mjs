import { readFile } from 'node:fs/promises';
import { test, expect, ready, view } from './fixtures.mjs';
test('boot requests are bounded, early requests adopted, fonts self-hosted, size budgets', async ({ page, guard }) => {
  await page.addInitScript(() => { window.addEventListener('first-chart-paint', () => { window.__paintResources = performance.getEntriesByType('resource').map(r => r.name); }); });
  await ready(page, './');
  const resources = await page.evaluate(() => window.__paintResources);
  expect(resources.length).toBeLessThanOrEqual(50);
  const tiles = guard.requests.filter(u => u.includes('/t/')); expect(new Set(tiles).size).toBe(tiles.length);
  expect(guard.requests.some(u => /fonts\.(googleapis|gstatic)\.com|unpkg|cdnjs/.test(u))).toBe(false);
  await expect(page.locator('link[rel=preload][as=fetch]')).toHaveAttribute('crossorigin', '');
  expect(await page.locator('link[rel=modulepreload]').count()).toBeGreaterThanOrEqual(1);
  const budget = JSON.parse(await readFile(new URL('../dist/budgets.json', import.meta.url)));
  for (const [name, max] of Object.entries(budget.limits)) expect(budget.bytes[name]).toBeLessThanOrEqual(max);
  expect(budget.bytes.inlineBoot).toBeLessThanOrEqual(1536);
});
test('the splash shows progress, then leaves; the page is the production page', async ({ page }) => {
  await ready(page);
  await expect(page.locator('#loadingStatus')).toHaveText('Ready!');
  await expect(page.locator('.header-bar .site-logo')).toContainText('archive.aero');
  await expect(page.locator('#toolsBtn .tools-btn-label')).toHaveText('Layers');
  await expect(page.locator('.leaflet-top.leaflet-right #toolsControl')).toHaveCount(1);
  await expect(page.locator('.ctl-col .leaflet-control-zoom + #utilRail')).toHaveCount(1);
  expect(await page.locator('svg').count()).toBeGreaterThanOrEqual(12);
  // Barlow is served by this origin, not a font CDN.
  expect(await page.evaluate(async () => { await document.fonts.ready; return getComputedStyle(document.querySelector('.site-logo')).fontFamily; })).toContain('Barlow');
});
test('manifest failure leaves the splash up with an actionable message', async ({ page }) => {
  await page.route('**/manifest.*.json', route => route.fulfill({ status: 503, body: '' }));
  await page.goto(view); await expect(page.locator('#loadingStatus')).toHaveText('Failed to load chart data. Check your connection and reload.');
  await expect(page.locator('#loadingSplash')).toBeVisible(); await expect(page.locator('#loader')).toBeHidden();
});
test('renderer unsupported error links to the current viewer', async ({ page }) => {
  await page.addInitScript(() => { const get = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function(type, ...args) { return this.id === 'mapCanvas' ? null : get.call(this, type, ...args); }; });
  await page.goto(view); await expect(page.locator('#loadingStatus')).toContainText('cannot run the new chart renderer'); await expect(page.locator('#loadingStatus a')).toHaveAttribute('href', '/');
  await expect(page.locator('#loadingSplash')).toBeVisible();
});
test('worker failure brings the splash back with the reason and stops playback', async ({ page }) => {
  await page.addInitScript(() => { const Original = window.Worker; window.Worker = class extends Original { constructor(...args) { super(...args); window.__failWorker = () => this.dispatchEvent(new ErrorEvent('error', { message: 'fixture worker failed', cancelable: true })); } }; });
  await page.goto(view); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await page.locator('#playBtn').click(); await expect(page.locator('#playBtn')).toHaveAttribute('aria-label', 'Pause animation');
  await page.evaluate(() => window.__failWorker());
  await expect(page.locator('#loadingStatus')).toHaveText('The chart viewer stopped working. Reload the page to continue.');
  await expect(page.locator('#loadingSplash')).toBeVisible(); await expect(page.locator('#loader')).toBeHidden();
  await expect(page.locator('#playBtn')).toHaveAttribute('aria-label', 'Play animation');
});
test('100 ms loading grace and playback readiness gate', async ({ page }) => {
  await page.clock.install(); await page.clock.pauseAt(new Date());
  let release; const gate = new Promise(resolve => release = resolve);
  await page.route('**/t/**', async route => { await gate; await route.fallback(); });
  await page.goto(view); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await page.clock.runFor(99); await expect(page.locator('#loader')).toBeHidden();
  await page.clock.runFor(1); await expect(page.locator('#loader')).toBeVisible(); await expect(page.locator('#loader')).toHaveText('Loading charts');
  await page.locator('#playBtn').click(); await page.clock.runFor(2100); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  release(); await expect(page.locator('#loader')).toBeHidden(); await page.locator('#playBtn').click();
});
test('warning banner follows its 1.5 to 8 second window', async ({ page }) => {
  await page.clock.install(); await ready(page);
  await expect(page.locator('#warningOverlay')).not.toHaveClass(/visible/); await page.clock.runFor(1500); await expect(page.locator('#warningOverlay')).toHaveClass(/visible/);
  await page.clock.runFor(6500); await expect(page.locator('#warningOverlay')).not.toHaveClass(/visible/);
});
test('the address bar is left alone while scrubbing; a share link carries the state', async ({ page }) => {
  await ready(page); const before = page.url();
  for (let i = 0; i < 6; i++) { await page.locator('#timeSelect').fill(i % 2 ? '1960-01-01' : '1970-01-01'); await page.locator('#timeSelect').dispatchEvent('change'); }
  expect(page.url()).toBe(before);
  await page.locator('#shareBtn').click(); await expect.poll(() => page.evaluate(() => window.__copied)).toContain('date=1960-01-01');
  await expect(page.locator('#toast')).toHaveText('Link copied'); await expect(page.locator('#toast')).toHaveClass(/visible/);
});
test('a failing tile degrades the view instead of ending it', async ({ page }) => {
  const warnings = []; page.on('console', message => { if (message.type() === 'warning') warnings.push(message.text()); });
  let failed = false;
  await page.route('**/t/sectionals/**', route => { if (failed) return route.fallback(); failed = true; return route.fulfill({ status: 404, body: '' }); });
  await ready(page); await expect(page.locator('#toast')).toHaveText('Some chart data failed to load — showing base map only');
  expect(warnings.some(text => /Tile unavailable/.test(text))).toBe(true);
});
test('header rules are scoped to the deployed base, never the whole origin', async () => {
  const headers = await readFile(new URL('../dist/_headers', import.meta.url), 'utf8');
  expect(headers.startsWith('/next/*\n')).toBe(true); expect(headers).not.toMatch(/^\/\*$/m);
  expect(headers).toContain('/next/sw.js\n  Cache-Control: no-cache');
  const budget = JSON.parse(await readFile(new URL('../dist/budgets.json', import.meta.url))); expect(budget.base).toBe('/next/');
});
test('the map is revealed when no chart can paint: outside coverage, or every tile failing', async ({ page }) => {
  await page.goto('./?date=1960-01-01&lat=0&lng=0&zoom=8'); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('body')).toHaveClass(/painted/); await expect(page.locator('#mapCanvas')).toHaveCSS('opacity', '1');
  await page.route('**/t/sectionals/**', route => route.fulfill({ status: 404, body: '' }));
  await page.goto(view); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('body')).toHaveClass(/painted/); await expect(page.locator('#loadingSplash')).toBeHidden();
});
