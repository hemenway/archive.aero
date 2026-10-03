import { readFile } from 'node:fs/promises';
import { test, expect, ready, view } from './fixtures.mjs';
test('boot requests are bounded, early requests adopted, no third party fonts, size budgets', async ({ page, guard }) => {
  await page.addInitScript(() => { window.addEventListener('first-chart-paint', () => { window.__paintResources = performance.getEntriesByType('resource').map(r => r.name); }); });
  await ready(page, './');
  const resources = await page.evaluate(() => window.__paintResources);
  expect(resources.length).toBeLessThanOrEqual(50);
  const tiles = guard.requests.filter(u => u.includes('/t/')); expect(new Set(tiles).size).toBe(tiles.length);
  expect(guard.requests.some(u => /fonts\.(googleapis|gstatic)\.com/.test(u))).toBe(false);
  await expect(page.locator('link[rel=preload][as=fetch]')).toHaveAttribute('crossorigin', '');
  await expect(page.locator('link[rel=modulepreload]')).toHaveCount(1);
  const budget = JSON.parse(await readFile(new URL('../dist/budgets.json', import.meta.url)));
  for (const [name, max] of Object.entries(budget.limits)) expect(budget.bytes[name]).toBeLessThanOrEqual(max);
  expect(budget.bytes.inlineBoot).toBeLessThanOrEqual(1536);
});
test('manifest failure exposes an actionable error with static chrome available', async ({ page }) => {
  await page.route('**/manifest.*.json', route => route.fulfill({ status: 503, body: '' }));
  await page.goto(view); await expect(page.locator('#fatalError')).toContainText('Manifest failed to load'); await expect(page.locator('#loader')).toBeHidden();
  await page.locator('#helpBtn').click(); await expect(page.getByRole('dialog')).toBeVisible();
});
test('renderer unsupported error links to the current viewer', async ({ page }) => {
  await page.addInitScript(() => { const get = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function(type, ...args) { return this.id === 'mapCanvas' ? null : get.call(this, type, ...args); }; });
  await page.goto(view); await expect(page.locator('#fatalError')).toContainText('cannot run the new chart renderer'); await expect(page.locator('#fatalError a')).toHaveAttribute('href', '/');
});
test('worker failure leaves a visible error and stops playback', async ({ page }) => {
  await page.addInitScript(() => { const Original = window.Worker; window.Worker = class extends Original { constructor(...args) { super(...args); setTimeout(() => { this.dispatchEvent(new ErrorEvent('error', { message: 'fixture worker failed', cancelable: true })); }, 150); } }; });
  await page.goto(view); await expect(page.locator('#fatalError')).toContainText('Data worker failed'); await expect(page.locator('#loader')).toBeHidden();
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
  await expect(page.locator('#warningOverlay')).toBeHidden(); await page.clock.runFor(1500); await expect(page.locator('#warningOverlay')).toBeVisible();
  await page.clock.runFor(6500); await expect(page.locator('#warningOverlay')).toBeHidden();
});
test('URL replaceState is throttled to four per second during scrubbing', async ({ page }) => {
  await ready(page); await page.addInitScript(() => {});
  await page.evaluate(() => { const original = history.replaceState.bind(history); window.__replacements = []; history.replaceState = (...args) => { window.__replacements.push(performance.now()); original(...args); }; });
  for (let i = 0; i < 10; i++) { await page.locator('#timeSelect').fill(i % 2 ? '1960-01-01' : '1970-01-01'); await page.locator('#timeSelect').dispatchEvent('change'); }
  await page.waitForTimeout(350); const times = await page.evaluate(() => window.__replacements); for (let i = 1; i < times.length; i++) expect(times[i] - times[i - 1]).toBeGreaterThanOrEqual(245);
});
test('a failing tile degrades the view instead of ending it', async ({ page }) => {
  const warnings = []; page.on('console', message => { if (message.type() === 'warning') warnings.push(message.text()); });
  let failed = false;
  await page.route('**/t/sectionals/**', route => { if (failed) return route.fallback(); failed = true; return route.fulfill({ status: 404, body: '' }); });
  await ready(page); await expect(page.locator('#fatalError')).toBeHidden();
  expect(warnings.some(text => /Tile unavailable/.test(text))).toBe(true);
});
test('header rules are scoped to the deployed base, never the whole origin', async () => {
  const headers = await readFile(new URL('../dist/_headers', import.meta.url), 'utf8');
  expect(headers.startsWith('/next/*\n')).toBe(true); expect(headers).not.toMatch(/^\/\*$/m);
  expect(headers).toContain('/next/sw.js\n  Cache-Control: no-cache');
  const budget = JSON.parse(await readFile(new URL('../dist/budgets.json', import.meta.url))); expect(budget.base).toBe('/next/');
});
