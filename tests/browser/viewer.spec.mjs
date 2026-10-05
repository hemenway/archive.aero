import { test, expect } from '@playwright/test';
import { installFixtures } from './fixtures.mjs';

const view = '/?date=1960-01-01&lat=32.7767&lng=-96.7970&zoom=10';
async function ready(page, url = view, date = '1960-01-01') {
  await page.goto(url);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  await expect(page.locator('#timeSelect')).toHaveValue(date);
  await expect.poll(() => page.locator('#map canvas.leaflet-tile').evaluateAll(canvases =>
    canvases.some(canvas => canvas.getContext('2d').getImageData(128, 128, 1, 1).data[3] > 0)
  ), { message: 'a real PMTiles raster tile should render visible pixels' }).toBe(true);
}

test('bundle boot, date selection, timeline buttons and slider keyboard navigation', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await ready(page);
  await page.locator('#nextBtn').click();
  await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.locator('#prevBtn').click();
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await page.getByRole('slider', { name: 'Historical chart date' }).focus();
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.keyboard.press('End');
  await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.keyboard.press('Home');
  await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.locator('#timeSelect').fill('1965-06-15');
  await page.locator('#timeSelect').dispatchEvent('change');
  await expect(page.locator('#lblRange')).toHaveText('Full lower-48 coverage');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('pin, solo view, share link and restoration in a fresh page', async ({ page, context }) => {
  const diagnostics = await installFixtures(page);
  await ready(page);
  await page.locator('#map').click({ position: { x: 170, y: 250 } });
  await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.getByRole('button', { name: 'View alone', exact: true }).click();
  await expect(page.locator('#soloBar')).toBeVisible();
  await page.getByRole('button', { name: 'Show all charts', exact: true }).click();
  await expect(page.locator('#soloBar')).toBeHidden();
  await page.locator('#shareBtn').click();
  await expect.poll(() => page.evaluate(() => window.__copied)).toBeTruthy();
  const link = await page.evaluate(() => window.__copied);
  const url = new URL(link);
  expect(url.searchParams.get('date')).toBe('1960-01-01');
  expect(url.searchParams.get('pin')).toMatch(/^-?\d+\.\d+,-?\d+\.\d+$/);
  const fresh = await context.newPage();
  const freshDiagnostics = await installFixtures(fresh);
  await ready(fresh, link);
  await expect(fresh.locator('#pinLoc')).toHaveText('Test Dallas');
  await fresh.locator('#shareBtn').click();
  await expect.poll(() => fresh.evaluate(() => window.__copied)).toBe(link);
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
  expect(freshDiagnostics).toEqual({ errors: [], unexpected: [] });
});

test('play advances frames and pause stops playback', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await ready(page);
  await page.clock.install();
  await page.locator('#playBtn').click();
  await page.clock.runFor(2100);
  await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.locator('#playBtn').click();
  await page.clock.runFor(5000);
  await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('metadata outage falls back to CSV', async ({ page }) => {
  const diagnostics = await installFixtures(page, { bundleFails: true });
  await ready(page);
  await page.locator('#prevBtn').click();
  await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('fatal data outage shows actionable error rather than frozen progress', async ({ page }) => {
  const diagnostics = await installFixtures(page, { bundleFails: true, csvFails: true });
  await page.goto(view);
  await expect(page.locator('#loadingStatus')).toContainText('Failed to load chart data');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('real canvas flicker and stale-render guard', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await page.goto('/tests/flicker-regression-guard.html');
  await expect.poll(() => page.evaluate(() => window.__RESULT__), { timeout: 20000 }).toBeTruthy();
  const result = await page.evaluate(() => window.__RESULT__);
  expect(result.failures).toEqual([]);
  expect(result.passed).toBe(true);
  expect(result.total).toBeGreaterThan(5);
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('a loading pill shows above the timeline while chart tiles are in flight', async ({ page }) => {
  // CSV fallback mode reads each era from its own archive. Header reads
  // (range 0-) go through so boot completes; tile reads are held, so the
  // first frame is in flight when the splash lifts. (Stepping eras later
  // would not do: the scrub prefetcher has already warmed the neighbours.)
  const diagnostics = await installFixtures(page, { bundleFails: true });
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/sectionals/*.pmtiles', async route => {
    if (!/^bytes=0-/.test(route.request().headers().range || '')) await held;
    await route.fallback();
  });
  await page.goto(view);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  await expect(page.locator('#loader')).toBeVisible();
  await expect(page.locator('#loader')).toHaveText('Loading charts');
  release();
  await expect(page.locator('#loader')).toBeHidden();
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('without a share link the map opens on the lower 48 and never geolocates', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  const hosts = [];
  page.on('request', request => hosts.push(new URL(request.url()).hostname));
  await page.goto('/?date=1960-01-01');
  await expect(page.locator('#loadingSplash')).toBeHidden();
  await page.locator('#shareBtn').click();
  await expect.poll(() => page.evaluate(() => window.__copied)).toContain('zoom=');
  const link = new URL(await page.evaluate(() => window.__copied));
  // The lower-48 box is fitted, then clamped to the map's zoom floor: its
  // projected midpoint (about 38.0N 95.95W) at z6, on any viewport.
  expect(link.searchParams.get('zoom')).toBe('6');
  expect(Math.abs(parseFloat(link.searchParams.get('lat')) - 38.0)).toBeLessThan(0.6);
  expect(Math.abs(parseFloat(link.searchParams.get('lng')) + 95.95)).toBeLessThan(0.6);
  expect(hosts).not.toContain('get.geojs.io');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('the era index is read once, by the early fetch, and a share link never touches the newest era', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  const heads = [];
  const archives = new Set();
  page.on('request', request => {
    const url = new URL(request.url());
    const range = request.headers().range || '';
    if (url.pathname.endsWith('.bundle') && range.startsWith('bytes=0-')) heads.push(range);
    if (url.pathname.endsWith('.pmtiles') && !url.pathname.includes('/basemap/')) archives.add(url.pathname.split('/').pop());
  });
  await ready(page);
  // One head read, sized by CONFIG.bundleHeadBytes: the inline script's
  // request is adopted by MetaBundle.load rather than repeated.
  expect(heads).toEqual(['bytes=0-98303']);
  // ?date=1960-01-01 paints its own era only; boot used to paint the newest
  // frame (1970) first and read its tiles for nothing.
  expect([...archives]).toEqual(['1960-01-01_to_1970-01-01.pmtiles']);
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('the address bar follows the timeline, the map and the pin', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await page.goto('/');
  await expect(page.locator('#loadingSplash')).toBeHidden();
  // Untouched, a bare URL stays bare.
  await page.waitForTimeout(600);
  expect(new URL(page.url()).search).toBe('');
  await page.locator('#prevBtn').click();
  await expect(page).toHaveURL(/\?date=1960-01-01&lat=-?[\d.]+&lng=-?[\d.]+&zoom=6$/);
  await page.locator('#map').focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/&zoom=6&pin=-?[\d.]+,-?[\d.]+$/);
  // The address is a working permalink: a reload restores date and pin.
  await page.reload();
  await expect(page.locator('#loadingSplash')).toBeHidden();
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await expect(page.locator('#pinPanel')).toBeVisible();
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('playback skips editions that change nothing in view', async ({ page }) => {
  // Three eras; only the first and last cover Dallas. Playing from the
  // first must jump straight to the last, not sit on the Alaska-only era.
  const dallas = [-98, 31, -95, 34];
  const diagnostics = await installFixtures(page, { eraBounds: [dallas, [-150, 60, -140, 65], dallas] });
  await ready(page, '/?date=1950-01-01&lat=32.7767&lng=-96.7970&zoom=10', '1950-01-01');
  await page.clock.install();
  await page.locator('#playBtn').click();
  await page.clock.runFor(2100);
  await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.locator('#playBtn').click();
  // The step buttons still move one edition at a time.
  await page.locator('#prevBtn').click();
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});
