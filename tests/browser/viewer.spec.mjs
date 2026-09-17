import { test, expect } from '@playwright/test';
import { installFixtures } from './fixtures.mjs';

const view = '/?date=1960-01-01&lat=32.7767&lng=-96.7970&zoom=10';
async function ready(page, url = view) {
  await page.goto(url);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
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
