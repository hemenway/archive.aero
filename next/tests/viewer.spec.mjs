import { test, expect, ready, view } from './fixtures.mjs';
test('dates, bounds, timeline keys, heat strip and scoped keyboard ownership', async ({ page }) => {
  await ready(page);
  await page.locator('#nextBtn').click(); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.locator('#prevBtn').click(); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await page.locator('#trackWrapper').focus();
  await page.keyboard.press('Home'); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.keyboard.press('ArrowLeft'); await expect(page.locator('#trackWrapper')).toHaveAttribute('aria-valuenow', '0');
  await expect(page.locator('#trackWrapper')).toHaveAttribute('aria-valuetext', /January 1, 1950/);
  await page.keyboard.press('PageUp'); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await page.keyboard.press('PageDown'); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.keyboard.press('End'); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.keyboard.press('ArrowRight'); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.locator('#timeSelect').fill('1965-06-15'); await page.locator('#timeSelect').dispatchEvent('change');
  await expect(page.locator('#lblRange')).toHaveText('Full lower-48 coverage');
  await page.locator('#map').focus(); await page.keyboard.press('ArrowLeft'); await expect(page.locator('#timeSelect')).toHaveValue('1965-06-15');
  await page.locator('#toolsBtn').click(); await page.locator('#toolOpacitySlider').focus(); await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#toolOpacitySlider')).toHaveValue('99'); await expect(page.locator('#timeSelect')).toHaveValue('1965-06-15');
  await page.keyboard.press('Escape'); await expect(page.locator('#toolsBtn')).toBeFocused();
  await page.locator('#timeSelect').focus(); await page.keyboard.press('Space'); await expect(page.locator('#playBtn')).toHaveAttribute('aria-pressed', 'false');
  expect(await page.locator('#heatCanvas').evaluate(c => c.getContext('2d').getImageData(5, 5, 1, 1).data[3])).toBeGreaterThan(0);
});
test('playback wraps and pause stops; live regions are quiet while playing', async ({ page }) => {
  await ready(page, './?date=1970-01-01&lat=32.7767&lng=-96.797&zoom=10');
  await expect(page.locator('#loader')).toBeHidden(); await page.clock.install();
  await page.locator('#map').focus(); await page.keyboard.press('Space');
  await expect(page.locator('#playBtn')).toHaveAccessibleName('Pause timeline'); await expect(page.locator('#timelineStatus')).toHaveAttribute('aria-live', 'off');
  await page.clock.runFor(2100); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.locator('#playBtn').click(); await page.clock.runFor(5000); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await expect(page.locator('#timelineStatus')).toHaveAttribute('aria-live', 'polite');
});
test('pin, solo, exit, share and restoration in a fresh page', async ({ page, context }) => {
  await ready(page); await page.locator('#map').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('#pinPanel')).toBeFocused(); await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.getByRole('button', { name: 'View alone', exact: true }).click(); await expect(page.locator('#soloBar')).toBeVisible();
  await page.locator('#pinClose').click(); await expect(page.locator('#soloBar')).toBeVisible();
  await page.locator('#soloExit').click(); await expect(page.locator('#soloBar')).toBeHidden();
  await page.locator('#map').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.locator('#shareBtn').click(); await expect.poll(() => page.evaluate(() => window.__copied)).toBeTruthy(); const link = await page.evaluate(() => window.__copied);
  const url = new URL(link); expect(url.searchParams.get('pin')).toBe('32.7767,-96.7970'); expect(url.searchParams.get('date')).toBe('1960-01-01');
  const fresh = await context.newPage(); await ready(fresh, link); await expect(fresh.locator('#pinLoc')).toHaveText('Test Dallas');
  await fresh.locator('#shareBtn').click(); await expect.poll(() => fresh.evaluate(() => window.__copied)).toBe(link);
  await fresh.locator('#pinPanel').focus(); await fresh.keyboard.press('Escape'); await expect(fresh.locator('#pinPanel')).toBeHidden(); await expect(fresh.locator('#map')).toBeFocused();
});
test('default fits the lower 48 without geolocation; share lat/lng without zoom and date clamping', async ({ page, guard }) => {
  await ready(page, './'); await page.locator('#shareBtn').click(); const url = new URL(await page.evaluate(() => window.__copied));
  expect(Math.abs(+url.searchParams.get('lat') - 38)).toBeLessThan(0.6); expect(+url.searchParams.get('lng')).toBeCloseTo(-95.95, 1); expect(+url.searchParams.get('zoom')).toBeGreaterThanOrEqual(4); expect(+url.searchParams.get('zoom')).toBeLessThanOrEqual(6);
  expect(await page.evaluate(() => window.__geoCalls)).toBe(0); expect(guard.requests.some(u => u.includes('geojs'))).toBe(false);
  await ready(page, './?lat=32.7767&lng=-96.797&date=1800-01-01'); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.locator('#shareBtn').click(); expect(new URL(await page.evaluate(() => window.__copied)).searchParams.get('zoom')).toBe('10');
  await ready(page, './?date=2100-01-01'); await expect(page.locator('#timeSelect')).toHaveValue('1970-12-31');
});
test('modal traps focus, restores it and Escape closes one surface', async ({ page }) => {
  await ready(page); await page.locator('#map').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.locator('#trackWrapper').focus(); await page.keyboard.press('?'); await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.locator('#main-content')).toHaveAttribute('inert', ''); await expect(page.locator('#closeShortcuts')).toBeFocused();
  await page.keyboard.press('Tab'); await expect(page.locator('#closeShortcuts')).toBeFocused(); await page.keyboard.press('Shift+Tab'); await expect(page.locator('#closeShortcuts')).toBeFocused();
  await page.keyboard.press('Escape'); await expect(page.locator('#trackWrapper')).toBeFocused(); await expect(page.locator('#pinPanel')).toBeVisible();
  await page.locator('#toolsBtn').click(); await page.locator('#toolOpacitySlider').focus(); await page.keyboard.press('Escape'); await expect(page.locator('#pinPanel')).toBeVisible();
  await page.locator('#map').focus(); await page.keyboard.press('Escape'); await expect(page.locator('#pinPanel')).toBeHidden();
  await page.locator('#menuToggle').click(); await expect(page.getByRole('navigation', { name: 'Site menu' })).toBeVisible(); await page.keyboard.press('Escape'); await expect(page.locator('#siteMenu')).toBeHidden();
});
test('airfield browser keyboard path, filters and persisted preferences', async ({ page, browserName }) => {
  await ready(page); await page.locator('#toolsBtn').click(); await page.locator('#airfieldsBtn').click();
  await page.locator('#afBrowser summary').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#afSelect')).toBeEnabled(); await expect(page.locator('#afCount')).toHaveText('2 airfields in view.');
  await page.locator('#afSelect').focus(); await page.selectOption('#afSelect', '0'); await page.keyboard.press(browserName === 'webkit' ? 'Alt+Tab' : 'Tab'); await expect(page.locator('#afOpen')).toBeFocused(); await page.keyboard.press('Enter');
  await expect(page.locator('#afPanel')).toBeFocused(); await expect(page.locator('#afName')).toHaveText('Test Field'); await expect(page.locator('#afDates')).toHaveText('In operation 1930 – present');
  await page.keyboard.press('Escape'); await expect(page.locator('#afPanel')).toBeHidden(); await expect(page.locator('#toolsBtn')).toBeFocused();
  await page.locator('#toolsBtn').click(); await page.locator('[data-af-status=gone]').click(); await expect(page.locator('#afCount')).toHaveText('1 airfields in view.');
  await page.reload(); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true'); await page.locator('#toolsBtn').click(); await expect(page.locator('#airfieldsBtn')).toHaveAttribute('aria-checked', 'true'); await expect(page.locator('[data-af-status=gone]')).toHaveAttribute('aria-pressed', 'false');
  await page.locator('#afBrowser summary').click(); await expect(page.locator('#afCount')).toHaveText('1 airfields in view.'); await page.locator('#airfieldsBtn').click(); await expect(page.locator('#afSelect')).toBeDisabled(); await expect(page.locator('#afCount')).toContainText('Turn on Airfields');
});
test('storage access failure still allows boot and toggles', async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(window, 'localStorage', { get() { throw new Error('Storage unavailable'); } }); });
  await ready(page); await page.locator('#toolsBtn').click(); await page.locator('#airfieldsBtn').click(); await expect(page.locator('#airfieldsBtn')).toHaveAttribute('aria-checked', 'true');
});
test('pointer drag and touchcancel leave timeline usable', async ({ page }) => {
  await ready(page); const track = page.locator('#trackWrapper'), box = await track.boundingBox();
  await track.dispatchEvent('pointerdown', { pointerId: 1, clientX: box.x + 4, clientY: box.y + 10, button: 0 });
  await track.dispatchEvent('pointermove', { pointerId: 1, clientX: box.x + box.width / 2, clientY: box.y + 10 });
  await page.waitForTimeout(50); await page.evaluate(() => window.dispatchEvent(new Event('touchcancel'))); const date = await page.locator('#timeSelect').inputValue();
  await track.dispatchEvent('pointermove', { pointerId: 1, clientX: box.x + box.width, clientY: box.y + 10 }); await page.waitForTimeout(50); await expect(page.locator('#timeSelect')).toHaveValue(date);
  await track.focus(); await page.keyboard.press('End'); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
});
test('skip link focuses the viewer; filters and opacity fetch no new tiles', async ({ page, browserName, guard }) => {
  await ready(page); await page.keyboard.press(browserName === 'webkit' ? 'Alt+Tab' : 'Tab'); await expect(page.locator('.skip-link')).toBeFocused(); await page.keyboard.press('Enter'); await expect(page.locator('#main-content')).toBeFocused();
  await expect(page.locator('#loader')).toBeHidden(); const before = guard.requests.filter(u => u.includes('/t/')).length;
  await page.locator('#toolsBtn').click(); await page.locator('#toolOpacitySlider').fill('60'); await page.locator('#toolOpacitySlider').dispatchEvent('input');
  await page.locator('[data-af-status=gone]').click(); await page.locator('[data-as-key=e]').click();
  await page.waitForTimeout(150); expect(guard.requests.filter(u => u.includes('/t/')).length).toBe(before);
});
test('Locate falls back from GPS to IP only on request', async ({ page, guard }) => {
  const url = 'https://get.geojs.io/v1/ip/geo.json'; guard.allowed.add(url);
  await page.route(url, route => route.fulfill({ json: { latitude: '40.5', longitude: '-100.25' } }));
  await ready(page); expect(await page.evaluate(() => window.__geoCalls)).toBe(0);
  await page.locator('#locateBtn').click(); await expect(page.locator('#toast')).toContainText('(IP)'); expect(await page.evaluate(() => window.__geoCalls)).toBe(1);
  await page.locator('#shareBtn').click(); const link = new URL(await page.evaluate(() => window.__copied)); expect(link.searchParams.get('lat')).toBe('40.5000'); expect(link.searchParams.get('lng')).toBe('-100.2500'); expect(link.searchParams.get('zoom')).toBe('10');
});
