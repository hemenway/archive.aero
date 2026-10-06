import { test, expect, ready, view } from './fixtures.mjs';
test('dates, bounds, timeline keys, heat strip and scoped keyboard ownership', async ({ page }) => {
  await ready(page);
  await page.locator('#nextBtn').click(); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.locator('#prevBtn').click(); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  const timeline = page.getByRole('slider', { name: 'Historical chart date' });
  await timeline.focus();
  await page.keyboard.press('Home'); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.keyboard.press('ArrowLeft'); await expect(timeline).toHaveAttribute('aria-valuenow', '0');
  await expect(timeline).toHaveAttribute('aria-valuetext', /January 1, 1950/);
  await page.keyboard.press('PageUp'); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await page.keyboard.press('PageDown'); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.keyboard.press('End'); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await page.keyboard.press('ArrowRight'); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
  await expect(timeline).toHaveAttribute('aria-valuenow', await timeline.getAttribute('aria-valuemax'));
  await page.locator('#timeSelect').fill('1965-06-15'); await page.locator('#timeSelect').dispatchEvent('change');
  await expect(page.locator('#lblRange')).toHaveText('Full lower-48 coverage');
  // The focused map owns the arrow keys for panning; the date stays put.
  await page.locator('#shareBtn').click(); const before = new URL(await page.evaluate(() => window.__copied));
  await page.locator('#map').focus(); await page.keyboard.press('ArrowLeft'); await expect(page.locator('#timeSelect')).toHaveValue('1965-06-15');
  await page.locator('#shareBtn').click(); await expect.poll(async () => +new URL(await page.evaluate(() => window.__copied)).searchParams.get('lng')).toBeLessThan(+before.searchParams.get('lng'));
  await page.locator('#toolsBtn').click(); await expect(page.locator('#toolsBtn')).toHaveAttribute('aria-expanded', 'true');
  const opacity = page.getByRole('slider', { name: 'Chart opacity' });
  await opacity.focus(); await page.keyboard.press('ArrowLeft');
  await expect(opacity).toHaveValue('99'); await expect(page.locator('#toolOpacityValue')).toHaveText('99'); await expect(page.locator('#timeSelect')).toHaveValue('1965-06-15');
  await page.keyboard.press('Escape'); await expect(page.locator('#toolsBtn')).toBeFocused(); await expect(page.locator('#toolsBtn')).toHaveAttribute('aria-expanded', 'false');
  await page.locator('#timeSelect').focus(); await page.keyboard.press('Space'); await expect(page.locator('#playBtn')).toHaveAttribute('aria-label', 'Play or pause animation');
  expect(await page.locator('#heatCanvas').evaluate(c => c.getContext('2d').getImageData(5, 5, 1, 1).data[3])).toBeGreaterThan(0);
  await expect(page.locator('#ticksContainer .tick .tick-label').first()).toHaveText('1950');
});
test('playback wraps and pause stops; live regions are quiet while playing', async ({ page }) => {
  await ready(page, './?date=1970-01-01&lat=32.7767&lng=-96.797&zoom=10');
  await expect(page.locator('#loader')).toBeHidden(); await page.clock.install();
  await page.locator('#map').focus(); await page.keyboard.press('Space');
  await expect(page.locator('#playBtn')).toHaveAttribute('aria-label', 'Pause animation'); await expect(page.locator('#playBtn')).toHaveText('⏸');
  await expect(page.locator('#pinDates')).toHaveAttribute('aria-live', 'off');
  await page.clock.runFor(2100); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await page.locator('#playBtn').click(); await page.clock.runFor(5000); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await expect(page.locator('#playBtn')).toHaveAttribute('aria-label', 'Play animation'); await expect(page.locator('#playBtn')).toHaveText('▶');
  await expect(page.locator('#pinDates')).toHaveAttribute('aria-live', 'polite');
});
test('pin card, solo, exit, share and restoration in a fresh page', async ({ page, context }) => {
  await ready(page); await page.locator('#map').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('#pinPanel')).toBeFocused(); await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await expect(page.locator('#pinBadge')).toHaveText('ed. 1'); await expect(page.locator('#pinDates')).toHaveText('Jan 1960 – Jan 1970');
  await page.getByRole('button', { name: 'View alone', exact: true }).click(); await expect(page.locator('#soloBar')).toBeVisible();
  await expect(page.locator('#soloLabel')).toHaveText('Viewing Test Dallas · Jan 1960');
  await page.locator('#pinClose').click(); await expect(page.locator('#pinPanel')).toBeHidden(); await expect(page.locator('#soloBar')).toBeVisible();
  await page.locator('#soloExit').click(); await expect(page.locator('#soloBar')).toBeHidden();
  await page.goto(view); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await page.locator('#map').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.locator('#shareBtn').click(); await expect.poll(() => page.evaluate(() => window.__copied)).toBeTruthy(); const link = await page.evaluate(() => window.__copied);
  const url = new URL(link); expect(url.searchParams.get('pin')).toBe('32.7767,-96.7970'); expect(url.searchParams.get('date')).toBe('1960-01-01');
  const fresh = await context.newPage(); await ready(fresh, link); await expect(fresh.locator('#pinLoc')).toHaveText('Test Dallas');
  await fresh.locator('#shareBtn').click(); await expect.poll(() => fresh.evaluate(() => window.__copied)).toBe(link);
  await fresh.locator('#pinPanel').focus(); await fresh.keyboard.press('Escape'); await expect(fresh.locator('#pinPanel')).toBeHidden(); await expect(fresh.locator('#map')).toBeFocused();
});
test('a chart without its own artifact is shown alone from its era archive; any timeline change leaves solo', async ({ page }) => {
  await ready(page, './?date=1970-06-01&lat=32.7767&lng=-96.797&zoom=10');
  await page.locator('#map').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#pinDates')).toHaveText('Jan 1960 – Jan 1971');
  await page.getByRole('button', { name: 'View alone', exact: true }).click();
  await expect(page.locator('#soloLabel')).toHaveText('Viewing Test Dallas · Jan 1960'); await expect(page.locator('#soloBar')).toBeVisible();
  // Escape steps out of solo first, then clears the pin.
  await page.locator('#map').focus(); await page.keyboard.press('Escape'); await expect(page.locator('#soloBar')).toBeHidden(); await expect(page.locator('#pinPanel')).toBeVisible();
  await page.getByRole('button', { name: 'View alone', exact: true }).click(); await expect(page.locator('#soloBar')).toBeVisible();
  await page.locator('#prevBtn').click(); await expect(page.locator('#soloBar')).toBeHidden();
  // "View alone" turns hidden charts back on.
  await page.locator('#toolsBtn').click(); await page.locator('#chartsToggleBtn').click(); await expect(page.locator('#chartsToggleBtn')).toHaveAttribute('aria-pressed', 'false'); await page.locator('#toolsBtn').click();
  await page.getByRole('button', { name: 'View alone', exact: true }).click(); await expect(page.locator('#chartsToggleBtn')).toHaveAttribute('aria-pressed', 'true');
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
  await page.keyboard.press('Escape'); await expect(page.getByRole('dialog')).toBeHidden(); await expect(page.locator('#trackWrapper')).toBeFocused(); await expect(page.locator('#pinPanel')).toBeVisible();
  await expect(page.locator('#main-content')).not.toHaveAttribute('inert');
  await page.locator('#toolsBtn').click(); await page.locator('#toolOpacitySlider').focus(); await page.keyboard.press('Escape'); await expect(page.locator('#pinPanel')).toBeVisible();
  await page.locator('#map').focus(); await page.keyboard.press('Escape'); await expect(page.locator('#pinPanel')).toBeHidden();
  await page.locator('#menuToggle').click(); await expect(page.getByRole('navigation', { name: 'Site menu' })).toBeVisible(); await expect(page.locator('#menuToggle')).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('Escape'); await expect(page.locator('#siteMenu')).toBeHidden(); await expect(page.locator('#menuToggle')).toHaveAttribute('aria-expanded', 'false');
  // Opening one of the menu and the layers panel closes the other.
  await page.locator('#toolsBtn').click(); await expect(page.locator('#toolsPanel')).toBeVisible();
  await page.locator('#menuToggle').click(); await expect(page.locator('#toolsPanel')).toBeHidden(); await expect(page.locator('#siteMenu')).toBeVisible();
  await page.locator('#toolsBtn').click(); await expect(page.locator('#siteMenu')).toBeHidden(); await expect(page.locator('#toolsPanel')).toBeVisible();
});
test('airfield browser keyboard path, filters and persisted preferences', async ({ page, browserName }) => {
  await ready(page); await page.locator('#toolsBtn').click();
  // Airfields start off; the switch turns them on.
  await expect(page.locator('#airfieldsBtn')).toHaveAttribute('aria-pressed', 'false'); await page.locator('#airfieldsBtn').click();
  await page.locator('#afBrowser summary').focus(); await page.keyboard.press('Enter'); await expect(page.locator('#afSelect')).toBeEnabled(); await expect(page.locator('#afCount')).toHaveText('2 airfields in view.');
  await page.locator('#afSelect').focus(); await page.selectOption('#afSelect', '0'); await page.keyboard.press(browserName === 'webkit' ? 'Alt+Tab' : 'Tab'); await expect(page.locator('#afOpen')).toBeFocused(); await page.keyboard.press('Enter');
  await expect(page.locator('#afPanel')).toBeFocused(); await expect(page.locator('#afName')).toHaveText('Test Field'); await expect(page.locator('#afDates')).toHaveText('In operation 1930 – presentopen');
  await expect(page.locator('#afLinks a')).toHaveText(['OurAirports']); await expect(page.locator('#afLinks a')).toHaveAttribute('href', 'https://ourairports.com/airports/KDAL/');
  await page.keyboard.press('Escape'); await expect(page.locator('#afPanel')).toBeHidden(); await expect(page.locator('#toolsBtn')).toBeFocused();
  await page.locator('#toolsBtn').click(); await page.locator('[data-af-status=gone]').click(); await expect(page.locator('#afCount')).toHaveText('1 airfields in view.');
  await page.reload(); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true'); await page.locator('#toolsBtn').click(); await expect(page.locator('#airfieldsBtn')).toHaveAttribute('aria-pressed', 'true'); await expect(page.locator('[data-af-status=gone]')).toHaveAttribute('aria-pressed', 'false');
  await page.locator('#afBrowser summary').click(); await expect(page.locator('#afCount')).toHaveText('1 airfields in view.'); await page.locator('#airfieldsBtn').click(); await expect(page.locator('#afSelect')).toBeDisabled(); await expect(page.locator('#afCount')).toContainText('Turn on Airfields');
});
test('clicking an airfield dot opens its card without moving the pin; the name shows on hover', async ({ page, isMobile }) => {
  await ready(page); await page.locator('#toolsBtn').click(); await page.locator('#airfieldsBtn').click(); await page.locator('#toolsBtn').click();
  // The view is centred on Test Field.
  const box = await page.locator('#mapCanvas').boundingBox(), x = box.x + box.width / 2, y = box.y + box.height / 2;
  if (!isMobile) { await page.mouse.move(x, y); await expect(page.locator('.af-tip')).toHaveText('Test Field'); await page.mouse.move(x + 60, y + 90); await expect(page.locator('.af-tip')).toBeHidden(); }
  await page.mouse.click(x, y); await expect(page.locator('#afPanel')).toBeVisible(); await expect(page.locator('#afName')).toHaveText('Test Field'); await expect(page.locator('#pinPanel')).toBeHidden();
  await page.locator('#afClose').click(); await expect(page.locator('#afPanel')).toBeHidden();
  // A click on open map sets the pin instead.
  await page.mouse.click(x + 60, y + 90); await expect(page.locator('#pinPanel')).toBeVisible(); await expect(page.locator('#afPanel')).toBeHidden();
});
test('airspace switch, status line, pin stack and map credits', async ({ page }) => {
  await ready(page); await page.locator('#toolsBtn').click();
  await expect(page.locator('#airspaceBtn')).toHaveAttribute('aria-pressed', 'false'); await expect(page.locator('#asStatus')).toHaveText('US (FAA NASR), France (SIA), Brazil (DECEA)');
  const credits = '© OpenStreetMap contributors · Protomaps · airfields Freeman';
  await expect(page.locator('#mapAttribution')).toHaveText(credits);
  await page.locator('#airspaceBtn').click(); await expect(page.locator('#airspaceBtn')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#asStatus')).toHaveText('United States · FAA NASR cycle Jan 1, 1960'); await expect(page.locator('#mapAttribution')).toHaveText(`${credits}, airspace FAA NASR`);
  await page.locator('#map').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('#pinAirspace .pin-as-title')).toHaveText('Airspace · FAA NASR Jan 1, 1960');
  await expect(page.locator('#pinAirspace .pin-as-row')).toHaveText('DSFC – 2,500 MSLTEST FIELD'); await expect(page.locator('#pinAirspace .pin-as-cls')).toHaveClass(/pin-as-cls-D/);
  // Class A–D off: the row leaves the stack, and the choice persists. The layers panel is still open; on a phone the
  // pin card sits over it (as in production), so the chip is clicked directly.
  await page.locator('[data-as-key=bcd]').dispatchEvent('click'); await expect(page.locator('#pinAirspace .pin-as-row')).toHaveCount(0); await expect(page.locator('#pinAirspace .pin-as-note')).toHaveText('No class airspace here');
  await page.locator('#timeSelect').fill('1950-06-01'); await page.locator('#timeSelect').dispatchEvent('change'); await expect(page.locator('#asStatus')).toHaveText('United States · no FAA NASR data before Jan 1, 1960');
  await page.reload(); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true'); await expect(page.locator('#airspaceBtn')).toHaveAttribute('aria-pressed', 'true'); await expect(page.locator('[data-as-key=bcd]')).toHaveAttribute('aria-pressed', 'false');
  await page.locator('#toolsBtn').click(); await page.locator('#airspaceBtn').click(); await expect(page.locator('#mapAttribution')).toHaveText(credits);
});
test('zoom buttons, zoom readout and the layers footer', async ({ page }) => {
  await ready(page);
  await expect(page.locator('#chartInfoTotal')).toHaveText('4'); await expect(page.locator('#chartInfoEffective')).toHaveText('2'); await expect(page.locator('#chartInfoZoom')).toHaveText('10');
  await page.locator('#zoomInBtn').click(); await expect(page.locator('#chartInfoZoom')).toHaveText('11');
  await page.locator('#zoomOutBtn').click(); await page.locator('#zoomOutBtn').click(); await expect(page.locator('#chartInfoZoom')).toHaveText('9');
  await page.locator('#map').focus(); await page.keyboard.press('+'); await expect(page.locator('#chartInfoZoom')).toHaveText('10');
  await page.goto('./?date=1960-01-01&lat=32.7767&lng=-96.797&zoom=14'); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('#zoomInBtn')).toHaveClass(/leaflet-disabled/); await expect(page.locator('#zoomInBtn')).toHaveAttribute('aria-disabled', 'true'); await expect(page.locator('#zoomOutBtn')).toHaveAttribute('aria-disabled', 'false');
});
test('storage access failure still allows boot and toggles', async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(window, 'localStorage', { get() { throw new Error('Storage unavailable'); } }); });
  await ready(page); await page.locator('#toolsBtn').click(); await page.locator('#airfieldsBtn').click(); await expect(page.locator('#airfieldsBtn')).toHaveAttribute('aria-pressed', 'true');
});
test('timeline drag and touchcancel leave the timeline usable', async ({ page }) => {
  await ready(page); const track = page.locator('#trackWrapper'), box = await track.boundingBox();
  const move = x => page.evaluate(([x, y]) => window.dispatchEvent(new MouseEvent('mousemove', { clientX: x, clientY: y, bubbles: true, cancelable: true })), [x, box.y + 10]);
  await track.dispatchEvent('mousedown', { button: 0, clientX: box.x + 4, clientY: box.y + 10 }); await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01');
  await move(box.x + box.width / 2); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await page.evaluate(() => window.dispatchEvent(new Event('touchcancel')));
  await move(box.x + box.width); await page.waitForTimeout(80); await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
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
  await page.locator('#locateBtn').click(); await expect(page.locator('#toast')).toHaveText('Centered on your approximate location (IP)'); expect(await page.evaluate(() => window.__geoCalls)).toBe(1);
  await page.locator('#shareBtn').click(); const link = new URL(await page.evaluate(() => window.__copied)); expect(link.searchParams.get('lat')).toBe('40.5000'); expect(link.searchParams.get('lng')).toBe('-100.2500'); expect(link.searchParams.get('zoom')).toBe('10');
});
