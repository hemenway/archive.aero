import { test, expect, ready } from './fixtures.mjs';
// The chart view: "View alone" opens one chart at its edition-page address
// (/next/sectionals/chart/<slug>/<date>), with the panel, title, share link
// and history behaving like a page of its own; edition pages also boot directly.
const PAGE = '/next/sectionals/chart/test_dallas/1960-01-01';

test('View alone opens the chart view at the edition address; links, back, share and exit', async ({ page }) => {
  await ready(page); await page.locator('#map').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.getByRole('button', { name: 'View alone', exact: true }).click();
  const panel = page.locator('#chartView');
  await expect(panel).toBeVisible(); await expect(panel).toBeFocused();
  await expect(panel.locator('.cv-title')).toHaveText('Test Dallas');
  await expect(panel.locator('.cv-dates')).toHaveText('In effect January 1, 1960 – January 1, 1970');
  await expect(panel.locator('.cv-pos')).toHaveText('2 of 3');
  await expect(page.locator('#soloBar')).toBeHidden(); await expect(page.locator('#pinPanel')).toBeHidden();
  await expect(page).toHaveURL(url => url.pathname === PAGE);
  await expect(page).toHaveTitle('Test Dallas sectional chart, January 1960 (edition 1) · archive.aero');
  await expect(page.locator('link[rel=canonical]')).toHaveAttribute('href', `http://127.0.0.1:4183${PAGE}`);
  // The share link is the edition page plus the view; no date, no pin.
  await panel.getByRole('button', { name: 'Copy link' }).click();
  const shared = new URL(await page.evaluate(() => window.__copied));
  expect(shared.pathname).toBe(PAGE); expect(shared.searchParams.get('date')).toBeNull(); expect(shared.searchParams.get('lat')).not.toBeNull();
  // The earlier edition is a link: same location, the timeline follows, history grows.
  await panel.locator('a.cv-prev').click();
  await expect(page).toHaveURL(url => url.pathname === '/next/sectionals/chart/test_dallas/1950-01-01');
  await expect(page.locator('#timeSelect')).toHaveValue('1950-01-01'); await expect(panel.locator('.cv-pos')).toHaveText('1 of 3');
  await expect(panel.locator('.cv-prev')).toHaveText('Earliest held'); await expect(panel).toBeVisible();
  // The untiled 1970 edition is listed, not linked.
  await panel.locator('.cv-all summary').click();
  await expect(panel.locator('.cv-list li').first()).toContainText('not yet tiled');
  await page.goBack();
  await expect(page).toHaveURL(url => url.pathname === PAGE); await expect(panel.locator('.cv-pos')).toHaveText('2 of 3');
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  // Leaving by the panel: the address is the viewer's permalink of this view again.
  await panel.getByRole('button', { name: 'Show all charts' }).click();
  await expect(panel).toBeHidden(); await expect(page.locator('#soloBar')).toBeHidden();
  await expect(page).toHaveURL(url => url.pathname === '/next/' && url.searchParams.get('date') === '1960-01-01' && url.searchParams.has('zoom'));
  await expect(page).toHaveTitle(/archive.aero - Historical/);
  await expect(page.locator('link[rel=canonical]')).toHaveAttribute('href', 'https://archive.aero/');
});

test('an edition address boots on its chart; Escape and a timeline change leave the view', async ({ page }) => {
  await ready(page, PAGE);
  const panel = page.locator('#chartView');
  await expect(panel).toBeVisible(); await expect(panel.locator('.cv-title')).toHaveText('Test Dallas');
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01'); await expect(page.locator('#soloBar')).toBeHidden();
  await expect(page).toHaveTitle(/Test Dallas sectional chart, January 1960/);
  // The map stays on the chart (fitted to its extent), not on the lower 48.
  await page.locator('#shareBtn').click(); const shared = new URL(await page.evaluate(() => window.__copied));
  expect(+shared.searchParams.get('zoom')).toBeGreaterThan(6); expect(shared.pathname).toBe(PAGE);
  await panel.focus(); await page.keyboard.press('Escape');
  await expect(panel).toBeHidden(); await expect(page.locator('#map')).toBeFocused();
  await expect(page).toHaveURL(url => url.pathname === '/next/' && url.searchParams.get('date') === '1960-01-01');
  // Back in: a timeline step leaves the chart view like it leaves solo.
  await page.locator('#map').focus(); await page.keyboard.press('Enter');
  await page.getByRole('button', { name: 'View alone', exact: true }).click(); await expect(panel).toBeVisible();
  await page.locator('#nextBtn').click();
  await expect(panel).toBeHidden(); await expect(page).toHaveURL(url => url.pathname === '/next/' && url.searchParams.get('date') === '1970-01-01');
});

test('a shared view on an edition address is kept; an untiled date says so', async ({ page }) => {
  await ready(page, `${PAGE}?lat=40&lng=-100&zoom=5`);
  await expect(page.locator('#chartView')).toBeVisible();
  await page.locator('#shareBtn').click(); const shared = new URL(await page.evaluate(() => window.__copied));
  expect(shared.searchParams.get('lat')).toBe('40.0000'); expect(shared.searchParams.get('zoom')).toBe('5');
  await ready(page, '/next/sectionals/chart/test_dallas/1970-01-01');
  await expect(page.locator('#toast')).toHaveText('That chart is not in the archive');
  await expect(page.locator('#chartView')).toBeHidden(); await expect(page.locator('#timeSelect')).toHaveValue('1970-01-01');
});
