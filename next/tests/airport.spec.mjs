import { test, expect, ready } from './fixtures.mjs';
// The airport page: /next/airports/<ident> boots the viewer on the airport with the
// editions-by-decade panel; an edition link opens the chart view; back returns.
const PAGE = '/next/airports/ktst';

test('an airport address boots centred on the airport with its editions; links, back, exit', async ({ page }) => {
  await ready(page, PAGE);
  const panel = page.locator('#chartView');
  await expect(panel).toBeVisible();
  await expect(panel.locator('.cv-title')).toHaveText('Test Field International Airport');
  await expect(panel.locator('.cv-dates')).toHaveText('3 chart editions, January 1950 – January 1970');
  await expect(page).toHaveTitle('Test Field International Airport on aeronautical charts, 1950–1970 · archive.aero');
  await expect(page.locator('link[rel=canonical]')).toHaveAttribute('href', `http://127.0.0.1:4183${PAGE}`);
  await expect(page).toHaveURL(url => url.pathname === PAGE);
  // Centred on the airport at the airport zoom (no server-rendered aside on the dev server: the record fits the view).
  await page.locator('#shareBtn').click(); const shared = new URL(await page.evaluate(() => window.__copied));
  expect(Math.abs(+shared.searchParams.get('lat') - 32.8471)).toBeLessThan(0.01); expect(+shared.searchParams.get('zoom')).toBe(10);
  // The newest decade is open and lists the untiled edition without a link.
  await expect(panel.locator('.ap-decade[open] summary')).toContainText('1970s');
  await expect(panel.locator('.ap-decade[open] .pin-unavailable')).toContainText('not yet tiled');
  // An edition link opens the chart view at its own address; back returns to the airport page.
  await panel.locator('.ap-decade').nth(1).locator('summary').click();
  await panel.locator('a[data-ap-chart="test_dallas/1960-01-01"]').click();
  await expect(page).toHaveURL(url => url.pathname === '/next/sectionals/chart/test_dallas/1960-01-01');
  await expect(panel.locator('.cv-kicker')).toHaveText('Chart view');
  await page.goBack();
  await expect(page).toHaveURL(url => url.pathname === PAGE);
  await expect(panel.locator('.cv-kicker')).toHaveText('Airport');
  // Leaving by the panel: the viewer's permalink.
  await panel.getByRole('button', { name: 'Show all charts' }).click();
  await expect(panel).toBeHidden();
  await expect(page).toHaveURL(url => url.pathname === '/next/' && url.searchParams.has('date'));
  await expect(page).toHaveTitle(/archive.aero - Historical/);
});

test('a timeline step leaves the airport page; an unknown airport is a toast', async ({ page }) => {
  await ready(page, PAGE);
  const panel = page.locator('#chartView'); await expect(panel).toBeVisible();
  await page.locator('#nextBtn').click();
  await expect(panel).toBeHidden(); await expect(page).toHaveURL(url => url.pathname === '/next/');
  await ready(page, '/next/airports/kxyz');
  await expect(page.locator('#chartView')).toBeHidden();
  await expect(page.locator('#toast')).toContainText('not in the archive');
});
