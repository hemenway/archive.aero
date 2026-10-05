import { test, expect } from '@playwright/test';
// The view sits inside fixture era 0's z8 tile (8/60/100, red) and inside pin
// shard 391's ring, away from the tile's transparent border and white diagonal.
const view = './?date=1951-01-01&lat=36.3&lng=-95.5&zoom=8';
const DATA = 'http://127.0.0.1:4185';

async function boot(page, url = view) {
  const problems = [], responses = [];
  page.on('pageerror', error => problems.push(`pageerror: ${error.message}`));
  page.on('console', message => { if (message.type() === 'error') problems.push(`console: ${message.text()}`); });
  page.on('response', response => { if (response.url().startsWith(DATA)) responses.push({ status: response.status(), url: response.url(), encoding: response.headers()['content-encoding'] }); });
  await page.route('https://plausible.io/**', route => route.fulfill({ contentType: 'application/javascript', body: '' }));
  await page.goto(url);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('body')).toHaveClass(/painted/);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  return { problems, responses };
}
// The WebGL canvas does not preserve its drawing buffer, so pixels are read from a screenshot.
async function centrePixel(page) {
  await page.addStyleTag({ content: '#warningOverlay,#toast,#pinPanel{display:none!important}' });
  const box = await page.locator('#mapCanvas').boundingBox(), shot = (await page.screenshot()).toString('base64');
  return page.evaluate(async ({ shot, x, y, width }) => {
    const bitmap = await createImageBitmap(new Blob([Uint8Array.from(atob(shot), c => c.charCodeAt(0))], { type: 'image/png' }));
    const scale = bitmap.width / width, canvas = new OffscreenCanvas(bitmap.width, bitmap.height), context = canvas.getContext('2d');
    context.drawImage(bitmap, 0, 0); return Array.from(context.getImageData(Math.round(x * scale), Math.round(y * scale), 1, 1).data.slice(0, 3));
  }, { shot, x: box.x + box.width / 2, y: box.y + box.height / 2, width: page.viewportSize().width });
}

test('real modules boot under the build CSP, fetch tiles from the Worker endpoint and paint chart pixels', async ({ page }) => {
  const { problems, responses } = await boot(page);
  await expect(page.locator('#loader')).toBeHidden();
  const tiles = responses.filter(r => r.url.includes('/t/'));
  expect(tiles.some(r => r.status === 200 && r.url.includes('/t/sectionals/1948-01-01_to_1952-01-01.'))).toBe(true);
  expect(tiles.filter(r => ![200, 204].includes(r.status))).toEqual([]);
  expect(new Set(tiles.map(r => r.url)).size).toBe(tiles.length); // early fetches adopted, nothing requested twice
  expect(page.workers().some(worker => /\/worker\.[A-Z0-9]+\.js$/.test(worker.url()))).toBe(true); // the data plane runs in its Worker, not the main-thread fallback
  await expect.poll(() => centrePixel(page)).toEqual([205, 70, 64]);
  expect(problems).toEqual([]);
});

test('a newer date swaps to that era; an era with no tile here leaves the basemap', async ({ page }) => {
  const { problems } = await boot(page);
  await page.locator('#timeSelect').fill('1953-06-01'); await page.locator('#timeSelect').dispatchEvent('change');
  // 1953-06: era 0 has ended; eras 1-3 cover other longitudes, so the centre shows no chart colour.
  await expect.poll(async () => (await centrePixel(page)).join()).not.toBe('205,70,64');
  await page.locator('#timeSelect').fill('1951-01-01'); await page.locator('#timeSelect').dispatchEvent('change');
  await expect.poll(() => centrePixel(page)).toEqual([205, 70, 64]);
  expect(problems).toEqual([]);
});

test('airspace, airfields and pins come through the real data plane', async ({ page }) => {
  const { problems, responses } = await boot(page);
  await page.locator('#toolsBtn').click(); await page.locator('#airspaceBtn').click();
  await expect(page.locator('#asStatus')).toContainText('United States · FAA NASR cycle');
  await expect.poll(() => responses.some(r => r.url.includes('/t/airspace/') && r.url.endsWith('/metadata') && r.status === 200)).toBe(true);
  await expect(page.locator('#mapAttribution')).toContainText('airspace FAA NASR');
  await page.locator('#airfieldsBtn').click(); await page.locator('#afBrowser summary').click();
  await expect(page.locator('#afCount')).toContainText('airfields in view');
  // The browser's first entry opens the production airfield card (and closes the layers panel).
  await page.locator('#afOpen').click(); await expect(page.locator('#afPanel')).toBeVisible();
  await expect(page.locator('#afName')).toContainText('Fixture field'); await expect(page.locator('#afDates')).not.toBeEmpty();
  await page.keyboard.press('Escape'); await expect(page.locator('#afPanel')).toBeHidden();
  await page.locator('#map').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('#pinLoc')).toHaveText('Fixture 391'); await expect(page.locator('#pinBadge')).toHaveText('ed. 1');
  // The fixture artifact holds z8 only and solo view flies out to fit the chart, below z8: the tiles asked for at the
  // start may be cancelled before they answer, so the request, not its response, is what shows the artifact was used.
  const asked = []; page.on('request', request => asked.push(request.url()));
  await page.getByRole('button', { name: 'View alone', exact: true }).click(); await expect(page.locator('#soloBar')).toBeVisible();
  await expect(page.locator('#soloLabel')).toHaveText('Viewing Fixture 391 · Jan 1950');
  await expect.poll(() => asked.some(url => url.includes('/t/sectionals/chart/fixture/1950-01-01.'))).toBe(true);
  await page.locator('#soloExit').click(); await expect(page.locator('#soloBar')).toBeHidden();
  expect(problems).toEqual([]);
});
