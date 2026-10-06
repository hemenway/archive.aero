import { test, expect, ready } from './fixtures.mjs';
test.use({ serviceWorkers: 'allow' });
test('service worker precaches the shell and manifest, cleans old versions and excludes tiles', async ({ page, context, browserName }) => {
  await ready(page);
  // Cache families are per registration directory: an old version of this shell
  // is swept, a root-scoped worker's family (`archive-next-_-…`) is left alone.
  await page.evaluate(async () => {
    await caches.open('archive-next-_next_-obsolete'); await caches.open('archive-next-_-root');
    await navigator.serviceWorker.register('./sw.js');
    await navigator.serviceWorker.ready;
  });
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  const result = await page.evaluate(async () => {
    const names = await caches.keys(), cache = await caches.open(names.find(n => n.startsWith('archive-next-_next_-')));
    return { names, paths: (await cache.keys()).map(r => new URL(r.url).pathname) };
  });
  expect(result.names).not.toContain('archive-next-_next_-obsolete'); expect(result.names).toContain('archive-next-_-root');
  expect(result.paths.some(p => p.includes('/manifest.'))).toBe(true); expect(result.paths.some(p => p.includes('/worker.'))).toBe(true);
  expect(result.paths.some(p => p.includes('/t/'))).toBe(false);
  // Playwright WebKit's offline reload returns an internal engine error.
  // Exercise the same navigation fallback with an HTTP outage there.
  if (browserName === 'webkit') await page.goto('./?__testOffline=1', { waitUntil: 'domcontentloaded' });
  else { await context.setOffline(true); await page.reload({ waitUntil: 'domcontentloaded' }); }
  await expect(page.locator('#menuToggle')).toBeVisible(); await page.locator('#menuToggle').click(); await expect(page.locator('#siteMenu')).toBeVisible();
  await context.setOffline(false);
});
