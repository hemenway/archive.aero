import { test as base, expect } from '@playwright/test';
// Fail every test on unexpected requests and uncaught application errors.
export const test = base.extend({
  guard: [async ({ page, context }, use) => {
    const errors = [], unexpected = [], requests = [], allowed = new Set();
    const attach = async target => {
      target.on('pageerror', e => errors.push(e.message));

      await target.addInitScript(() => {
        window.__geoCalls = 0;
        Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => window.__copied = text } });
        Object.defineProperty(navigator, 'geolocation', { configurable: true, value: { getCurrentPosition(ok, fail) { window.__geoCalls++; fail(new Error('GPS denied')); } } });
      });
    };
    await attach(page); context.on('page', target => attach(target));
    context.on('request', req => { const url = new URL(req.url()); requests.push(url.href); if (url.protocol === 'blob:' || url.hostname === 'plausible.io' && url.pathname.startsWith('/js/') || allowed.has(url.href)) return; if (!(url.origin === 'http://127.0.0.1:4183' && (url.pathname.startsWith('/next/t/') || /^\/next\/(?:index\.html|sw\.js|(?:app|chunk|worker)\.[\w-]+\.js|styles\.[\w-]+\.css|manifest\.[\w-]+\.json)?$/.test(url.pathname)))) unexpected.push(url.href); });
    await context.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname === 'plausible.io' && url.pathname.startsWith('/js/')) return route.fulfill({ contentType: 'application/javascript', body: '' });
      if (url.protocol === 'blob:') return route.continue();
      if (url.origin === 'http://127.0.0.1:4183' && (url.pathname.startsWith('/next/t/') || /^\/next\/(?:index\.html|sw\.js|(?:app|chunk|worker)\.[\w-]+\.js|styles\.[\w-]+\.css|manifest\.[\w-]+\.json)?$/.test(url.pathname))) return route.continue();
      unexpected.push(url.href); return route.abort();
    });
    await use({ errors, unexpected, requests, allowed });
    expect(errors).toEqual([]); expect(unexpected).toEqual([]);
  }, { auto: true }]
});
export { expect };
export const view = './?date=1960-01-01&lat=32.7767&lng=-96.7970&zoom=10';
export async function ready(page, url = view) {
  await page.goto(url); await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('body')).toHaveClass(/painted/);
  await expect(page.locator('#fatalError')).toBeHidden();
}
