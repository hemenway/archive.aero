import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { restoreFacilities } from '../../worker-atc/src/facilities.js';

const fixture = readFileSync(new URL('../fixtures/atc-facilities.html', import.meta.url));
test.use({ javaScriptEnabled: false });

test.beforeEach(async ({ page }) => {
  await page.route('**/atc/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/atc/facility-photos') {
      await route.fulfill({ contentType: 'text/html', body: Buffer.from(restoreFacilities(fixture, url)) });
    } else if (url.pathname === '/atc/anniston-fss-1969') {
      await route.fulfill({ contentType: 'text/html', body: '<h1>Anniston FSS, 1969</h1>' });
    } else await route.fulfill({ status: 404 });
  });
});

test('state, city and photo navigation work without JavaScript', async ({ page }) => {
  await page.goto('/atc/facility-photos');
  await page.locator('summary').filter({ hasText: /^Alabama$/ }).click();
  await page.getByRole('link', { name: 'Anniston', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Anniston, Alabama' })).toBeVisible();
  const result = page.locator('.facility-results a');
  await expect(result).toHaveCount(1);
  await expect(result).toHaveAttribute('href', '/atc/anniston-fss-1969');
  await result.click();
  await expect(page.getByRole('heading', { name: 'Anniston FSS, 1969' })).toBeVisible();
});

test('multiword states expand from the keyboard and city filters survive reload', async ({ page }) => {
  await page.goto('/atc/facility-photos');
  await page.locator('summary').filter({ hasText: /^New York$/ }).press('Enter');
  await expect(page.getByRole('link', { name: 'Albany', exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Albany', exact: true }).click();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Albany, New York' })).toBeVisible();
  await expect(page.locator('.facility-results a')).not.toHaveCount(0);
  await expect(page.locator('[aria-current="page"]')).toHaveText('Albany');
});
