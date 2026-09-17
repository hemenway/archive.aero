import { test, expect } from '@playwright/test';
import { installFixtures } from './fixtures.mjs';

const view = '/?date=1960-01-01&lat=32.7767&lng=-96.7970&zoom=10';
const inventory = {
  generated: '2026-09-16',
  locations: {
    'Test Dallas': { era: 'modern', ref: 'dallas', gaps: [], charts: [
      { d: '1965-01-01', e: '1965-07-01', ed: '1', f: '' },
      { d: '1966-01-01', e: '1966-07-01', ed: '3', f: 'L' },
    ] },
  },
  groups: [{ name: 'Dallas', old: [], modern: ['Test Dallas'] }],
  rings: { dallas: [[[-98, 31], [-95, 31], [-95, 34], [-98, 34], [-98, 31]]] },
};

async function servePage(page, name) {
  await page.route(`**/${name}`, route => route.fulfill({
    path: new URL(`../../${name}.html`, import.meta.url).pathname, contentType: 'text/html',
  }));
}

test('viewer keyboard ownership, date bounds, modal trapping and focus restoration', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await page.goto(view);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  const timeline = page.getByRole('slider', { name: 'Historical chart date' });
  await timeline.focus();
  await page.keyboard.press('Home');
  await expect(timeline).toHaveAttribute('aria-valuenow', '0');
  await expect(timeline).toHaveAttribute('aria-valuetext', /January 1, 1950/);
  await page.keyboard.press('ArrowLeft');
  await expect(timeline).toHaveAttribute('aria-valuenow', '0');
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('#timeSelect')).toHaveValue('1960-01-01');
  await page.keyboard.press('End');
  await expect(timeline).toHaveAttribute('aria-valuenow', await timeline.getAttribute('aria-valuemax'));
  await page.keyboard.press('ArrowRight');
  await expect(timeline).toHaveAttribute('aria-valuenow', await timeline.getAttribute('aria-valuemax'));

  const before = await page.locator('#timeSelect').inputValue();
  await page.locator('#map').focus();
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#timeSelect')).toHaveValue(before);
  await timeline.focus();
  await page.keyboard.press('?');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.locator('#main-content')).toHaveAttribute('inert', '');
  await expect(page.locator('#closeShortcuts')).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.locator('#closeShortcuts')).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(page.locator('#closeShortcuts')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toBeHidden();
  await expect(timeline).toBeFocused();
  await expect(page.locator('#main-content')).not.toHaveAttribute('inert');

  await page.locator('#toolsBtn').click();
  const opacity = page.getByRole('slider', { name: 'Chart opacity' });
  await opacity.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(opacity).toHaveValue('99');
  await expect(page.locator('#timeSelect')).toHaveValue(before);
  await page.keyboard.press('Escape');
  await expect(page.locator('#toolsBtn')).toBeFocused();
  await expect(page.locator('#toolsBtn')).toHaveAttribute('aria-expanded', 'false');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('keyboard users can inspect map center and browse canvas airfields', async ({ page }) => {
  const diagnostics = await installFixtures(page);
  await page.route('**/airfields.json', route => route.fulfill({ json: {
    type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Point', coordinates: [-96.797, 32.7767] },
      properties: { name: 'Test Field', start_year: 1930, status: 'open', state: 'TX' } }],
  } }));
  await page.goto(view);
  await expect(page.locator('#loadingSplash')).toBeHidden();
  await page.locator('#map').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#pinPanel')).toBeFocused();
  await expect(page.locator('#pinLoc')).toHaveText('Test Dallas');
  await page.locator('#helpBtn').click();
  await page.keyboard.press('Escape');
  await expect(page.locator('#pinPanel')).toBeVisible();
  await page.locator('#pinClose').focus();
  await page.keyboard.press('Escape');
  await expect(page.locator('#map')).toBeFocused();
  await expect(page.locator('#pinPanel')).toBeHidden();

  await page.locator('#toolsBtn').click();
  await page.locator('#afBrowser summary').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#afSelect')).toBeEnabled();
  await expect(page.locator('#afCount')).toHaveText('1 airfields in view.');
  await page.locator('#afOpen').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#afPanel')).toBeFocused();
  await expect(page.locator('#afName')).toHaveText('Test Field');
  await page.keyboard.press('Escape');
  await expect(page.locator('#afPanel')).toBeHidden();
  await expect(page.locator('#toolsBtn')).toBeFocused();
  await page.locator('#toolsBtn').click();
  await page.locator('#airfieldsBtn').click();
  await expect(page.locator('#afSelect')).toBeDisabled();
  await expect(page.locator('#afCount')).toContainText('Turn on Airfields');
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

async function contributionPage(page) {
  const diagnostics = await installFixtures(page);
  await servePage(page, 'contribute');
  await page.route('**/timeline_data.json', route => route.fulfill({ json: inventory }));
  await page.goto('/contribute');
  await expect(page.locator('#shelfStats')).toContainText('1 charts');
  return diagnostics;
}

test('atlas native date and region controls expose coverage and open the chosen scan', async ({ page }) => {
  const diagnostics = await contributionPage(page);
  const slider = page.getByRole('slider', { name: 'Atlas date' });
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(page.locator('#atDate')).toHaveValue('1928-01-01');
  await expect(page.locator('#atStepB')).toBeDisabled();
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#atDate')).toHaveValue('1928-01-01');
  await page.keyboard.press('End');
  await expect(page.locator('#atDate')).toHaveValue('2026-12-31');
  await expect(page.locator('#atStepF')).toBeDisabled();
  await page.locator('#atDate').fill('1965-09-01');
  await page.locator('#atDate').dispatchEvent('change');
  await expect(page.locator('#atRegionInfo')).toContainText('missing');
  await expect(page.locator('#atOpen')).toBeDisabled();
  await page.locator('#atDate').fill('1965-02-01');
  await page.locator('#atDate').dispatchEvent('change');
  await expect(page.locator('#atOpen')).toBeEnabled();
  await expect(slider).toHaveAttribute('aria-valuetext', 'February 1, 1965');
  await page.locator('#atPlay').click();
  await expect(page.locator('#atPlay')).toHaveAccessibleName('Pause timeline');
  await expect(page.locator('#atRegionInfo')).toHaveAttribute('aria-live', 'off');
  await page.locator('#atRegion').focus();
  await expect(page.locator('#atPlay')).toHaveAccessibleName('Play timeline');
  await page.locator('#atDate').fill('1965-02-01');
  await page.locator('#atDate').dispatchEvent('change');
  await page.locator('#atOpen').focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/date=1965-01-01&lat=32.5000&lng=-96.5000&zoom=7/);
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('inventory offers one tab stop per lane, edition labels, year search and collapse semantics', async ({ page }) => {
  const diagnostics = await contributionPage(page);
  const heading = page.locator('.grp-header').first();
  await heading.focus();
  await page.keyboard.press('Enter');
  await expect(heading).toHaveAttribute('aria-expanded', 'true');
  const list = page.getByRole('listbox', { name: /Test Dallas/ });
  await page.keyboard.press('Tab');
  await expect(list).toBeFocused();
  const options = list.getByRole('option');
  await expect(options).toHaveCount(3);
  await expect(options.first()).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('ArrowRight');
  await expect(options.nth(1)).toHaveAttribute('aria-selected', 'true');
  await expect(list).toHaveAttribute('aria-activedescendant', await options.nth(1).getAttribute('id'));
  await expect(options.nth(1)).toHaveAccessibleName(/estimated date.*missing from archive/);
  await expect(page.locator('#editionDetail')).toContainText('missing from archive');
  await page.keyboard.press('End');
  await expect(options.last()).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Home');
  await page.keyboard.type('1966');
  await expect(options.last()).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Escape');
  await expect(page.locator('#tip')).toBeHidden();
  await page.locator('#collapseAll').click();
  await expect(list).toBeHidden();
  await expect(heading).toHaveAttribute('aria-expanded', 'false');
  await expect(page.locator('.grp-body')).toHaveAttribute('hidden', '');
  await page.locator('#shelfSearch').fill('nonexistent');
  await expect(page.locator('#shelfStats')).toContainText('0 charts');
  await expect(page.locator('#shelfSearch')).toBeFocused();
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});

test('editorial pages have working skip links and a single main landmark', async ({ page, browserName }) => {
  const diagnostics = await installFixtures(page);
  for (const name of ['about', 'sources', 'contribute']) {
    await servePage(page, name);
    await page.route('**/timeline_data.json', route => route.fulfill({ json: inventory }));
    await page.goto('/' + name);
    await expect(page.getByRole('main')).toHaveCount(1);
    // WebKit follows macOS's links-skipped-by-Tab preference; Option+Tab
    // includes links without changing the site's native anchor semantics.
    await page.keyboard.press(browserName === 'webkit' ? 'Alt+Tab' : 'Tab');
    await expect(page.locator('.skip-link')).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('main')).toBeFocused();
  }
  expect(diagnostics).toEqual({ errors: [], unexpected: [] });
});
