import { defineConfig, devices } from '@playwright/test';

// Headless Linux uses WPE, whose page process crashes in this canvas-heavy suite.
// CI provides Xvfb so WebKit can use GTK while retaining the Safari device settings.
const linuxCi = !!process.env.CI && process.platform === 'linux';

export default defineConfig({
  // A cold WebKit on the CI runner can take longer than 8 s to paint the first
  // chart (the root suite allows 15 s for the same boot).
  testDir: './tests', fullyParallel: true, timeout: 40000,
  // Isolate browsers on CI: concurrent WebKit runs stalled even assertion polling.
  expect: { timeout: 15000 }, forbidOnly: !!process.env.CI, retries: 0, workers: process.env.CI ? 1 : 3,
  reporter: 'list', outputDir: './test-results',
  use: { baseURL: 'http://127.0.0.1:4183/next/', serviceWorkers: 'block', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'], headless: !linuxCi } },
    { name: 'webkit-mobile', use: { ...devices['iPhone 13'], headless: !linuxCi } }
  ],
  webServer: { command: 'node server.mjs', url: 'http://127.0.0.1:4183/next/', reuseExistingServer: false, timeout: 10000 }
});
