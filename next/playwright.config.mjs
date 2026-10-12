import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
  // A cold WebKit on the CI runner can take longer than 8 s to paint the first
  // chart (the root suite allows 15 s for the same boot).
  testDir: './tests', fullyParallel: true, timeout: 40000,
  expect: { timeout: 15000 }, forbidOnly: !!process.env.CI, retries: 0, workers: 3,
  reporter: 'list', outputDir: './test-results',
  use: { baseURL: 'http://127.0.0.1:4183/next/', serviceWorkers: 'block', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'webkit-mobile', use: { ...devices['iPhone 13'] } }
  ],
  webServer: { command: 'node server.mjs', url: 'http://127.0.0.1:4183/next/', reuseExistingServer: false, timeout: 10000 }
});
