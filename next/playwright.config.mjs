import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
  testDir: './tests', fullyParallel: true, timeout: 25000,
  expect: { timeout: 8000 }, forbidOnly: !!process.env.CI, retries: 0, workers: 3,
  reporter: 'list', outputDir: './test-results',
  use: { baseURL: 'http://127.0.0.1:4183/next/', serviceWorkers: 'block', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'webkit-mobile', use: { ...devices['iPhone 13'] } }
  ],
  webServer: { command: 'node server.mjs', url: 'http://127.0.0.1:4183/next/', reuseExistingServer: false, timeout: 10000 }
});
