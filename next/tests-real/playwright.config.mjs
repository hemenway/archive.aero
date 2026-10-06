import { defineConfig, devices } from '@playwright/test';
// Real renderer + real data plane + the actual Worker code. Needs WebGL2, so it
// is run on demand (npm run test:next:real), not in the default CI chain.
export default defineConfig({
  testDir: '.', testMatch: '*.spec.mjs', fullyParallel: false, workers: 1, timeout: 30000,
  expect: { timeout: 10000 }, forbidOnly: !!process.env.CI, retries: 0,
  reporter: 'list', outputDir: './test-results',
  use: { baseURL: 'http://127.0.0.1:4184/next/', serviceWorkers: 'block', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], launchOptions: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] } } },
    { name: 'webkit-mobile', use: { ...devices['iPhone 13'] } },
  ],
  webServer: { command: 'node server.mjs', url: 'http://127.0.0.1:4184/next/', reuseExistingServer: false, timeout: 30000 },
});
