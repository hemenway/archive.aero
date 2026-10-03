import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
  testDir: './tests', fullyParallel: true, workers: 2, timeout: 30000,
  reporter: [['list']], outputDir: '/tmp/archive-renderer-test-results',
  use: { baseURL: 'http://127.0.0.1:4181', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], launchOptions: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] } } },
    { name: 'webkit-mobile', use: { ...devices['iPhone 13'] } },
  ],
  webServer: { command: 'node server.mjs', url: 'http://127.0.0.1:4181', reuseExistingServer: false },
});
