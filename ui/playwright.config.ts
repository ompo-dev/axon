import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: 'workspace.spec.ts',
  timeout: 90000,
  expect: { timeout: 15000 },
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:4329',
    channel: 'msedge',
    headless: true,
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure',
  },
  webServer: {
    env: { AXON_UI_EPHEMERAL: '1' },
    command: 'bun server.mjs 4329',
    url: 'http://127.0.0.1:4329/api/state',
    timeout: 15000,
    reuseExistingServer: !process.env.CI,
  },
});
