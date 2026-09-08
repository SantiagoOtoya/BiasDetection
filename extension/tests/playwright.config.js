// Isolated Playwright config for the content.js extractor tests.
// Real Chromium is required because the Phase 1 hardening depends on live
// layout/style/geometry (getComputedStyle, getBoundingClientRect).
const { defineConfig, devices } = require("@playwright/test");

module.exports = defineConfig({
  testDir: ".",
  fullyParallel: true,
  reporter: "list",
  use: {
    viewport: { width: 1280, height: 800 },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
