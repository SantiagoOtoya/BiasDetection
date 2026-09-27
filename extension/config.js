// Shared configuration + small helpers for the Bias Detection extension.
// Loaded before popup.js / options.js via <script> so BiasConfig is a global.

const BiasConfig = {
  // Default backend URL. Override in the options page (stored in chrome.storage).
  DEFAULT_BACKEND_URL: "http://localhost:8000",

  STORAGE_KEY: "biasDetectionSettings",

  // Resolve the effective settings, merging stored overrides over defaults.
  async getSettings() {
    const defaults = { backendUrl: this.DEFAULT_BACKEND_URL };
    try {
      const stored = await chrome.storage.sync.get(this.STORAGE_KEY);
      const saved = stored?.[this.STORAGE_KEY] || {};
      return { ...defaults, ...saved };
    } catch (err) {
      return defaults;
    }
  },

  async saveSettings(settings) {
    await chrome.storage.sync.set({ [this.STORAGE_KEY]: settings });
  },

  // Normalize a user-entered URL: trim, strip trailing slash.
  normalizeUrl(url) {
    return (url || "").trim().replace(/\/+$/, "");
  },
};

// Make available in both module and classic contexts.
if (typeof window !== "undefined") {
  window.BiasConfig = BiasConfig;
}
