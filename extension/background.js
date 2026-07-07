// Service worker (MV3). Currently minimal: the popup talks to the backend
// directly via fetch and to the page via chrome.scripting. This worker is a
// placeholder for future message routing / lifecycle hooks.

chrome.runtime.onInstalled.addListener(() => {
  // Reserved for first-run setup (e.g. seeding default settings).
});
