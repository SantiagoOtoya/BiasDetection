// Options page: view/edit the backend URL and test connectivity via /health.

const input = document.getElementById("backend-url");
const saveButton = document.getElementById("save-button");
const testButton = document.getElementById("test-button");
const status = document.getElementById("status");

function setStatus(message, ok = true) {
  status.textContent = message;
  status.style.color = ok ? "var(--level-low)" : "var(--danger)";
}

async function load() {
  const settings = await BiasConfig.getSettings();
  input.value = settings.backendUrl;
}

async function save() {
  const backendUrl = BiasConfig.normalizeUrl(input.value) || BiasConfig.DEFAULT_BACKEND_URL;
  input.value = backendUrl;
  await BiasConfig.saveSettings({ backendUrl });
  setStatus("Saved.");
}

async function testConnection() {
  const backendUrl = BiasConfig.normalizeUrl(input.value) || BiasConfig.DEFAULT_BACKEND_URL;
  setStatus("Testing…", true);
  try {
    const res = await fetch(backendUrl + "/health", { method: "GET" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json().catch(() => ({}));
    const mode = data.mode ? " (" + data.mode + " mode)" : "";
    setStatus("Connected" + mode + ".");
  } catch (err) {
    setStatus("Cannot reach backend: " + err.message, false);
  }
}

saveButton.addEventListener("click", save);
testButton.addEventListener("click", testConnection);
document.addEventListener("DOMContentLoaded", load);
