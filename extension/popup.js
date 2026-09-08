// Popup controller: manages UI states (idle / loading / error / report) and
// renders analysis results. Scraping + backend calls are wired in Phase 4;
// for now runAnalysis() drives the state machine with a placeholder.

const STATES = ["idle", "loading", "error", "report"];

const el = {};

function cacheElements() {
  el.states = {
    idle: document.getElementById("state-idle"),
    loading: document.getElementById("state-loading"),
    error: document.getElementById("state-error"),
    report: document.getElementById("state-report"),
  };
  el.runButton = document.getElementById("run-button");
  el.rerunButton = document.getElementById("rerun-button");
  el.retryButton = document.getElementById("retry-button");
  el.cancelButton = document.getElementById("cancel-button");
  el.optionsLink = document.getElementById("options-link");
  el.highlightButton = document.getElementById("highlight-button");
  el.loadingMessage = document.getElementById("loading-message");
  el.errorMessage = document.getElementById("error-message");
  el.backendNote = document.getElementById("backend-note");
  el.footerStatus = document.getElementById("footer-status");
  el.scoreValue = document.getElementById("score-value");
  el.scoreLevel = document.getElementById("score-level");
  el.scoreCaption = document.getElementById("score-caption");
  el.reportText = document.getElementById("report-text");
  el.factcheckList = document.getElementById("factcheck-list");
}

function showState(name) {
  STATES.forEach((state) => {
    el.states[state].classList.toggle("hidden", state !== name);
  });
  el.footerStatus.textContent = name.charAt(0).toUpperCase() + name.slice(1);
}

function setLoadingMessage(message) {
  el.loadingMessage.textContent = message;
}

function showError(message) {
  el.errorMessage.textContent = message || "Unexpected error.";
  showState("error");
}

// --- Rendering (reused by later phases) ---

function renderReport(result) {
  renderScore(result.overall_score, result.bias_level, result.score_caption);
  renderReportText(result.report);
  renderFactChecks(result.fact_checks || []);
  highlightOn = false;
  const hasSelections = (result.selected_sentences || []).length > 0;
  el.highlightButton.textContent = "Highlight on page";
  el.highlightButton.disabled = !hasSelections;
  showState("report");
  renderRelevanceDebug(result.meta || {});
}

// Surface relevance-filter counts in the footer + console so we can verify
// that irrelevant page furniture is being filtered (and article text is not).
function renderRelevanceDebug(meta) {
  const rel = meta.relevance;
  if (!rel) return;
  el.footerStatus.textContent =
    rel.sentences_after_relevance_filter +
    " relevant / " +
    rel.total_sentences +
    " scraped · " +
    rel.sentences_removed_as_irrelevant +
    " filtered";
  console.debug("[BiasDetection] relevance filter:", rel);
  console.debug("[BiasDetection] full analysis meta:", meta);
}

function renderScore(score, level, caption) {
  el.scoreValue.textContent = Number.isFinite(score) ? Math.round(score) : "—";
  const levelName = (level || "unknown").toLowerCase();
  el.scoreLevel.textContent = level || "Unknown";
  el.scoreLevel.className = "score-level level-" + levelName;
  el.scoreCaption.textContent = caption || "";
}

function renderReportText(text) {
  el.reportText.innerHTML = "";
  const clean = (text || "").trim();
  if (!clean) {
    const p = document.createElement("p");
    p.className = "empty-note";
    p.textContent = "No report was produced for this page.";
    el.reportText.appendChild(p);
    return;
  }
  clean.split(/\n\s*\n/).forEach((para) => {
    const p = document.createElement("p");
    p.textContent = para.trim();
    el.reportText.appendChild(p);
  });
}

function renderFactChecks(items) {
  el.factcheckList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "empty-note";
    li.textContent = "No checkable claims were identified.";
    el.factcheckList.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const status = (item.status || "unverified").toLowerCase();
    const li = document.createElement("li");
    li.className = "factcheck-item status-border-" + status;

    const badge = document.createElement("span");
    badge.className = "status-badge status-" + status;
    badge.textContent = status;
    li.appendChild(badge);

    const claim = document.createElement("span");
    claim.className = "claim";
    claim.textContent = item.claim || "";
    li.appendChild(claim);

    if (item.rationale) {
      const rationale = document.createElement("span");
      rationale.className = "rationale";
      rationale.textContent = item.rationale;
      li.appendChild(rationale);
    }
    el.factcheckList.appendChild(li);
  });
}

// --- Run flow ---

let lastResult = null;

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab.");
  if (/^(chrome|edge|about|chrome-extension|view-source):/i.test(tab.url || "")) {
    throw new Error("This page can't be analyzed. Open a news article or web page and try again.");
  }
  return tab;
}

// Send a message to the content script, injecting it first if it isn't loaded
// (content scripts don't auto-attach to pages opened before the extension).
async function messageTab(tabId, message) {
  try {
    return await chrome.tabs.sendMessage(tabId, message);
  } catch (err) {
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    return await chrome.tabs.sendMessage(tabId, message);
  }
}

async function scrapePage(tabId) {
  const result = await messageTab(tabId, { type: "SCRAPE" });
  if (!result || !result.ok || !result.text) {
    // Fail closed: do not POST anything to /analyze when extraction fails.
    if (result && result.reason === "insufficient_prose") {
      throw new Error(
        "Couldn't find enough article text on this page (mostly menus, popups, " +
          "or UI). Open a full article and try again."
      );
    }
    throw new Error("Couldn't find readable article text on this page.");
  }
  return result;
}

async function requestAnalysis(backendUrl, payload) {
  let res;
  try {
    res = await fetch(backendUrl + "/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    throw new Error(
      "Can't reach the backend at " + backendUrl + ". Start the server or update the URL in settings."
    );
  }
  if (!res.ok) {
    throw new Error("Backend error (HTTP " + res.status + ").");
  }
  return res.json();
}

async function runAnalysis() {
  try {
    showState("loading");
    setLoadingMessage("Reading the page…");
    const settings = await BiasConfig.getSettings();
    const tab = await getActiveTab();

    const scraped = await scrapePage(tab.id);

    setLoadingMessage("Analyzing for bias…");
    const result = await requestAnalysis(settings.backendUrl, {
      text: scraped.text,
      title: scraped.title,
      url: scraped.url,
      lead_text: scraped.leadText || "",
    });

    lastResult = result;
    renderReport(result);
  } catch (err) {
    showError(err && err.message ? err.message : String(err));
  }
}

async function refreshBackendNote() {
  const settings = await BiasConfig.getSettings();
  el.backendNote.textContent = "Backend: " + settings.backendUrl;
}

function wireEvents() {
  el.runButton.addEventListener("click", runAnalysis);
  el.rerunButton.addEventListener("click", runAnalysis);
  el.retryButton.addEventListener("click", runAnalysis);
  el.cancelButton.addEventListener("click", () => showState("idle"));
  el.optionsLink.addEventListener("click", () => chrome.runtime.openOptionsPage());
  el.highlightButton.addEventListener("click", toggleHighlight);
}

let highlightOn = false;

async function toggleHighlight() {
  try {
    const sentences = (lastResult?.selected_sentences || []).map((s) => s.text);
    if (!sentences.length) {
      el.highlightButton.textContent = "Nothing to highlight";
      return;
    }
    const tab = await getActiveTab();
    if (highlightOn) {
      await messageTab(tab.id, { type: "CLEAR_HIGHLIGHT" });
      highlightOn = false;
      el.highlightButton.textContent = "Highlight on page";
    } else {
      const res = await messageTab(tab.id, { type: "HIGHLIGHT", sentences });
      highlightOn = true;
      const n = res && typeof res.highlighted === "number" ? res.highlighted : sentences.length;
      el.highlightButton.textContent = "Clear highlights (" + n + ")";
    }
  } catch (err) {
    el.highlightButton.textContent = "Highlight unavailable";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  wireEvents();
  refreshBackendNote();
  showState("idle");
});
