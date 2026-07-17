// Content script: extracts the main article text from the page using a
// Readability-lite heuristic, and responds to messages from the popup.
//
// Messages handled:
//   { type: "PING" }    -> { ok: true }
//   { type: "SCRAPE" }  -> { ok, title, text, url, sentenceCount }
//   { type: "HIGHLIGHT", sentences: [...] } -> handled in Phase 10 (no-op stub)

const UNLIKELY_SELECTORS = [
  "nav", "header", "footer", "aside", "form", "button", "iframe", "dialog",
  "[role='navigation']", "[role='banner']", "[role='complementary']",
  "[role='dialog']", "[role='alertdialog']", "[aria-hidden='true']",
  ".nav", ".menu", ".sidebar", ".advert", ".ad", ".ads", ".promo",
  ".newsletter", ".subscribe", ".signup", ".social", ".share", ".related",
  ".recommended", ".trending", ".popular", ".comments", ".comment",
  ".footer", ".header", ".breadcrumb", ".cookie", ".modal", ".paywall",
  ".recirc", ".sponsored", ".author-bio", ".bio", ".tags", ".taglist",
  "figure figcaption",
];

// Token-aware class/id keyword filter: matches "ad" in "ad-slot" or "top_ad"
// but NOT inside words like "read", "header", or "shadow".
const BOILERPLATE_TOKEN_RE = new RegExp(
  "(^|[-_\\s])(" +
    [
      "ad", "ads", "advertisement", "advert", "sponsored", "sponsor", "promo",
      "newsletter", "subscribe", "signup", "related", "recommended",
      "recommendations", "trending", "popular", "comment", "comments",
      "footer", "sidebar", "share", "social", "cookie", "consent", "modal",
      "paywall", "outbrain", "taboola", "recirc", "breadcrumb", "masthead",
    ].join("|") +
    ")([-_\\s]|$)",
  "i"
);

const POSITIVE_HINTS = /(article|content|post|story|entry|main|body|text|prose)/i;
const NEGATIVE_HINTS = /(comment|meta|footer|footnote|nav|sidebar|sponsor|ad-|advert|promo|share|social|related|recirc|widget|caption|breadcrumb)/i;

function isBoilerplateElement(node) {
  const idClass = (node.id || "") + " " + (typeof node.className === "string" ? node.className : "");
  return BOILERPLATE_TOKEN_RE.test(idClass);
}

function isVisible(node) {
  const style = window.getComputedStyle(node);
  if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
    return false;
  }
  const rect = node.getBoundingClientRect();
  return rect.width > 0 || rect.height > 0 || node.getClientRects().length > 0;
}

function linkDensity(node) {
  const textLen = (node.textContent || "").trim().length;
  if (!textLen) return 1;
  let linkLen = 0;
  node.querySelectorAll("a").forEach((a) => {
    linkLen += (a.textContent || "").length;
  });
  return linkLen / textLen;
}

// Score a candidate container by the amount of genuine paragraph text it holds.
function scoreCandidate(node) {
  const paragraphs = node.querySelectorAll("p, li");
  let score = 0;
  paragraphs.forEach((p) => {
    const text = (p.textContent || "").trim();
    if (text.length < 25) return;
    score += 1; // base point for a real paragraph
    score += Math.min(3, Math.floor(text.length / 100)); // reward length
    score += (text.match(/,/g) || []).length * 0.25; // commas ~ prose
  });

  const id = (node.id || "") + " " + (node.className || "");
  if (POSITIVE_HINTS.test(id)) score += 5;
  if (NEGATIVE_HINTS.test(id)) score -= 5;
  if (node.tagName === "ARTICLE") score += 8;
  if (node.getAttribute && node.getAttribute("role") === "main") score += 6;

  score *= Math.max(0, 1 - linkDensity(node)); // penalize link-heavy blocks
  return score;
}

function collectCandidates() {
  const selectors = ["article", "main", "[role='main']", "section", "div"];
  const set = new Set();
  selectors.forEach((sel) => {
    document.querySelectorAll(sel).forEach((node) => set.add(node));
  });
  return Array.from(set).filter((node) => {
    if (!isVisible(node)) return false;
    const text = (node.textContent || "").trim();
    return text.length >= 200; // ignore tiny containers
  });
}

// Extract clean paragraph text (and the lead paragraph) from the chosen container.
function extractContent(container) {
  const clone = container.cloneNode(true);
  clone.querySelectorAll(UNLIKELY_SELECTORS.join(",")).forEach((n) => n.remove());
  clone.querySelectorAll("script, style, noscript, svg, template").forEach((n) => n.remove());
  // Aggressive pass: remove anything whose class/id matches boilerplate keywords.
  clone.querySelectorAll("*").forEach((n) => {
    if (isBoilerplateElement(n)) n.remove();
  });

  const blocks = [];
  let leadParagraph = "";
  clone.querySelectorAll("p, li, blockquote, h2, h3").forEach((node) => {
    const text = (node.textContent || "").replace(/\s+/g, " ").trim();
    if (text.length < 25) return;
    // Skip link/button-heavy blocks (menus, tag lists, "read more" clusters).
    if (linkDensity(node) > 0.5) return;
    if (node.querySelectorAll("button, input, select").length > 0) return;
    blocks.push(text);
    if (!leadParagraph && node.tagName === "P" && text.length >= 60) {
      leadParagraph = text;
    }
  });

  // Fallback: if the structured pass found little, use the raw text.
  if (blocks.join(" ").length < 200) {
    return {
      text: (clone.textContent || "").replace(/\s+/g, " ").trim(),
      leadParagraph,
    };
  }
  return { text: blocks.join("\n\n"), leadParagraph };
}

function getArticleTitle() {
  const h1 = document.querySelector("article h1, main h1, h1");
  if (h1 && (h1.textContent || "").trim()) return h1.textContent.trim();
  const og = document.querySelector("meta[property='og:title']");
  if (og && og.content) return og.content.trim();
  return (document.title || "").trim();
}

function countSentences(text) {
  const parts = text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'([])/);
  return parts.filter((s) => s.trim().length > 0).length;
}

function scrapeArticle() {
  const candidates = collectCandidates();
  let best = null;
  let bestScore = -Infinity;
  let bestArticle = null;
  let bestArticleScore = -Infinity;
  candidates.forEach((node) => {
    const score = scoreCandidate(node);
    if (score > bestScore) {
      bestScore = score;
      best = node;
    }
    if (node.tagName === "ARTICLE" && score > bestArticleScore) {
      bestArticleScore = score;
      bestArticle = node;
    }
  });

  // Prefer a real <article> element when it holds comparable content: sidebars
  // and recirculation grids sometimes out-score it on raw paragraph count.
  let container = best || document.body;
  if (
    bestArticle &&
    bestArticle !== best &&
    bestArticleScore >= bestScore * 0.5 &&
    (bestArticle.textContent || "").trim().length >= 500
  ) {
    container = bestArticle;
  }

  const title = getArticleTitle();
  let { text, leadParagraph } = extractContent(container);

  // Absolute fallback so we never return empty on a text-bearing page.
  if (!text || text.length < 120) {
    text = (document.body.textContent || "").replace(/\s+/g, " ").trim();
  }

  // Topic anchor for the backend relevance filter: headline + first real paragraph.
  const leadText = [title, leadParagraph].filter(Boolean).join(". ").slice(0, 600);

  return {
    ok: text.length > 0,
    title,
    text,
    leadText,
    url: location.href,
    sentenceCount: countSentences(text),
  };
}

// --- In-page highlighting (Phase 10) ---

const HIGHLIGHT_CLASS = "bias-detection-highlight";
const HIGHLIGHT_STYLE_ID = "bias-detection-style";
const BLOCK_SELECTOR = "p, li, blockquote, h1, h2, h3, h4, td, dd, figcaption";

function ensureHighlightStyle() {
  if (document.getElementById(HIGHLIGHT_STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = HIGHLIGHT_STYLE_ID;
  style.textContent =
    "." + HIGHLIGHT_CLASS + "{background:#ffe58a;color:#1a1a1a;" +
    "border-radius:2px;padding:0 1px;box-shadow:0 0 0 1px rgba(224,138,58,.4);}";
  (document.head || document.documentElement).appendChild(style);
}

function normalizeForMatch(text) {
  return text.replace(/\s+/g, " ").trim();
}

// Build a whitespace-flexible, case-insensitive regex for a target sentence.
function sentenceRegex(sentence) {
  const trimmed = normalizeForMatch(sentence).slice(0, 240);
  if (trimmed.length < 12) return null;
  const escaped = trimmed
    .replace(/[.*+?^${}()|[\]\\]/g, "\\$&") // escape regex specials
    .replace(/['‘’]/g, "['‘’]") // straight/curly apostrophes
    .replace(/["“”]/g, "[\"“”]") // straight/curly quotes
    .replace(/\s+/g, "\\s+"); // flexible whitespace
  try {
    return new RegExp(escaped, "i");
  } catch (err) {
    return null;
  }
}

// Wrap a slice [start,end) of a single text node in a highlight mark.
function wrapSlice(textNode, start, end) {
  const range = document.createRange();
  range.setStart(textNode, start);
  range.setEnd(textNode, end);
  const mark = document.createElement("mark");
  mark.className = HIGHLIGHT_CLASS;
  try {
    range.surroundContents(mark);
    return mark;
  } catch (err) {
    return null;
  }
}

// Highlight the first occurrence of `regex` inside a block element.
function highlightInBlock(block, regex) {
  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      const parent = node.parentElement;
      if (parent && parent.closest("." + HIGHLIGHT_CLASS)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const nodes = [];
  let raw = "";
  while (walker.nextNode()) {
    const node = walker.currentNode;
    nodes.push({ node, start: raw.length });
    raw += node.nodeValue;
  }
  nodes.forEach((n) => (n.end = n.start + n.node.nodeValue.length));

  const match = regex.exec(raw);
  if (!match) return null;

  const matchStart = match.index;
  const matchEnd = match.index + match[0].length;
  let firstMark = null;

  // Wrap each text node segment overlapping the match (process in DOM order;
  // splitting one text node does not shift the others' recorded offsets).
  for (const info of nodes) {
    const overlapStart = Math.max(matchStart, info.start);
    const overlapEnd = Math.min(matchEnd, info.end);
    if (overlapStart >= overlapEnd) continue;
    const mark = wrapSlice(info.node, overlapStart - info.start, overlapEnd - info.start);
    if (mark && !firstMark) firstMark = mark;
  }
  return firstMark;
}

function clearHighlights() {
  const marks = document.querySelectorAll("mark." + HIGHLIGHT_CLASS);
  marks.forEach((mark) => {
    const parent = mark.parentNode;
    if (!parent) return;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parent.normalize();
  });
  return marks.length;
}

function highlightSentences(sentences) {
  ensureHighlightStyle();
  clearHighlights();

  const blocks = Array.from(document.querySelectorAll(BLOCK_SELECTOR));
  const normalizedBlocks = blocks.map((b) => ({
    el: b,
    norm: normalizeForMatch(b.textContent || ""),
  }));

  let highlighted = 0;
  let firstMark = null;

  (sentences || []).forEach((sentence) => {
    const regex = sentenceRegex(sentence);
    if (!regex) return;
    const target = normalizeForMatch(sentence).slice(0, 240);
    const block = normalizedBlocks.find((b) => b.norm.includes(target.slice(0, 60)));
    if (!block) return;
    const mark = highlightInBlock(block.el, regex);
    if (mark) {
      highlighted += 1;
      if (!firstMark) firstMark = mark;
    }
  });

  if (firstMark) {
    firstMark.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  return highlighted;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "PING") {
    sendResponse({ ok: true });
    return true;
  }
  if (message?.type === "SCRAPE") {
    try {
      sendResponse(scrapeArticle());
    } catch (err) {
      sendResponse({ ok: false, error: String(err && err.message ? err.message : err) });
    }
    return true;
  }
  if (message?.type === "HIGHLIGHT") {
    try {
      const count = highlightSentences(message.sentences || []);
      sendResponse({ ok: true, highlighted: count });
    } catch (err) {
      sendResponse({ ok: false, error: String(err && err.message ? err.message : err) });
    }
    return true;
  }
  if (message?.type === "CLEAR_HIGHLIGHT") {
    try {
      const count = clearHighlights();
      sendResponse({ ok: true, cleared: count });
    } catch (err) {
      sendResponse({ ok: false, error: String(err && err.message ? err.message : err) });
    }
    return true;
  }
  return false;
});
