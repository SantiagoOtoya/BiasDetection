// Playwright DOM tests for extension/content.js article extraction (Phase 1).
//
// Each test loads a real HTML fixture into Chromium, injects content.js (with a
// minimal `chrome` stub so its message listener doesn't throw), calls the global
// scrapeArticle(), and asserts:
//   - the unwanted site-UI marker is ABSENT from extracted text, and
//   - the legitimate article marker REMAINS present.
// The UI-only fixture asserts { ok:false, reason:"insufficient_prose" }.

const path = require("path");
const { test, expect } = require("@playwright/test");

const CONTENT_JS = path.resolve(__dirname, "..", "content.js");
const ART = "REALARTICLEBODY7781";

// Three substantial sentences (>250 chars, >=2 sentences) of genuine prose.
const ARTICLE_BODY = `
  <h1>Global Markets React to New Policy</h1>
  <p>${ART} The central bank announced a significant shift in monetary policy on Tuesday, surprising analysts who had expected rates to stay unchanged this quarter.</p>
  <p>Economists said the decision reflects mounting concern about persistent inflation, and several noted that consumer prices have climbed steadily across most sectors this year.</p>
  <p>Officials declined to say whether further adjustments would follow, though they emphasized a careful, data-driven approach in the months ahead.</p>
`;

function pageHtml(body) {
  return `<!doctype html><html><head><meta charset="utf-8"><title>Test Article</title></head><body>${body}</body></html>`;
}

async function scrape(page, body) {
  await page.addInitScript(() => {
    window.chrome = { runtime: { onMessage: { addListener() {} } } };
  });
  await page.setContent(pageHtml(body), { waitUntil: "load" });
  await page.addScriptTag({ path: CONTENT_JS });
  return page.evaluate(() => scrapeArticle());
}

test.describe("article extraction hardening", () => {
  test("normal article with <article> tag", async ({ page }) => {
    const r = await scrape(page, `<article>${ARTICLE_BODY}</article>`);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
  });

  test("newsletter modal over article (role=dialog/aria-modal)", async ({ page }) => {
    const body = `
      <div class="page">
        <div role="dialog" aria-modal="true" class="newsletter" style="position:fixed;inset:0;z-index:9999;background:#fff;">
          <p>NEWSLETTER_UI_MARKER Subscribe to our newsletter to get the latest headlines delivered to your inbox every single morning.</p>
        </div>
        ${ARTICLE_BODY}
      </div>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("NEWSLETTER_UI_MARKER");
  });

  test("cookie consent banner (fixed + high z-index + button)", async ({ page }) => {
    const body = `
      <div class="page">
        <div class="c-9f31" style="position:fixed;bottom:0;left:0;width:100%;height:120px;z-index:5000;background:#eee;">
          <p>COOKIE_UI_MARKER We use cookies to improve your experience. Accept all cookies to continue browsing this website.</p>
          <button>Accept all</button>
        </div>
        ${ARTICLE_BODY}
      </div>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("COOKIE_UI_MARKER");
  });

  test("subscription modal (.subscribe)", async ({ page }) => {
    const body = `
      <div class="page">
        <div class="subscribe" style="position:fixed;inset:0;z-index:8000;background:#fff;">
          <p>SUBSCRIBE_UI_MARKER Subscribe now for unlimited access to our award-winning journalism and exclusive investigative reporting.</p>
        </div>
        ${ARTICLE_BODY}
      </div>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("SUBSCRIBE_UI_MARKER");
  });

  test("login modal (role=dialog + .login)", async ({ page }) => {
    const body = `
      <div class="page">
        <div role="dialog" class="login" style="position:fixed;inset:0;z-index:9000;background:#fff;">
          <p>LOGIN_UI_MARKER Log in to continue reading. Sign in with your account to access this member-only content today.</p>
          <form><input type="email"><button>Sign in</button></form>
        </div>
        ${ARTICLE_BODY}
      </div>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("LOGIN_UI_MARKER");
  });

  test("hashed/random-class overlay detected via layout only", async ({ page }) => {
    // No role/aria/keyword class — only fixed + full-viewport + high z-index + form.
    const body = `
      <div class="page">
        <div class="x7f2a9q" style="position:fixed;inset:0;z-index:9999;background:#fff;">
          <p>HASHED_UI_MARKER Enter your email to keep reading this article and receive our daily briefing every morning.</p>
          <form><input type="email"><button>Continue</button></form>
        </div>
        ${ARTICLE_BODY}
      </div>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("HASHED_UI_MARKER");
  });

  test("hidden DOM text (display:none and aria-hidden)", async ({ page }) => {
    const body = `
      <div class="page">
        <div style="display:none;"><p>HIDDEN_DISPLAY_MARKER This paragraph is fully hidden with display none and must never be scraped.</p></div>
        <div aria-hidden="true"><p>HIDDEN_ARIA_MARKER This paragraph is aria-hidden and should be treated as invisible to extraction.</p></div>
        ${ARTICLE_BODY}
      </div>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("HIDDEN_DISPLAY_MARKER");
    expect(r.text).not.toContain("HIDDEN_ARIA_MARKER");
  });

  test("article without an <article> tag (div soup)", async ({ page }) => {
    const body = `
      <header><nav><a href="#">Home</a> <a href="#">World</a></nav></header>
      <div class="post-body">${ARTICLE_BODY}</div>
      <footer><p>FOOTER_UI_MARKER Copyright 2026. All rights reserved. Terms of service and privacy policy.</p></footer>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    expect(r.text).not.toContain("FOOTER_UI_MARKER");
  });

  test("article legitimately discussing subscriptions and cookies is preserved", async ({ page }) => {
    const body = `
      <article>
        <h1>How Publishers Use Cookies and Subscriptions</h1>
        <p>${ART} This report examines how modern news organizations rely on subscription revenue after years of declining advertising income across the industry.</p>
        <p>COOKIE_TOPIC_MARKER Many publishers also deploy cookies to track reader engagement, a practice that regulators in several countries have begun to scrutinize closely.</p>
        <p>Analysts argue that the shift toward subscriptions has changed how newsrooms think about which stories to pursue and how to measure their success.</p>
      </article>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(true);
    expect(r.text).toContain(ART);
    // The genuine sentence mentioning cookies/subscriptions must survive.
    expect(r.text).toContain("COOKIE_TOPIC_MARKER");
  });

  test("UI-only page returns insufficient_prose and no text", async ({ page }) => {
    const body = `
      <header><nav><a href="#">Home</a> <a href="#">News</a> <a href="#">Sports</a></nav></header>
      <div class="cta"><button>Subscribe</button> <button>Log in</button></div>
      <div class="c-abc" style="position:fixed;bottom:0;width:100%;height:100px;z-index:5000;">
        <p>ONLY_UI_MARKER Accept all cookies to continue.</p><button>Accept</button>
      </div>
      <footer><a href="#">Terms</a> <a href="#">Privacy</a></footer>`;
    const r = await scrape(page, body);
    expect(r.ok).toBe(false);
    expect(r.reason).toBe("insufficient_prose");
    expect(r.text).toBeUndefined();
  });

  test("benchmark: extraction stays fast with many blocks + overlays", async ({ page }) => {
    const paras = Array.from(
      { length: 300 },
      (_, i) =>
        `<p>Paragraph ${i} discusses the ongoing economic situation in detail, noting that many factors continue to influence the broader market outlook this year.</p>`
    ).join("");
    const overlays = Array.from(
      { length: 5 },
      (_, i) =>
        `<div role="dialog" aria-modal="true" style="position:fixed;inset:0;z-index:${9000 + i};"><p>OVERLAY_${i} Subscribe now.</p><form><input><button>Go</button></form></div>`
    ).join("");
    const body = `<div class="page">${overlays}<div class="post-body"><h1>Big Article ${ART}</h1>${paras}</div></div>`;

    await page.addInitScript(() => {
      window.chrome = { runtime: { onMessage: { addListener() {} } } };
    });
    await page.setContent(pageHtml(body), { waitUntil: "load" });
    await page.addScriptTag({ path: CONTENT_JS });
    const ms = await page.evaluate(() => {
      const t0 = performance.now();
      const r = scrapeArticle();
      const t1 = performance.now();
      return { ms: t1 - t0, ok: r.ok, len: (r.text || "").length };
    });
    console.log(
      `benchmark: scrapeArticle over ~300 blocks + 5 overlays = ${ms.ms.toFixed(1)} ms (ok=${ms.ok}, chars=${ms.len})`
    );
    expect(ms.ok).toBe(true);
    expect(ms.ms).toBeLessThan(500);
  });
});
