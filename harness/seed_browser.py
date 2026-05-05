"""Seed consent cookies and auto-dismiss scripts into the cocoa sandbox's
browser *before* cocoa starts navigating.

Why this exists: cocoa-agent's browser (Chromium inside the AIO sandbox) hits
consent walls on Google, YouTube, and any site with a cookiebot/onetrust
banner — which can block the task entirely or eat iterations on dismissal.
Probe (scripts/probe_cookie_seed.py) confirmed that cookies set via
Playwright-over-CDP land on the same Chromium that cocoa drives through
the sandbox HTTP API (the sandbox serves a single shared browser with one
context), so pre-seeding here is enough.

Called from harness/run_task.py right after wait_for_sandbox() returns.
"""
from __future__ import annotations


_CONSENT_COOKIES = [
    # Google — SOCS is the "Settings Of Consent" cookie, CONSENT is the
    # legacy accept flag. Setting both bypasses the EU/UK consent wall.
    {"name": "SOCS",    "value": "CAESHAgBEhJnd3NfMjAyMzExMTYtMF9SQzIaAmVuIAEaBgiAt4avBg", "domain": ".google.com",  "path": "/"},
    {"name": "CONSENT", "value": "YES+cb.20210418-17-p0.en+FX+667",                         "domain": ".google.com",  "path": "/"},
    # YouTube (separate eTLD).
    {"name": "SOCS",    "value": "CAESHAgBEhJnd3NfMjAyMzExMTYtMF9SQzIaAmVuIAEaBgiAt4avBg", "domain": ".youtube.com", "path": "/"},
]


_DISMISS_INIT_SCRIPT = r"""
(() => {
  const sels = [
    'button#L2AGLb',                       // google.com "Accept all"
    'button[aria-label*="Accept" i]',
    'button[aria-label*="I agree" i]',
    '#onetrust-accept-btn-handler',        // OneTrust
    '[data-testid="cookie-policy-accept"]',
    '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',  // Cookiebot
    'button[mode="primary"]',
  ];
  const try_click = () => {
    for (const s of sels) {
      try {
        const el = document.querySelector(s);
        if (el) { el.click(); return true; }
      } catch (e) {}
    }
    return false;
  };
  if (document.readyState !== 'loading') try_click();
  window.addEventListener('DOMContentLoaded', try_click);
  window.addEventListener('load', try_click);
  // Retry for late-rendered banners.
  setTimeout(try_click, 1500);
  setTimeout(try_click, 4000);
})();
"""


def seed_sandbox_browser(base_url: str = "http://localhost:8080") -> dict:
    """Pre-seed the cocoa sandbox's browser with consent cookies + dismiss script.

    Returns a small dict describing what happened (for logging). Never raises
    under normal conditions — a seeding failure must not break cocoa's run.
    """
    result = {"seeded": False, "cookies": 0, "cdp_url": None, "error": None}
    try:
        from agent_sandbox import Sandbox
        from playwright.sync_api import sync_playwright

        client = Sandbox(base_url=base_url)

        # Force Chromium to boot. get_info() returns no CDP URL until the
        # browser has actually been launched (lazy-boot behavior observed in
        # AIO sandbox 1.0.0.152).
        try:
            client.browser.screenshot()
        except Exception:
            pass

        info = client.browser.get_info()
        info_d = info.model_dump() if hasattr(info, "model_dump") else dict(info)
        data = (info_d or {}).get("data") or {}
        cdp_url = data.get("cdp_url") or data.get("cdpUrl") or data.get("webSocketDebuggerUrl")
        result["cdp_url"] = cdp_url
        if not cdp_url:
            result["error"] = "no cdp_url in browser.get_info().data"
            return result

        p = sync_playwright().start()
        try:
            browser = p.chromium.connect_over_cdp(cdp_url)
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            ctx.add_cookies(_CONSENT_COOKIES)
            ctx.add_init_script(_DISMISS_INIT_SCRIPT)
            result["seeded"] = True
            result["cookies"] = len(_CONSENT_COOKIES)
            browser.close()
        finally:
            p.stop()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result
