"""Faithfulness judge — fetches URLs the agent cited and verifies the
claims in the summary JSON against the live page content.

Two failure modes this catches that the existing LLM judge misses:
  1. Fabricated URLs (404 / DNS error / paywalled).
  2. Real URL but stat/quote/title in the agent's prose doesn't match
     anything on the page (the agent hallucinated and only attached a
     plausible URL).

Two verification modes:
  * substring (deterministic) — `verify_url_claims`: fetches the page,
    strips HTML, looks for the claim string (or regex) in the rendered
    text.  Fast, free, but limited to literal substring matching.
  * LLM-agentic — `verify_url_claims_llm`: fetches the page, hands the
    page text + the agent's claim (and/or an arbitrary natural-language
    question from the Effectiveness judge) to gpt-4o-mini, which returns
    a structured {supported, evidence_quote, reasoning, answer}.  This
    handles paraphrase, numeric tolerance, and cross-paragraph evidence
    that pure substring matching cannot.

The runner prefers LLM-agentic when an OPENAI_API_KEY is present and
falls back to substring otherwise.  Both modes return the same shape
(FaithfulnessFinding) so downstream aggregation is unchanged.

Soft-failing on network errors so the verifier doesn't get blocked by
transient outages.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Iterable


DEFAULT_TIMEOUT_S = 25
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
LLM_PAGE_BUDGET_CHARS = 8000  # how much page text we hand to the LLM per call
LLM_DEFAULT_MODEL = os.environ.get("EVOLVEBENCH_FAITH_MODEL", "gpt-4o-mini")


@dataclass
class FaithfulnessFinding:
    url: str
    fetched: bool
    http_status: int | None
    claim: str           # what the agent said the page contains
    claim_matched: bool  # did we find it / does the page support it
    detail: str
    # Optional LLM-agentic fields (empty for substring mode)
    question: str = ""           # natural-language question from the Eff judge
    answer: str = ""             # LLM's answer to that question
    evidence_quote: str = ""     # verbatim snippet from the page
    reasoning: str = ""          # LLM's brief justification
    mode: str = "substring"      # "substring" | "llm"

    def to_dict(self) -> dict:
        return asdict(self)


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Refuse non-http(s) schemes and hosts that resolve to private,
    loopback, link-local, or multicast addresses. Stops the judge from
    being turned into an SSRF probe by a hostile agent output."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"unparseable url: {e}"
    if parsed.scheme not in ("http", "https"):
        return False, f"refusing scheme {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return False, "no host"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return False, f"dns: {e}"
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False, f"refusing non-public address {ip_str}"
    return True, ""


_CHALLENGE_MARKERS = (
    "enable javascript", "please enable js",
    "verify you are human", "verify you're a human",
    "captcha", "cf-browser-verification", "cf-challenge",
    "amazon.com/errors", "sorry, we just need to make sure",
    "checking your browser before accessing",
    "access denied", "blocked because of malicious",
    "unusual traffic from your computer",
)


def _looks_like_bot_wall(text: str, status: int | None) -> bool:
    """Heuristic: does this response look like anti-bot chrome rather
    than real content?"""
    if status in (403, 429, 503) and len(text) < 8000:
        return True
    if not text:
        return True
    lower = text.lower()
    for m in _CHALLENGE_MARKERS:
        if m in lower:
            return True
    # Ultra-thin body after strip: nav-only shell
    stripped = text.strip()
    if len(stripped) < 800 and status == 200:
        return True
    return False


def _pdf_bytes_to_text(raw: bytes) -> str:
    """Extract text from a PDF byte string. Returns '' if pypdf is
    unavailable or the PDF has no extractable text (scanned image)."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        parts = []
        for page in reader.pages[:40]:  # cap pages for cost
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = re.sub(r"\s+", " ", "\n".join(parts))
        return text.strip()
    except Exception:
        return ""


def _looks_like_pdf(url: str, raw: bytes) -> bool:
    return url.lower().endswith(".pdf") or raw[:5] == b"%PDF-"


def _fetch_text_urllib(url: str, timeout: int) -> tuple[bool, int | None, str, str]:
    """Fast path: plain urllib. Good for static / server-rendered pages
    and PDFs (extracted with pypdf)."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            status = r.status
            raw = r.read(5_000_000)  # PDFs are larger than HTML
        if _looks_like_pdf(url, raw):
            text = _pdf_bytes_to_text(raw)
            if text:
                return True, status, text, ""
            return False, status, "", "pdf had no extractable text"
        text = re.sub(r"<script.*?</script>", " ", raw.decode("utf-8", "replace"), flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return True, status, text, ""
    except Exception as e:
        return False, None, "", f"{type(e).__name__}: {str(e)[:200]}"


_FALLBACK_UA_POOL = (
    DEFAULT_UA,
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
)

MAX_PLAYWRIGHT_RETRIES = 5


def _fetch_text_playwright(url: str, timeout: int, attempt: int = 0) -> tuple[bool, int | None, str, str]:
    """One Playwright fetch attempt. `attempt` (0-indexed) selects a
    slightly different UA / viewport / wait strategy so retries don't
    all look identical to server-side fingerprinters."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        return False, None, "", "playwright not installed"

    ua = _FALLBACK_UA_POOL[attempt % len(_FALLBACK_UA_POOL)]
    viewports = [
        {"width": 1440, "height": 900},
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 375,  "height": 812},  # mobile
        {"width": 1280, "height": 720},
    ]
    viewport = viewports[attempt % len(viewports)]
    # Escalate patience across retries — later attempts wait longer for JS.
    network_idle_ms = 6000 + 2000 * attempt
    body_ms = 5000 + 1500 * attempt

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ])
            context = browser.new_context(
                user_agent=ua,
                viewport=viewport,
                locale="en-US",
            )
            page = context.new_page()
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            status = resp.status if resp else None
            try:
                page.wait_for_load_state("networkidle", timeout=network_idle_ms)
            except Exception:
                pass
            body = page.inner_text("body", timeout=body_ms)
            browser.close()
        text = re.sub(r"\s+", " ", body or "").strip()
        return True, status, text, ""
    except Exception as e:
        return False, None, "", f"{type(e).__name__}: {str(e)[:200]}"


def _fetch_text_playwright_retrying(url: str, timeout: int,
                                      max_retries: int = MAX_PLAYWRIGHT_RETRIES) -> tuple[bool, int | None, str, str]:
    """Playwright + up to `max_retries` retries with UA/viewport rotation
    and exponential backoff when the response still looks like anti-bot
    chrome or thin content."""
    import time as _t
    best = (False, None, "", "no attempts")
    for attempt in range(max_retries):
        ok, status, text, err = _fetch_text_playwright(url, timeout, attempt=attempt)
        if ok and not _looks_like_bot_wall(text, status):
            return ok, status, text, err  # success on this attempt
        # Track best so far — prefer any successful ok=True over failed
        if ok and (not best[0] or len(text) > len(best[2])):
            best = (ok, status, text, err)
        elif not best[0]:
            best = (ok, status, text, err)
        # Back off before the next attempt: 0.5s, 1s, 2s, 4s
        if attempt < max_retries - 1:
            _t.sleep(0.5 * (2 ** attempt))
    return best


def _fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[bool, int | None, str, str]:
    """Return (ok, status, body_text, error_message).

    Fast urllib first; if the response looks like a bot wall (JS-only
    shell, challenge page, 403/429/503 with thin body), fall back to
    Playwright so the judge sees the same rendered DOM the agent saw."""
    safe, reason = _is_safe_url(url)
    if not safe:
        return False, None, "", reason

    ok, status, text, err = _fetch_text_urllib(url, timeout)
    if ok and not _looks_like_bot_wall(text, status):
        return True, status, text, ""

    # Fall back to a real browser with retries (rotates UA, viewport,
    # and JS-wait patience across attempts).
    ok2, status2, text2, err2 = _fetch_text_playwright_retrying(url, timeout)
    if ok2:
        return True, status2, text2, ""

    # Neither worked. Return whichever produced content, or the last error.
    if ok:
        return True, status, text, ""
    return False, status2 or status, "", (err2 or err)


def _normalize_claim(c: str) -> str:
    return re.sub(r"[\s ]+", " ", str(c)).strip().lower()


def verify_url_claims(checks: Iterable[dict], timeout: int = DEFAULT_TIMEOUT_S) -> list[FaithfulnessFinding]:
    """
    checks is an iterable of {url, claim} pairs where `claim` is a short
    string we expect to appear in the rendered page text (case-insensitive).
    The claim can be a literal substring or a regex pattern starting `re:`.
    """
    out: list[FaithfulnessFinding] = []
    for ch in checks:
        url = ch.get("url")
        claim = ch.get("claim", "")
        if not url:
            out.append(FaithfulnessFinding(
                url=url or "", fetched=False, http_status=None,
                claim=claim, claim_matched=False,
                detail="missing url"))
            continue
        ok, status, text, err = _fetch_text(url, timeout)
        if not ok:
            out.append(FaithfulnessFinding(
                url=url, fetched=False, http_status=status,
                claim=claim, claim_matched=False,
                detail=err))
            continue
        if not claim:
            # URL-only check (no specific claim) — just record the fetch
            out.append(FaithfulnessFinding(
                url=url, fetched=True, http_status=status,
                claim="", claim_matched=True,
                detail=f"page len={len(text)}"))
            continue
        # Claim match
        matched: bool
        if claim.startswith("re:"):
            pat = claim[3:]
            try:
                matched = bool(re.search(pat, text, re.I))
            except re.error as e:
                matched = False
                err = f"bad regex: {e}"
        else:
            matched = _normalize_claim(claim) in _normalize_claim(text)
        out.append(FaithfulnessFinding(
            url=url, fetched=True, http_status=status,
            claim=claim, claim_matched=matched,
            detail=f"page len={len(text)}" if matched else f"claim not found in page (len={len(text)})"))
    return out


# ============================================================
# Schema-guided URL-key grounding (Option A / two-turn re-extract)
# ============================================================

def _walk_url_key(summary, path_pattern: str):
    """Yield (urls_list, enclosing_dict) for every match of a path pattern
    like 'options[].url' or 'reference_product.url'.

    A URL field may hold either a single URL string or a LIST of URL
    strings (the agent is told it may cite several pages when the
    supporting evidence spans more than one). Either way one
    (urls, enclosing) unit is yielded per field occurrence — the
    grounding agent fetches the whole list and treats the pages as one
    combined evidence set for the sibling fields."""
    parts = path_pattern.split(".")

    def normalise(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str)]
        return []

    def rec(node, remaining):
        if not remaining:
            return
        head = remaining[0]
        tail = remaining[1:]
        if head.endswith("[]"):
            key = head[:-2]
            if isinstance(node, dict) and key in node and isinstance(node[key], list):
                for item in node[key]:
                    if not tail:
                        urls = normalise(item)
                        if urls:
                            yield urls, node
                    else:
                        yield from rec(item, tail)
        else:
            if isinstance(node, dict) and head in node:
                child = node[head]
                if not tail:
                    urls = normalise(child)
                    if urls:
                        yield urls, node
                else:
                    yield from rec(child, tail)

    yield from rec(summary, parts)


def _sibling_schema_fields(summary_schema, url_path_pattern: str) -> list[dict]:
    """Return the fields at the same nesting depth as the URL field —
    these are the extraction targets for the page the URL points to."""
    prefix = ".".join(url_path_pattern.split(".")[:-1])  # e.g. "options[]"
    prefix_dot = (prefix + ".") if prefix else ""
    out = []
    all_fields = list(getattr(summary_schema, "required", [])) + list(getattr(summary_schema, "optional", []))
    for f in all_fields:
        # Only same-level siblings under the same prefix, and skip the URL field itself.
        if not f.path.startswith(prefix_dot):
            continue
        tail = f.path[len(prefix_dot):]
        if "." in tail or tail.endswith("[]"):
            continue  # deeper nesting or nested list — not a direct sibling scalar
        # Skip the URL field itself
        if f.path == url_path_pattern:
            continue
        # Skip anything that looks like another URL field
        if tail.lower().endswith("_url") or tail.lower() == "url":
            continue
        out.append({"path": tail, "type_hint": getattr(f, "type_hint", "string"),
                    "description": getattr(f, "description", "")})
    return out


def _get(node, dotted):
    cur = node
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def verify_url_keys(summary: dict, url_key_paths: list[str], summary_schema,
                    timeout: int = DEFAULT_TIMEOUT_S,
                    model: str = "gpt-4o-mini") -> "list[FaithfulnessFinding]":
    """Schema-guided forward re-extraction with per-field cross-check.

    For each URL discovered at any of the `url_key_paths` in the agent's
    summary:
      1. Identify sibling schema fields (same nesting depth) — these are
         what the agent claims to have sourced from this URL's page.
      2. Fetch the page.
      3. Ask an LLM to independently extract those sibling fields from
         the page, without seeing the agent's answer.
      4. Compare judge-extracted values against agent-reported values
         per field.

    Each URL becomes one FaithfulnessFinding whose `claim_matched` is
    True iff at least half of the checked fields matched (EXACT or
    EQUIVALENT). The finding's `detail` carries the per-field JSON so
    the Effectiveness judge can read it.
    """
    findings: list[FaithfulnessFinding] = []
    llm_available = bool(os.environ.get("OPENAI_API_KEY"))

    for pattern in url_key_paths:
        siblings = _sibling_schema_fields(summary_schema, pattern)
        for urls, enclosing in _walk_url_key(summary, pattern):
            # Guard: only ground genuine web URLs. Local file paths, numbers,
            # names, or other strings mistakenly listed as URL keys are
            # dropped from the list (never counted as failures).
            urls = [u for u in urls
                    if isinstance(u, str) and u.lower().startswith(("http://", "https://"))]
            if not urls:
                continue
            # Agent's reported values for the sibling fields
            agent_reported = {s["path"]: enclosing.get(s["path"]) for s in siblings}

            # Fetch EVERY URL in the list; combine the fetched pages into
            # one evidence set. Split evidence across pages is legitimate —
            # the agent was told it may cite several pages when no single
            # page carries all the values.
            fetched_parts = []
            statuses = []
            errors = []
            for u in urls[:5]:  # cap per-field list to bound cost
                ok, status, text, err = _fetch_text(u, timeout)
                statuses.append(status)
                if ok:
                    fetched_parts.append(f"=== PAGE: {u} ===\n{text}")
                else:
                    errors.append(f"{u}: {err}")
            url_label = urls[0] if len(urls) == 1 else f"{urls[0]} (+{len(urls)-1} more)"
            status = statuses[0] if statuses else None
            if not fetched_parts:
                findings.append(FaithfulnessFinding(
                    url=url_label, fetched=False, http_status=status,
                    claim=f"per-field check across {len(siblings)} sibling fields",
                    claim_matched=False,
                    detail=f"all fetches failed: {'; '.join(errors)[:300]}",
                    mode="llm" if llm_available else "substring",
                ))
                continue
            text = "\n\n".join(fetched_parts)
            url = url_label

            if not siblings:
                findings.append(FaithfulnessFinding(
                    url=url, fetched=True, http_status=status,
                    claim="url reachable",
                    claim_matched=True,
                    detail="reachable; no sibling schema fields to cross-check",
                    mode="llm" if llm_available else "substring",
                ))
                continue

            if not llm_available:
                # Substring fallback — check if agent-reported string values appear on the page
                page_lc = text.lower()
                matched = 0
                per_field = {}
                for s in siblings:
                    v = agent_reported.get(s["path"])
                    if isinstance(v, str) and v.strip() and v.lower() in page_lc:
                        matched += 1
                        per_field[s["path"]] = "EXACT"
                    else:
                        per_field[s["path"]] = "UNKNOWN"
                findings.append(FaithfulnessFinding(
                    url=url, fetched=True, http_status=status,
                    claim=f"{matched}/{len(siblings)} sibling fields substring-matched",
                    claim_matched=matched >= (len(siblings) + 1) // 2,
                    detail=json.dumps(per_field),
                    mode="substring",
                ))
                continue

            # LLM path: two turns
            schema_lines = "\n".join(
                f'- {s["path"]} ({s["type_hint"]}) — {s["description"]}' for s in siblings
            )
            extract_prompt = (
                "You are the grading system. Independently extract the following "
                "typed fields from the webpage below into a JSON object matching the "
                "field paths exactly. Use null for any field you cannot verify from "
                "the page. Do NOT infer, do NOT guess — only report what the page "
                "states.\n\n"
                "IMPORTANT — enumerate variants when several are valid:\n"
                "  * If the field type is a list (list[string], list[object], "
                "list[...]), include EVERY item present on the page, not just a "
                "subset. For a product's features, list ALL feature bullets you "
                "see. For citations, list every citation. Etc.\n"
                "  * If the field is a scalar (string / number / boolean) but "
                "the page shows more than one valid value for it — e.g. a "
                "product title appears in a short form, a long form, and a "
                "breadcrumb — return a LIST of the variants you saw instead of "
                "a single string. The consumer of your JSON will treat a list "
                "as \"any of these\". If there is only one obvious value, "
                "return the scalar as usual.\n\n"
                f"SOURCE: {url}\n\nEXPECTED FIELDS:\n{schema_lines}\n\n"
                "PAGE TEXT (may contain several pages, each introduced by "
                "'=== PAGE: <url> ==='; a field counts as verified if ANY "
                "page supports it):\n"
                f"{text}\n\n"
                "Return only a JSON object. No preamble."
            )
            judge_extracted = _openai_chat(extract_prompt, model=model)
            try:
                je_json = json.loads(_strip_fences(judge_extracted))
            except Exception:
                je_json = None

            if je_json is None:
                findings.append(FaithfulnessFinding(
                    url=url, fetched=True, http_status=status,
                    claim="judge extraction produced non-JSON",
                    claim_matched=False,
                    detail=(judge_extracted or "")[:300],
                    mode="llm",
                ))
                continue

            cross_prompt = (
                "For each field, decide whether the AGENT's value and the JUDGE's "
                "independent extraction from the same page agree. Reply with a "
                "single JSON object mapping each field path to one of: EXACT, "
                "EQUIVALENT, MISMATCH, JUDGE_NULL.\n\n"
                "BE GENEROUS — the goal is to grade whether the agent's answer is "
                "grounded in the page, not to demand character-perfect equality. "
                "Default to EQUIVALENT whenever the two sides are consistent; "
                "reserve MISMATCH for actual factual contradictions.\n\n"
                "SUBSET / SHORTENING RULE — the core case. A shorter or "
                "subsetted version of the other side is EQUIVALENT, as long as "
                "no fact in the shorter version contradicts the longer one:\n"
                "  * Scalar string: agent gives \"Tolaccea 40L Carry On Backpack\", "
                "judge gives \"Tolaccea Travel Backpack for Men Women, 40L-50L "
                "Carry On Backpack with Wet Dry Compartment, TSA Approved "
                "Laptop Daypack Fits 15.6\\\"\" → EQUIVALENT. Agent's string is "
                "a shortening/subset of the judge's; nothing in it contradicts.\n"
                "  * List: agent's list has 3 items, judge's list has 8 items, "
                "and every agent item is informationally similar to something "
                "in the judge's list (or vice versa). No contradiction → "
                "EQUIVALENT. It does NOT matter that the two lists don't fully "
                "overlap; either side is allowed to be a subset of the other.\n"
                "  * Two disjoint lists that both truthfully describe the same "
                "underlying entity (e.g. two different subsets of a product's "
                "feature bullets, neither contradicts the other): EQUIVALENT.\n\n"
                "OTHER RULES:\n"
                "  * If the JUDGE's value is a LIST of variants, the agent's "
                "value is EQUIVALENT if it matches ANY ONE of the judge's "
                "variants (variants signal \"any of these is fine\").\n"
                "  * For numbers, ±1% or a rounding-level difference is "
                "EQUIVALENT.\n"
                "  * For casing/whitespace/punctuation-only differences, "
                "EXACT is fine (or EQUIVALENT).\n"
                "  * MISMATCH is reserved for factual contradictions: "
                "different product/entity, wrong number outside tolerance, "
                "opposite claim.\n"
                "  * Use JUDGE_NULL when the judge did not extract anything "
                "for this field.\n\n"
                f"AGENT REPORTED:\n{json.dumps(agent_reported, indent=2)}\n\n"
                f"JUDGE EXTRACTED (may include lists of variants):\n{json.dumps(je_json, indent=2)}\n\n"
                "Return only the JSON verdict object."
            )
            cross_raw = _openai_chat(cross_prompt, model=model)
            try:
                verdicts = json.loads(_strip_fences(cross_raw))
            except Exception:
                verdicts = None

            if not isinstance(verdicts, dict):
                findings.append(FaithfulnessFinding(
                    url=url, fetched=True, http_status=status,
                    claim="cross-check produced non-JSON",
                    claim_matched=False,
                    detail=(cross_raw or "")[:300],
                    mode="llm",
                ))
                continue

            matched = sum(1 for v in verdicts.values() if v in ("EXACT", "EQUIVALENT"))
            total = len(verdicts)

            findings.append(FaithfulnessFinding(
                url=url, fetched=True, http_status=status,
                claim=f"{matched}/{total} fields agreed (EXACT or EQUIVALENT)",
                claim_matched=matched >= (total + 1) // 2,
                detail=json.dumps({
                    "agent": agent_reported,
                    "judge": je_json,
                    "verdicts": verdicts,
                    "matched_fields": matched,
                    "total_fields": total,
                }, ensure_ascii=False),
                mode="llm",
            ))

    return findings


def _openai_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    r = OpenAI(**kwargs).chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content or ""


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def aggregate(findings: list[FaithfulnessFinding]) -> dict:
    """Roll up to a single score in [0,5] for the effectiveness mixer."""
    if not findings:
        return {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0,
                "details": []}
    fetched = sum(1 for f in findings if f.fetched)
    matched = sum(1 for f in findings if f.claim_matched)
    # Weight: 0.4 fetched, 0.6 claim-matched
    norm = 0.4 * fetched / len(findings) + 0.6 * matched / len(findings)
    mode = "llm" if any(f.mode == "llm" for f in findings) else "substring"
    return {"score_5": round(5.0 * norm, 2),
            "fetched": fetched, "matched": matched, "total": len(findings),
            "mode": mode,
            "details": [f.to_dict() for f in findings]}


# ============================================================
# LLM-agentic faithfulness verifier
# ============================================================

_LLM_FAITH_SYSTEM = """You are a precise source-checking sub-agent.  You are given:
  - the rendered text of a web page the AI agent cited as evidence,
  - the AI agent's specific claim about that page,
  - (optionally) a follow-up question from the Effectiveness judge.

Decide whether the page actually supports the claim, and answer the question
if one is given.  Be strict: paraphrases of the same fact count as support;
numeric values must be within plausible rounding; if the page is empty,
paywalled, or off-topic, mark unsupported.

Return ONLY a JSON object with this shape:
  {
    "supported": true|false,
    "evidence_quote": "<verbatim snippet from the page, <=40 words; '' if none>",
    "answer": "<answer to the question, or '' if no question>",
    "reasoning": "<one short sentence>"
  }
No extra prose.
"""


def _llm_verify_one(*, page_text: str, claim: str, question: str,
                     model: str, max_retries: int = 3) -> dict:
    """Call the LLM faithfulness sub-agent once.  Returns parsed dict
    or a {error: ...} fallback."""
    try:
        import openai
    except ImportError:
        return {"error": "openai library not available"}
    excerpt = (page_text or "")[:LLM_PAGE_BUDGET_CHARS]
    user = (
        f"## Page text (first {LLM_PAGE_BUDGET_CHARS} chars; whitespace-collapsed)\n"
        f"{excerpt}\n\n"
        f"## Agent's claim\n{claim or '(none)'}\n\n"
        f"## Follow-up question from the Effectiveness judge\n{question or '(none)'}\n"
    )
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                            base_url=os.environ.get("OPENAI_BASE_URL") or None)
    last_err = ""
    for attempt in range(max_retries):
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_FAITH_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=500,
                temperature=0,
            )
            raw = (comp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, re.S)
            blob = m.group(0) if m else raw
            try:
                return json.loads(blob)
            except json.JSONDecodeError as e:
                last_err = f"json: {e}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(1.5 ** attempt)
    return {"error": last_err}


def verify_url_claims_llm(checks: Iterable[dict],
                            timeout: int = DEFAULT_TIMEOUT_S,
                            model: str | None = None,
                            ) -> list[FaithfulnessFinding]:
    """LLM-agentic faithfulness verifier.

    Each check is a dict with:
      - url    : required
      - claim  : optional — the agent's claim about the page
      - question : optional — follow-up question from the Eff judge

    Returns FaithfulnessFinding per check, with the same shape as
    `verify_url_claims` so aggregate() works either way.
    """
    out: list[FaithfulnessFinding] = []
    mdl = model or LLM_DEFAULT_MODEL
    for ch in checks:
        url = ch.get("url")
        claim = ch.get("claim", "") or ""
        question = ch.get("question", "") or ""
        if not url:
            out.append(FaithfulnessFinding(
                url=url or "", fetched=False, http_status=None,
                claim=claim, claim_matched=False,
                detail="missing url", question=question, mode="llm"))
            continue
        ok, status, text, err = _fetch_text(url, timeout)
        if not ok:
            out.append(FaithfulnessFinding(
                url=url, fetched=False, http_status=status,
                claim=claim, claim_matched=False,
                detail=err, question=question, mode="llm"))
            continue
        verdict = _llm_verify_one(
            page_text=text, claim=claim, question=question, model=mdl)
        if "error" in verdict:
            # LLM unavailable — fall back to substring on the same page text
            matched = (bool(claim) and
                        _normalize_claim(claim) in _normalize_claim(text))
            out.append(FaithfulnessFinding(
                url=url, fetched=True, http_status=status,
                claim=claim, claim_matched=matched,
                detail=f"llm error → substring fallback: {verdict['error']}",
                question=question, mode="llm"))
            continue
        supported = bool(verdict.get("supported"))
        # If no claim was given but a question was, treat the question's
        # successful answering as the "claim_matched" signal.
        if not claim and question:
            supported = bool(verdict.get("answer", "").strip())
        out.append(FaithfulnessFinding(
            url=url, fetched=True, http_status=status,
            claim=claim, claim_matched=supported,
            detail=(f"page len={len(text)}; "
                     f"{verdict.get('reasoning','')[:120]}"),
            question=question,
            answer=verdict.get("answer", "") or "",
            evidence_quote=verdict.get("evidence_quote", "") or "",
            reasoning=verdict.get("reasoning", "") or "",
            mode="llm"))
    return out


def verify(checks: Iterable[dict],
            timeout: int = DEFAULT_TIMEOUT_S,
            prefer_llm: bool | None = None,
            model: str | None = None,
            ) -> list[FaithfulnessFinding]:
    """Smart dispatcher: LLM-agentic when an API key is available, else
    substring-match.  This is the function the runner should call."""
    if prefer_llm is None:
        prefer_llm = bool(os.environ.get("OPENAI_API_KEY"))
    if prefer_llm:
        return verify_url_claims_llm(checks, timeout=timeout, model=model)
    return verify_url_claims(checks, timeout=timeout)
