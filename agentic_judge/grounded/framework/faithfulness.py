"""Faithfulness judge — fetches URLs the agent cited and verifies the
claims in the summary JSON against the live page content.

Two failure modes this catches that the existing LLM judge misses:
  1. Fabricated URLs (404 / DNS error / paywalled).
  2. Real URL but stat/quote/title in the agent's prose doesn't match
     anything on the page (the agent hallucinated and only attached a
     plausible URL).

Returns a structured report with per-check verdicts.  Soft-failing on
network errors so the verifier doesn't get blocked by transient outages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable


DEFAULT_TIMEOUT_S = 25
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class FaithfulnessFinding:
    url: str
    fetched: bool
    http_status: int | None
    claim: str           # what the agent said the page contains
    claim_matched: bool  # did we find it
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[bool, int | None, str, str]:
    """Return (ok, status, body_text, error_message)."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            status = r.status
            raw = r.read(2_000_000)  # cap at 2 MB
        # Quick HTML→text: strip tags, collapse whitespace
        text = re.sub(r"<script.*?</script>", " ", raw.decode("utf-8", "replace"), flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return True, status, text, ""
    except Exception as e:
        return False, None, "", f"{type(e).__name__}: {str(e)[:200]}"


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


def aggregate(findings: list[FaithfulnessFinding]) -> dict:
    """Roll up to a single score in [0,5] for the effectiveness mixer."""
    if not findings:
        return {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0,
                "details": []}
    fetched = sum(1 for f in findings if f.fetched)
    matched = sum(1 for f in findings if f.claim_matched)
    # Weight: 0.4 fetched, 0.6 claim-matched
    norm = 0.4 * fetched / len(findings) + 0.6 * matched / len(findings)
    return {"score_5": round(5.0 * norm, 2),
            "fetched": fetched, "matched": matched, "total": len(findings),
            "details": [f.to_dict() for f in findings]}
