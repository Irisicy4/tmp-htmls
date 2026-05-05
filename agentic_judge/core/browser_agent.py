"""Browser-based verification agent using Playwright + GPT-4o vision."""

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

ANSWER_RE = re.compile(r'<Answer>(.*?)</Answer>', re.DOTALL)

_PROMPT_CACHE: dict[str, str] = {}

# Maps domain strings (lowercased) to prompt filename stems
_DOMAIN_TO_ARCHETYPE: dict[str, str] = {
    "shopping": "shopping",
    "data & ml engineering": "research_data",
    "finance & economics": "research_data",
    "legal": "research_data",
    "insurance & actuarial": "research_data",
    "medical & clinical & bio": "research_data",
    "marketing & analytics": "research_data",
    "travel & planning": "travel_planning",
    "real estate": "travel_planning",
    "logistics & supply chain": "travel_planning",
    "software engineering": "software_tech",
    "(self) media": "content_media",
    "design": "content_media",
    "hr & recruiting": "content_media",
    # domains present in this repo's task sets
    "daily activities": "general",
    "daily activities ": "general",  # guard against trailing whitespace variant
}

# Per-archetype criteria injected into the shared verify_prompt.txt template
_CATEGORY_GUIDANCE: dict[str, str] = {
    "shopping": (
        "## Judgment criteria for Shopping tasks:\n"
        "\n"
        "1. Is the product page (or search results page) accessible — does it load without a 404 or block?\n"
        "2. Do the products visible on the page match the category/type the agent was asked to research?\n"
        "3. Are the prices, names, or features the agent cited plausible given what appears on the page?\n"
        "4. If the agent cited a specific product URL, does that product exist on the page?\n"
        "\n"
        "Focus on: product existence, price plausibility, category match. Do NOT penalise the agent for minor price differences due to dynamic pricing."
    ),
    "research_data": (
        "## Judgment criteria for Research & Data tasks:\n"
        "(Domains: Data & ML Engineering, Finance & Economics, Legal, Insurance & Actuarial, Medical & Clinical & Bio, Marketing & Analytics)\n"
        "\n"
        "1. Is the source page accessible — does it load without a 404 or paywall block?\n"
        "2. Does the page contain data, documents, or information relevant to the task topic?\n"
        "3. Are the specific figures, names, dates, or statistics the agent cited findable on or plausibly derived from this page?\n"
        "4. If the agent cited a specific data source or document, does that source appear to exist on the page?\n"
        "\n"
        "Focus on: source accessibility, topical relevance, factual plausibility. Do NOT require exact figure matches — data pages update dynamically."
    ),
    "travel_planning": (
        "## Judgment criteria for Travel & Planning tasks:\n"
        "(Domains: Travel & Planning, Real Estate, Logistics & Supply Chain)\n"
        "\n"
        "1. Does the listing, property, route, or destination page load without a 404 or redirect to a homepage?\n"
        "2. Does the page show the location, property, or route the agent claimed to have researched?\n"
        "3. Are the prices, availability dates, addresses, or logistics details the agent cited consistent with what appears on the page?\n"
        "4. If the agent booked, saved, or confirmed something, is there any evidence of that action on the page?\n"
        "\n"
        "Focus on: listing existence, location/address match, price/availability plausibility."
    ),
    "software_tech": (
        "## Judgment criteria for Software Engineering tasks:\n"
        "(Domain: Software Engineering)\n"
        "\n"
        "1. Does the repository, documentation page, package page, or tool URL load without a 404?\n"
        "2. Does the page show the project, library, or tool the agent claimed to have found or used?\n"
        "3. Are the version numbers, API signatures, feature descriptions, or install instructions the agent cited consistent with what appears on the page?\n"
        "4. If the agent claimed to have filed an issue, submitted a PR, or made a change, is there evidence of that on the page?\n"
        "\n"
        "Focus on: project existence, version/API plausibility, claimed actions. Do NOT penalise for version differences due to releases after the task was run."
    ),
    "content_media": (
        "## Judgment criteria for Content & Media tasks:\n"
        "(Domains: (Self) Media, Design, HR & Recruiting)\n"
        "\n"
        "1. Does the profile, post, channel, job listing, or design page load without a 404 or login wall?\n"
        "2. Does the page show the creator, publication, job, or design work the agent claimed to have found?\n"
        "3. Are the titles, descriptions, dates, follower counts, or other metadata the agent cited consistent with what appears on the page?\n"
        "4. If the agent claimed to have submitted, posted, or applied to something, is there evidence of that action?\n"
        "\n"
        "Focus on: content existence, identity match, metadata plausibility."
    ),
    "general": (
        "## Judgment criteria:\n"
        "\n"
        "1. Does the page load without a 404 or hard error?\n"
        "2. Does the page contain content that is relevant to the task the agent was asked to perform?\n"
        "3. Is the agent's claimed result broadly consistent with what the browser found on the page?\n"
        "4. If pages loaded but you genuinely cannot determine whether the claim is correct from the available evidence, return verified: null."
    ),
}


def _get_prompt_for_category(category: str) -> str:
    """Return the prompt template for the given domain/category string.

    Looks up the archetype via _DOMAIN_TO_ARCHETYPE (case-insensitive).
    Falls back to the general archetype for unknown or Daily Activities domains.
    Caches each injected template by archetype name.
    """
    archetype = _DOMAIN_TO_ARCHETYPE.get(category.lower(), "general")
    if archetype not in _PROMPT_CACHE:
        if "_template" not in _PROMPT_CACHE:
            prompt_path = Path(__file__).parent.parent / "prompts" / "verify_prompt.txt"  # assumes run from repo root
            _PROMPT_CACHE["_template"] = prompt_path.read_text()
        _PROMPT_CACHE[archetype] = _PROMPT_CACHE["_template"].replace(
            "{category_guidance}", _CATEGORY_GUIDANCE[archetype]
        )
    return _PROMPT_CACHE[archetype]


def _build_observations_text(observations: list[dict]) -> str:
    """Build plain-text summary of browser observations for the prompt."""
    lines = []
    for i, obs in enumerate(observations, 1):
        lines.append(f"--- URL {i}: {obs['url']} ---")
        lines.append(f"Status: {obs['status']}")
        if obs.get("title"):
            lines.append(f"Page title: {obs['title']}")
        if obs.get("error"):
            lines.append(f"Error: {obs['error']}")
        if obs.get("page_text"):
            lines.append(f"Page content:\n{obs['page_text']}")
        if obs.get("screenshot_b64"):
            lines.append("[Screenshot attached]")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_gpt_response(raw: str) -> dict:
    """Parse <Answer>JSON</Answer> from GPT-4o response.

    Falls back to parsing the entire response as JSON if the tag is absent.
    Raises ValueError if neither parse succeeds.
    """
    match = ANSWER_RE.search(raw)
    if match:
        return json.loads(match.group(1))
    return json.loads(raw)


def _determine_null_reason(verified: Optional[bool], all_errored: bool) -> Optional[str]:
    """Return null_reason string or None.

    - None when verified is True or False (not null)
    - 'navigation_error' when all pages errored AND verified is null
    - 'gpt_uncertain' when at least one page loaded but GPT returned null
    """
    if verified is not None:
        return None
    if all_errored:
        return "navigation_error"
    return "gpt_uncertain"


def _call_gpt4o(
    instruction: str,
    claimed_result: str,
    observations: list[dict],
    model: str = "gpt-4o",
    category: str = "Unknown",
    execution_summary: str = "",
) -> dict:
    """Call GPT-4o with observations (optionally including screenshots)."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    template = _get_prompt_for_category(category)

    obs_text = _build_observations_text(observations)
    deliverable_url = observations[0]["url"] if observations else "none"

    user_text = template.format(
        task_instruction=instruction,
        claimed_result=claimed_result,
        execution_summary=execution_summary or "Not available.",
        browser_observations=obs_text,
        deliverable_url=deliverable_url,
    )

    # Build content list — text first, then screenshots
    content: list[dict] = [{"type": "text", "text": user_text}]
    for obs in observations:
        if obs.get("screenshot_b64"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{obs['screenshot_b64']}"},
            })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return _parse_gpt_response(response.choices[0].message.content)


async def _navigate_single_url(page, url: str, timeout_ms: int = 15000) -> dict:
    """Navigate to one URL, return observation dict."""
    try:
        await page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        title = await page.title()
        try:
            page_text = await page.inner_text("body")
            page_text = page_text[:8000]  # trim — important content is near the top
        except Exception:
            page_text = ""
        screenshot_bytes = await page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
        return {
            "url": url,
            "status": "loaded",
            "title": title,
            "page_text": page_text,
            "screenshot_b64": screenshot_b64,
            "error": None,
        }
    except Exception as exc:
        return {
            "url": url,
            "status": "error",
            "title": None,
            "page_text": "",
            "screenshot_b64": None,
            "error": str(exc),
        }


def _run_playwright(urls: list[str], timeout_ms: int = 15000) -> list[dict]:
    """Navigate up to 3 URLs with Playwright, returning observations."""
    from playwright.async_api import async_playwright  # local import — optional dep

    async def _run() -> list[dict]:
        observations = []
        action_count = 0
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for url in urls[:3]:
                if action_count >= 10:
                    break
                obs = await _navigate_single_url(page, url, timeout_ms=timeout_ms)
                observations.append(obs)
                action_count += 1
            await browser.close()
        return observations

    return asyncio.run(_run())


def verify(
    instruction: str,
    claimed_result: str,
    urls: list[str],
    model: str = "gpt-4o",
    timeout_seconds: int = 120,
    category: str = "Unknown",
    execution_summary: str = "",
) -> dict:
    """Verify a task result by navigating to its URLs and querying GPT-4o.

    Args:
        instruction: Original task instruction.
        claimed_result: Agent's task_result text.
        urls: URLs to navigate (instruction URLs first, already prioritised by caller).
        model: OpenAI model to use for judgment.
        timeout_seconds: Total timeout budget (used as Playwright per-page timeout).
        category: Domain/category string used to select the archetype prompt.
        execution_summary: Tool call trace from the agent run (ground truth of what it did).

    Returns:
        Dict with keys: verified, finding, confidence, null_reason, verification_method.
    """
    timeout_ms = min(timeout_seconds * 1000, 15000)

    try:
        observations = _run_playwright(urls, timeout_ms=timeout_ms)
    except Exception as exc:
        return {
            "verified": None,
            "finding": f"Playwright launch failed: {exc}",
            "confidence": "low",
            "null_reason": "navigation_error",
            "verification_method": "url_check",
        }

    has_screenshot = any(obs.get("screenshot_b64") for obs in observations)
    all_errored = all(obs["status"] == "error" for obs in observations)

    verification_method = "browser_navigation" if has_screenshot else "url_check"

    try:
        gpt_result = _call_gpt4o(
            instruction, claimed_result, observations,
            model=model, category=category, execution_summary=execution_summary,
        )
        verified = gpt_result.get("verified")
        finding = gpt_result.get("finding", "No finding returned.")
        confidence = gpt_result.get("confidence", "low")
    except Exception as exc:
        verified = None
        finding = f"GPT-4o call failed: {exc}"
        confidence = "low"

    null_reason = _determine_null_reason(verified=verified, all_errored=all_errored)

    return {
        "verified": verified,
        "finding": finding,
        "confidence": confidence,
        "null_reason": null_reason,
        "verification_method": verification_method,
    }
