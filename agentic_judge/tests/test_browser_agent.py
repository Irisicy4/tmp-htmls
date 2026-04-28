import json
from unittest.mock import patch

from agentic_judge.core.browser_agent import (
    _get_prompt_for_category,
    _build_observations_text,
    _parse_gpt_response,
    _determine_null_reason,
    verify,
)


def test_get_prompt_for_category_shopping():
    template = _get_prompt_for_category("Shopping")
    assert isinstance(template, str)
    assert "{task_instruction}" in template
    assert "{claimed_result}" in template


def test_build_observations_text_loaded_page():
    observations = [
        {"url": "https://amazon.com/dp/X", "status": "loaded", "title": "Amazon Product", "screenshot_b64": "abc123"},
    ]
    text = _build_observations_text(observations)
    assert "https://amazon.com/dp/X" in text
    assert "loaded" in text
    assert "Amazon Product" in text


def test_build_observations_text_error_page():
    observations = [
        {"url": "https://gone.com", "status": "error", "error": "Timeout exceeded", "title": None, "screenshot_b64": None},
    ]
    text = _build_observations_text(observations)
    assert "gone.com" in text
    assert "Timeout exceeded" in text


def test_parse_gpt_response_valid():
    raw = '<Answer>{"verified": true, "finding": "Page loaded fine.", "confidence": "high"}</Answer>'
    result = _parse_gpt_response(raw)
    assert result["verified"] is True
    assert result["finding"] == "Page loaded fine."
    assert result["confidence"] == "high"


def test_parse_gpt_response_null_verified():
    raw = '<Answer>{"verified": null, "finding": "Ambiguous content.", "confidence": "low"}</Answer>'
    result = _parse_gpt_response(raw)
    assert result["verified"] is None


def test_parse_gpt_response_false():
    raw = '<Answer>{"verified": false, "finding": "Page is 404.", "confidence": "high"}</Answer>'
    result = _parse_gpt_response(raw)
    assert result["verified"] is False


def test_determine_null_reason_verified():
    assert _determine_null_reason(verified=True, all_errored=False) is None
    assert _determine_null_reason(verified=False, all_errored=False) is None


def test_determine_null_reason_gpt_uncertain():
    assert _determine_null_reason(verified=None, all_errored=False) == "gpt_uncertain"


def test_determine_null_reason_navigation_error():
    assert _determine_null_reason(verified=None, all_errored=True) == "navigation_error"


def test_verify_returns_expected_shape():
    mock_obs = [
        {"url": "https://amazon.com", "status": "loaded", "title": "Amazon", "screenshot_b64": "base64data"}
    ]
    mock_gpt = {"verified": True, "finding": "Amazon page loaded.", "confidence": "high"}

    with patch("agentic_judge.core.browser_agent._run_playwright", return_value=mock_obs), \
         patch("agentic_judge.core.browser_agent._call_gpt4o", return_value=mock_gpt):
        result = verify(
            instruction="Find a backpack on Amazon",
            claimed_result="I found a backpack at https://amazon.com",
            urls=["https://amazon.com"],
        )

    assert result["verified"] is True
    assert result["verification_method"] == "browser_navigation"
    assert result["null_reason"] is None
    assert "finding" in result


def test_verify_all_errors_returns_url_check():
    mock_obs = [
        {"url": "https://gone.com", "status": "error", "error": "Timeout", "title": None, "screenshot_b64": None}
    ]
    mock_gpt = {"verified": None, "finding": "Page timed out.", "confidence": "low"}

    with patch("agentic_judge.core.browser_agent._run_playwright", return_value=mock_obs), \
         patch("agentic_judge.core.browser_agent._call_gpt4o", return_value=mock_gpt):
        result = verify(
            instruction="Check gone.com",
            claimed_result="I checked it",
            urls=["https://gone.com"],
        )

    assert result["verification_method"] == "url_check"
    assert result["verified"] is None
    assert result["null_reason"] == "navigation_error"


def test_get_prompt_for_category_routing():
    shopping = _get_prompt_for_category("Shopping")
    finance = _get_prompt_for_category("Finance & Economics")
    travel = _get_prompt_for_category("Travel & Planning")
    software = _get_prompt_for_category("Software Engineering")
    media = _get_prompt_for_category("(Self) Media")
    daily = _get_prompt_for_category("Daily Activities")
    unknown = _get_prompt_for_category("Unknown")

    # Each archetype prompt should be distinct
    assert shopping != finance
    assert travel != software
    # Fallback cases should use the same general prompt
    assert daily == unknown
    # All must have the 5 required placeholders
    for prompt in [shopping, finance, travel, software, media, daily]:
        assert "{task_instruction}" in prompt
        assert "{claimed_result}" in prompt
        assert "{execution_summary}" in prompt
        assert "{browser_observations}" in prompt
        assert "{deliverable_url}" in prompt
