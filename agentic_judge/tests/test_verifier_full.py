import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agentic_judge.core.verifier import verify_task

# Minimal result JSON matching this repo's flat format (result.json inside trial dir)
RESULT_JSON = {
    "task": "task-01-test-backpack",
    "passed": True,
    "score": 3.5,
    "dimension_scores": {"constraint_satisfaction": 5, "result_specificity": 3},
    "category": "Shopping",
}

# Agent output lives in a sibling agent_result.json
AGENT_RESULT_JSON = {
    "task_result": "I found backpacks on Amazon.",
}

TASK_TOML = 'version = "1.0"\n\n[metadata]\ncategory = "Shopping"\n'

INSTRUCTION_MD = "Find backpacks under $75 from https://www.amazon.com/dp/B09YRC9Y3G\n"

BROWSER_RESULT = {
    "verified": True,
    "finding": "Amazon product page loaded, listing matches backpack claim.",
    "confidence": "high",
    "null_reason": None,
    "verification_method": "browser_navigation",
}


def _write_fixtures(tmpdir: str) -> tuple[str, str]:
    """Write result.json + agent_result.json and task.toml + instruction.md.

    Returns (result_path, task_dir).
    """
    # Trial dir mirrors the real repo: trials/<task-name>/result.json
    trial_dir = os.path.join(tmpdir, "task-01-test-backpack")
    os.makedirs(trial_dir)
    result_path = os.path.join(trial_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(RESULT_JSON, f)
    with open(os.path.join(trial_dir, "agent_result.json"), "w") as f:
        json.dump(AGENT_RESULT_JSON, f)

    task_dir = os.path.join(tmpdir, "tasks", "task-01-test-backpack")
    os.makedirs(task_dir)
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write(TASK_TOML)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(INSTRUCTION_MD)

    return result_path, task_dir


def test_verify_task_with_url_returns_full_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path, task_dir = _write_fixtures(tmpdir)
        with patch("agentic_judge.core.verifier.browser_verify", return_value=BROWSER_RESULT):
            result = verify_task(result_path, task_dir)

    assert result["task_name"] == "task-01-test-backpack"
    assert result["category"] == "Shopping"
    assert result["static_judge_passed"] is True
    assert result["static_judge_score"] == 3.5
    assert result["static_judge_dimensions"] == {"constraint_satisfaction": 5, "result_specificity": 3}
    assert result["agentic_verified"] is True
    assert result["null_reason"] is None
    assert result["verification_method"] == "browser_navigation"
    assert result["deliverable_url"] == "https://www.amazon.com/dp/B09YRC9Y3G"
    assert isinstance(result["time_seconds"], float)


def test_verify_task_no_urls_returns_unverifiable():
    task_toml_no_urls = 'version = "1.0"\n\n[metadata]\ncategory = "Shopping"\n'
    instruction_no_urls = "Research backpacks and summarize your findings.\n"
    result_no_url = {**RESULT_JSON}
    agent_no_url = {"task_result": "I found backpacks but no URLs here."}

    with tempfile.TemporaryDirectory() as tmpdir:
        trial_dir = os.path.join(tmpdir, "task-01-test-backpack")
        os.makedirs(trial_dir)
        result_path = os.path.join(trial_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_no_url, f)
        with open(os.path.join(trial_dir, "agent_result.json"), "w") as f:
            json.dump(agent_no_url, f)

        task_dir = os.path.join(tmpdir, "tasks", "task-01-test-backpack")
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "task.toml"), "w") as f:
            f.write(task_toml_no_urls)
        with open(os.path.join(task_dir, "instruction.md"), "w") as f:
            f.write(instruction_no_urls)

        result = verify_task(result_path, task_dir)

    assert result["agentic_verified"] is None
    assert result["null_reason"] == "no_urls"
    assert result["verification_method"] == "unverifiable"
    assert result["deliverable_url"] is None


def test_verify_task_instruction_urls_take_priority():
    """Instruction URL must be deliverable_url even when task_result has a different URL."""
    task_toml = 'version = "1.0"\n\n[metadata]\ncategory = "Shopping"\n'
    instruction_with_url = "Check https://instruction-url.com for products\n"
    result_data = {**RESULT_JSON}
    agent_with_result_url = {"task_result": "I found items at https://result-url.com"}

    with tempfile.TemporaryDirectory() as tmpdir:
        trial_dir = os.path.join(tmpdir, "task-01-test-backpack")
        os.makedirs(trial_dir)
        result_path = os.path.join(trial_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f)
        with open(os.path.join(trial_dir, "agent_result.json"), "w") as f:
            json.dump(agent_with_result_url, f)

        task_dir = os.path.join(tmpdir, "tasks", "task-01-test-backpack")
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "task.toml"), "w") as f:
            f.write(task_toml)
        with open(os.path.join(task_dir, "instruction.md"), "w") as f:
            f.write(instruction_with_url)

        captured = {}

        def mock_browser_verify(instruction, claimed_result, urls, **kwargs):
            captured["urls"] = urls
            return BROWSER_RESULT

        with patch("agentic_judge.core.verifier.browser_verify", side_effect=mock_browser_verify):
            verify_task(result_path, task_dir)

    assert captured["urls"][0] == "https://instruction-url.com"
