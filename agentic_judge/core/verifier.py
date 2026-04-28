"""Per-task verification orchestration."""

import json
import re
import time
import tomllib
from pathlib import Path
from typing import Optional

from .browser_agent import verify as browser_verify

# Matches http(s) URLs, stopping at whitespace or common terminal punctuation
URL_RE = re.compile(r'https?://[^\s\'"<>\)\]]+')


def _extract_urls(text: Optional[str]) -> list[str]:
    """Return all URLs found in text, preserving order."""
    if not text:
        return []
    return URL_RE.findall(text)


def _parse_domain(task_toml_content: str) -> str:
    """Extract category from task.toml [metadata] section.

    Returns 'Unknown' if the field is missing or the content cannot be parsed.
    """
    try:
        data = tomllib.loads(task_toml_content)
        return data.get("metadata", {}).get("category", "Unknown")
    except Exception:
        return "Unknown"


def _get_priority_urls(
    instruction_urls: list[str],
    result_urls: list[str],
    max_urls: int = 3,
) -> list[str]:
    """Collect up to max_urls unique URLs, instruction URLs first.

    Instruction URLs fill slots 1..max_urls first (ground truth).
    Result URLs fill any remaining capacity.
    """
    seen: set[str] = set()
    combined: list[str] = []
    for url in instruction_urls + result_urls:
        if url not in seen and len(combined) < max_urls:
            seen.add(url)
            combined.append(url)
    return combined


def verify_task(
    result_json_path: str,
    task_dir: str,
    model: str = "gpt-4o",
    timeout_seconds: int = 120,
) -> dict:
    """Run agentic verification for one task.

    Args:
        result_json_path: Path to result.json from the static judge (inside the task's trial dir).
        task_dir: Path to the task directory containing task.toml and instruction.md.
        model: OpenAI model for GPT-4o judgment call.
        timeout_seconds: Browser timeout budget.

    Returns:
        Verification dict ready for result_writer.write_result().
    """
    with open(result_json_path) as f:
        result = json.load(f)

    # "task" is the field name in this repo; fall back to "task_name" then parent dir name
    task_name = result.get("task", result.get("task_name", Path(result_json_path).parent.name))

    # task_result and execution_summary live in sibling agent_result.json
    task_result_text = result.get("task_result", "") or ""
    execution_summary = result.get("execution_summary", "") or ""
    agent_result_path = Path(result_json_path).parent / "agent_result.json"
    if agent_result_path.exists():
        with open(agent_result_path) as f:
            agent_data = json.load(f)
        if not task_result_text:
            task_result_text = agent_data.get("task_result", "") or ""
        if not execution_summary:
            execution_summary = agent_data.get("execution_summary", "") or ""

    # Static judge fields: flat in this repo, wrapped in "eval" in legacy format
    static_passed = result.get("passed", result.get("eval", {}).get("passed", False))
    static_score = result.get("score", result.get("eval", {}).get("details", {}).get("overall_score"))
    static_dimensions = result.get(
        "dimension_scores",
        result.get("eval", {}).get("details", {}).get("dimension_scores", {}),
    )

    # Task metadata from task.toml + instruction.md
    task_toml_path = Path(task_dir) / "task.toml"
    task_toml_raw = task_toml_path.read_text()
    category = _parse_domain(task_toml_raw)
    instruction_path = Path(task_dir) / "instruction.md"
    instruction = instruction_path.read_text() if instruction_path.exists() else ""

    instruction_urls = _extract_urls(instruction)
    result_urls = _extract_urls(task_result_text)
    urls = _get_priority_urls(instruction_urls, result_urls, max_urls=3)

    start_time = time.time()

    if not urls:
        return {
            "task_name": task_name,
            "category": category,
            "static_judge_passed": bool(static_passed),
            "static_judge_score": static_score,
            "static_judge_dimensions": static_dimensions,
            "agentic_verified": None,
            "agentic_finding": "No URLs found in instruction or task_result.",
            "null_reason": "no_urls",
            "verification_method": "unverifiable",
            "deliverable_url": None,
            "time_seconds": round(time.time() - start_time, 2),
        }

    browser_result = browser_verify(
        instruction=instruction,
        claimed_result=task_result_text,
        urls=urls,
        model=model,
        timeout_seconds=timeout_seconds,
        category=category,
        execution_summary=execution_summary,
    )

    return {
        "task_name": task_name,
        "category": category,
        "static_judge_passed": bool(static_passed),
        "static_judge_score": static_score,
        "static_judge_dimensions": static_dimensions,
        "agentic_verified": browser_result["verified"],
        "agentic_finding": browser_result["finding"],
        "null_reason": browser_result["null_reason"],
        "verification_method": browser_result["verification_method"],
        "deliverable_url": urls[0] if urls else None,
        "time_seconds": round(time.time() - start_time, 2),
    }
