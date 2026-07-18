"""
History trace loading and formatting for injection into agent prompts.

Loads a previous run's result JSON for a task and formats the relevant
parts (feedback, scores, task_result) into a prompt section.
"""

import json
from pathlib import Path
from typing import Optional


def load_previous_result(results_base: Path, run_tag: str, task_name: str) -> Optional[dict]:
    """Load a previous run's result JSON for a specific task."""
    # Try flat structure: <run_tag>/<task_name>.json
    result_path = results_base / run_tag / f"{task_name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    # Try nested structure from volume download: <run_tag>/<run_tag>/<task_name>.json
    for p in (results_base / run_tag).rglob(f"{task_name}.json"):
        return json.loads(p.read_text())

    return None


def format_history_prompt(
    previous_result: dict,
    max_result_chars: int = 3000,
    max_feedback_chars: int = 1500,
) -> str:
    """Format a previous result into a prompt section for the agent.

    Includes: score, pass/fail, dimension scores, feedback, and truncated task_result.
    Does NOT include raw execution traces (too verbose, low signal).
    """
    score = previous_result.get("score", 0)
    passed = previous_result.get("passed", False)
    feedback = previous_result.get("feedback", "")
    task_result = previous_result.get("task_result", "")
    dim_scores = previous_result.get("dimension_scores", {})
    evidence = previous_result.get("evidence_summary", "")
    dim_reasoning = previous_result.get("dimension_reasoning", {})

    # Truncate long fields
    if len(task_result) > max_result_chars:
        task_result = task_result[:max_result_chars] + f"\n... [truncated, {len(task_result)} total chars]"
    if len(feedback) > max_feedback_chars:
        feedback = feedback[:max_feedback_chars] + "\n... [truncated]"

    lines = [
        "---",
        "LEARNING FROM PREVIOUS ATTEMPT:",
        "",
        f"A previous agent attempted this exact task and scored {score}/5 ({'PASSED' if passed else 'FAILED'}).",
        "",
    ]

    # Dimension scores
    if dim_scores:
        lines.append("Dimension scores from the evaluator:")
        for dim, val in dim_scores.items():
            if val is not None:
                lines.append(f"  - {dim}: {val}/5")
        lines.append("")

    # Evaluator feedback
    if feedback:
        lines.append("Evaluator feedback:")
        lines.append(feedback)
        lines.append("")

    # Dimension reasoning
    if dim_reasoning:
        lines.append("Detailed reasoning per dimension:")
        for dim, reason in dim_reasoning.items():
            if reason:
                lines.append(f"  - {dim}: {reason}")
        lines.append("")

    # Previous answer (truncated)
    if task_result:
        lines.append("Previous agent's response (for reference):")
        lines.append(task_result)
        lines.append("")

    lines.extend([
        "USE THIS INFORMATION TO:",
        "- Avoid repeating mistakes identified in the feedback",
        "- Focus on dimensions that scored poorly",
        "- Build on approaches that worked (high-scoring dimensions)",
        "- Produce a better result than the previous attempt",
        "---",
        "",
    ])

    return "\n".join(lines)


def build_history_injection(
    results_base: Path,
    history_run_tag: str,
    task_name: str,
    max_result_chars: int = 3000,
) -> str:
    """Load previous result and format for prompt injection. Returns empty string if not found."""
    prev = load_previous_result(results_base, history_run_tag, task_name)
    if not prev:
        return ""
    return format_history_prompt(prev, max_result_chars=max_result_chars)
