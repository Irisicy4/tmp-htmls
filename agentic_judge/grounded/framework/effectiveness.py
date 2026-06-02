"""Effectiveness judge — LLM-as-judge over the soft rubric dimensions.

This preserves the *existing* gpt-4o-style anchor scoring that the v1 tasks
already define, but injects the grounded summary JSON + hard-constraint
report + faithfulness report into the prompt context, so the LLM judge can
reference machine-verified facts rather than re-extracting them from the
agent's prose.

Soft dimensions remain task-specific (the task-spec module supplies the
prompt + dimension weights).  This module just wraps the call.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any


_SYSTEM_PROMPT = """You are an expert evaluator scoring an AI agent's
work on a research/web task.  Compared to a basic LLM judge, you have
three extra signals:

  1. The agent's structured summary JSON (already extracted).
  2. A hard-constraint report (machine-checkable predicates against the JSON).
  3. A faithfulness report (URL fetches verifying the agent's claims).

Use these to ground your scoring.  When a hard constraint failed, the
relevant dimension cannot be 5.  When a URL fetch failed, the
"specificity" / "data accuracy" dimension cannot be 5.  Score honestly."""


def call_llm_judge(*,
                    task_instruction: str,
                    agent_response: str,
                    summary_json: dict | None,
                    hard_constraint_report: list[dict],
                    faithfulness_report: dict,
                    dimensions: list[str],
                    dimension_weights: dict[str, float],
                    task_specific_rubric: str,
                    model: str | None = None,
                    max_retries: int = 4) -> dict:
    """Single-call wrapper around the existing gpt-4o judge contract.
    Returns the parsed dict from `<Answer>` tags, or {"error": ...}."""
    model = model or os.environ.get("EVOLVEBENCH_JUDGE_MODEL", "gpt-4o")
    try:
        import openai
    except ImportError:
        return {"error": "openai library not available"}

    user = _build_user_prompt(
        task_instruction=task_instruction,
        agent_response=agent_response,
        summary_json=summary_json,
        hard_constraint_report=hard_constraint_report,
        faithfulness_report=faithfulness_report,
        dimensions=dimensions,
        dimension_weights=dimension_weights,
        task_specific_rubric=task_specific_rubric,
    )
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                            base_url=os.environ.get("OPENAI_BASE_URL") or None)
    last_err = ""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=1500,
            )
            return _parse_answer_tag(completion.choices[0].message.content or "")
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 ** attempt)
    return {"error": last_err}


def _build_user_prompt(*,
                        task_instruction: str,
                        agent_response: str,
                        summary_json: dict | None,
                        hard_constraint_report: list[dict],
                        faithfulness_report: dict,
                        dimensions: list[str],
                        dimension_weights: dict[str, float],
                        task_specific_rubric: str) -> str:
    summary_dump = json.dumps(summary_json, indent=2) if summary_json else "(no summary JSON parsed)"
    hard_dump = json.dumps(hard_constraint_report, indent=2)
    faith_dump = json.dumps({k: v for k, v in faithfulness_report.items() if k != "details"}, indent=2)
    faith_details = json.dumps(faithfulness_report.get("details", [])[:10], indent=2)
    dim_lines = "\n".join(f"- {d} (weight {dimension_weights[d]:.2f})" for d in dimensions)
    return f"""## Task Instruction
{task_instruction}

## Agent Final Response (raw, truncated)
{(agent_response or "")[:8000]}

## Agent Summary JSON (already extracted)
{summary_dump}

## Hard-constraint report
{hard_dump}

## Faithfulness report (URL fetches)
Headline: {faith_dump}
Details: {faith_details}

## Task-specific rubric (anchors)
{task_specific_rubric}

## Dimensions to score (1-5 each)
{dim_lines}

## Instructions
Score each dimension 1-5.  Use the hard-constraint and faithfulness
reports as ground truth: any dimension covered by a FAILED hard
constraint or a failed faithfulness check cannot score 5.  Conversely,
all-PASS does not automatically mean 5 — the soft rubric anchors still
apply (presentation, comparison depth, etc.).

Return ONLY:

<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  {", ".join(f'"{d}": <1-5>' for d in dimensions)},
  "dimension_reasoning": {{
    {", ".join(f'"{d}": "<one sentence citing evidence>"' for d in dimensions)}
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true|false based on overall_score >= 3.0>
}}
</Answer>
"""


def _parse_answer_tag(text: str) -> dict:
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.S | re.I)
    raw = (m.group(1) if m else text).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"json parse: {e}", "raw": raw[:500]}
