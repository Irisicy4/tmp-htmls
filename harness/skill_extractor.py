"""
Skill extraction from task execution results.

Determines whether a completed task should be stored as a skill,
and extracts a structured markdown skill document using an LLM.
"""

import json
import os
import re
from typing import Any, Dict, Optional


GENERALIZATION_PROMPT = """\
You are evaluating whether a task and its solution represent a *generalizable* skill \
that could help an AI agent solve similar tasks in the future.

Task instruction:
{instruction}

Agent's solution summary:
{execution_summary}

Evaluation score: {score}/5

Answer with a JSON object:
{{
  "generalizable": true/false,
  "reasoning": "one sentence explaining why",
  "task_type": "short label like 'product-research' or 'data-analysis'",
  "tags": ["tag1", "tag2"]
}}

A skill is generalizable if the approach could apply to a class of similar tasks \
(e.g., "research and compare products", "fill out a web form", "write and debug code"). \
It is NOT generalizable if the solution is entirely specific to one unique scenario \
with no transferable strategy.
"""

EXTRACTION_PROMPT = """\
Extract a reusable skill document from this completed task. \
The skill should help an AI agent solve SIMILAR tasks in the future — \
generalize the strategy, do not just transcribe what happened.

Task instruction:
{instruction}

Agent's execution summary (step-by-step actions and outcomes):
{execution_summary}

Agent's final answer:
{task_result}

Evaluation score: {score}/5

IMPORTANT RULES:
1. Focus on the HIGH-LEVEL STRATEGY that led to progress. Skip retries, \
error loops, and dead ends — only distill steps that moved the task forward.
2. NEVER include specific BID values (bid1, bid2, etc.). BIDs are temporary \
element IDs that change every run. Describe elements by their visible label, \
role, or purpose instead (e.g. "click the Search button", not "dom_click(bid=bid9)").
3. Only hardcode URLs for canonical entry points (google.com, youtube.com, \
amazon.com). For site-specific pages, describe the navigation pattern.
4. If the score is low (below 3/5), be conservative: extract only the general \
approach pattern for this task type, not the specific (failed) execution steps. \
Use the Tips and Pitfalls section to document what went wrong.

Write the skill as a structured markdown document:
# Skill: <descriptive title>

## Task Pattern
When this skill applies (generalized description of the task type, not this \
specific task instance).

## Strategy
Numbered high-level steps the agent should follow.

## Key Action Steps
General tool usage patterns for this task type. No BIDs, no ephemeral values.

## Tips and Pitfalls
Common failure modes and how to avoid them, informed by what went wrong in \
the trajectory above.

Return ONLY the markdown document, no extra text.
"""


def _format_full_execution_trace(
    result: Dict[str, Any],
    max_chars_per_feedback: int = 2000,
    max_total_chars: int = 80000,
) -> str:
    """Build a detailed execution trace for judge evaluation."""
    trace = result.get("execution_trace", [])
    if not trace:
        return ""

    _SKIP_ACTION_KEYS = {"action_type", "tool_call_id"}

    lines: list[str] = []
    total_chars = 0

    for i, entry in enumerate(trace, 1):
        action = entry.get("action", {})
        feedback = entry.get("feedback", {})
        atype = action.get("action_type", "unknown")

        param_parts = []
        for k, v in action.items():
            if k in _SKIP_ACTION_KEYS:
                continue
            v_str = str(v)
            if len(v_str) > 500:
                v_str = v_str[:500] + "... [truncated]"
            param_parts.append(f"  {k}: {v_str}")

        params_block = "\n".join(param_parts) if param_parts else "  (no parameters)"

        msg = feedback.get("message", "")
        if len(msg) > max_chars_per_feedback:
            msg = msg[:max_chars_per_feedback] + f"... [truncated, {len(msg)} total chars]"

        step_text = f"Step {i}: {atype}\n{params_block}\n-> {msg}\n"

        total_chars += len(step_text)
        if total_chars > max_total_chars:
            lines.append(
                f"... [trace truncated at step {i}/{len(trace)}, "
                f"exceeded {max_total_chars} char limit]"
            )
            break

        lines.append(step_text)

    return "\n".join(lines)


def _format_execution_summary(result: Dict[str, Any], max_steps: int = 40) -> str:
    """Build a compact step-by-step summary from execution_trace."""
    if result.get("execution_summary"):
        return result["execution_summary"]

    trace = result.get("execution_trace", [])
    if not trace:
        return ""

    _SHOW_PARAMS = {
        "browser_navigate": ["url"],
        "dom_type": ["text"],
        "dom_click": ["bid"],
        "dom_hover": ["bid"],
        "dom_press": ["key"],
        "dom_scroll": ["direction", "amount"],
        "dom_query_selector": ["selector"],
        "code_execute": ["language"],
        "shell_execute": [],
        "file_write": ["path"],
        "file_read": ["path"],
        "browser_press": ["key"],
        "browser_type": ["text"],
        "task_complete": ["result"],
    }

    lines = []
    for i, entry in enumerate(trace[:max_steps], 1):
        action = entry.get("action", {})
        feedback = entry.get("feedback", {})
        atype = action.get("action_type", "?")

        show_keys = _SHOW_PARAMS.get(atype)
        if show_keys is None:
            param_str = ""
        else:
            parts = []
            for k in show_keys:
                v = action.get(k)
                if v is not None:
                    v_str = str(v)[:60].replace("\n", " ")
                    parts.append(f"{k}={v_str}")
            param_str = f"({', '.join(parts)})" if parts else ""

        msg = feedback.get("message", "")
        first_line = msg.split("\n")[0][:100]

        lines.append(f"{i}. {atype}{param_str} -> {first_line}")

    if len(trace) > max_steps:
        lines.append(f"... ({len(trace) - max_steps} more steps)")

    return "\n".join(lines)


class SkillExtractor:
    """Extracts and evaluates skills from task execution results."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        import openai
        base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def should_store(
        self,
        result: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Decide whether a task result should be stored as a skill."""
        threshold = config.get("skill_score_threshold", 3.0)

        if eval_result is None:
            return {"store": False, "reason": "no evaluation result"}

        score = (eval_result.get("details", {}) or {}).get("overall_score")
        if score is None:
            return {"store": False, "reason": "no score in evaluation"}

        if float(score) < threshold:
            return {
                "store": False,
                "reason": f"score {score} below threshold {threshold}",
            }

        instruction = result.get("instruction", "")
        execution_summary = _format_execution_summary(result)

        prompt = GENERALIZATION_PROMPT.format(
            instruction=instruction,
            execution_summary=execution_summary[:2000],
            score=score,
        )

        try:
            content = self._call_llm(prompt, max_tokens=300)
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if parsed.get("generalizable"):
                    return {
                        "store": True,
                        "reason": parsed.get("reasoning", "LLM judged generalizable"),
                        "task_type": parsed.get("task_type", ""),
                        "tags": parsed.get("tags", []),
                    }
                else:
                    return {
                        "store": False,
                        "reason": parsed.get("reasoning", "LLM judged not generalizable"),
                    }
        except Exception as e:
            return {"store": False, "reason": f"LLM call failed: {e}"}

        return {"store": False, "reason": "could not parse LLM response"}

    def extract_skill(
        self,
        task: Dict[str, Any],
        result: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a structured markdown skill document from a task result."""
        instruction = task.get("instruction", result.get("instruction", ""))
        execution_summary = _format_execution_summary(result)
        task_result = result.get("task_result", "")
        score = (eval_result.get("details", {}) or {}).get("overall_score", "?") if eval_result else "?"

        prompt = EXTRACTION_PROMPT.format(
            instruction=instruction,
            execution_summary=execution_summary[:3000],
            task_result=task_result[:2000],
            score=score,
        )

        return self._call_llm(prompt, max_tokens=1500)
