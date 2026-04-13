"""
Skill extraction from task execution results.

Three complementary mechanisms:

1. Per-task REFLECTION (Reflexion-style): for each task, generate a targeted
   feedback summary from Phase 1's evaluation — what scored well, what to
   improve, with the evaluator's exact dimension scores and reasoning.
   This is the primary signal for per-task improvement in Phase 2.

2. Cross-task workflow induction (AWM-style): batch all Phase 1 results,
   find common sub-routines across tasks, produce abstract reusable workflows
   with placeholder variables. Supplementary to reflections.

3. Per-task skill extraction (legacy): one generalized skill per task.
   Kept as fallback but reflections are preferred for same-task reruns.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-task REFLECTION generation (Reflexion-style)
# ---------------------------------------------------------------------------

def generate_reflection(
    task_name: str,
    eval_result: Dict[str, Any],
    instruction: str = "",
    task_result: str = "",
) -> Dict[str, Any]:
    """Generate a structured reflection from a Phase 1 evaluation result.

    This is the core per-task improvement mechanism. Instead of extracting a
    generic "skill", we produce a targeted reflection that tells the Phase 2
    agent exactly what the evaluator found and what needs to improve.

    Returns a dict with:
        task_name, score, passed, dimensions (sorted by score),
        strengths, weaknesses, improvement_instructions, reflection_text
    """
    details = (eval_result.get("details", {}) or {})
    score = details.get("overall_score", 0)
    dim_scores = details.get("dimension_scores", {}) or {}
    dim_reasoning = details.get("dimension_reasoning", {}) or {}
    evidence = details.get("evidence_summary", "")
    passed = eval_result.get("passed", False)

    # Sort dimensions by score to identify strengths and weaknesses
    sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1])

    # Identify strengths (scored 4-5) and weaknesses (scored 1-3)
    strengths = [(dim, val, dim_reasoning.get(dim, "")) for dim, val in sorted_dims if val >= 4]
    weaknesses = [(dim, val, dim_reasoning.get(dim, "")) for dim, val in sorted_dims if val <= 3]

    # Build the reflection text that will be injected into Phase 2
    lines = []
    lines.append("## Previous Attempt Feedback")
    lines.append(f"You previously attempted this exact task and scored **{score}/5**.")
    lines.append(f"The evaluator assessed your output on these dimensions:\n")

    for dim, val in sorted_dims:
        marker = "!!" if val <= 2 else (">" if val <= 3 else "OK")
        reasoning = dim_reasoning.get(dim, "")
        lines.append(f"- **{dim}: {val}/5** [{marker}]  {reasoning}")

    if evidence:
        lines.append(f"\n**Evaluator evidence:** {evidence}")

    if weaknesses:
        lines.append("\n## What You MUST Improve")
        lines.append("Focus your effort here — these dimensions hurt your score:\n")
        for dim, val, reasoning in weaknesses:
            lines.append(f"- **{dim}** (scored {val}/5): {reasoning}")

    if strengths:
        lines.append("\n## What Worked Well (maintain this)")
        for dim, val, reasoning in strengths:
            lines.append(f"- **{dim}** (scored {val}/5): {reasoning}")

    lines.append("\n## Key Improvement Instructions")
    # Generate specific improvement instructions based on weak dimensions
    for dim, val, reasoning in weaknesses:
        if val <= 2:
            lines.append(f"- CRITICAL — **{dim}**: This dimension scored {val}/5. "
                        f"The evaluator said: \"{reasoning}\" "
                        f"You must fundamentally change your approach for this dimension.")
        else:
            lines.append(f"- IMPROVE — **{dim}**: Scored {val}/5. {reasoning}")

    reflection_text = "\n".join(lines)

    return {
        "task_name": task_name,
        "score": score,
        "passed": passed,
        "dimension_scores": dim_scores,
        "strengths": [(d, v) for d, v, _ in strengths],
        "weaknesses": [(d, v) for d, v, _ in weaknesses],
        "evidence": evidence,
        "reflection_text": reflection_text,
    }


# ---------------------------------------------------------------------------
# Quality gating — dimension-level filtering (AWM insight: only learn from
# genuinely successful runs, not marginal passes)
# ---------------------------------------------------------------------------

def check_quality_gate(
    eval_result: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether a task result passes quality gating for skill extraction.

    Goes beyond a single score threshold: checks dimension-level scores to
    reject tasks where key dimensions scored poorly (even if overall passed).
    """
    if eval_result is None:
        return {"pass": False, "reason": "no evaluation result"}

    details = eval_result.get("details", {}) or {}
    score = details.get("overall_score")
    if score is None:
        return {"pass": False, "reason": "no score in evaluation"}

    # Overall threshold (raised from 3.0 to 3.5 by default)
    threshold = config.get("skill_score_threshold", 3.5)
    if float(score) < threshold:
        return {"pass": False, "reason": f"score {score} < threshold {threshold}"}

    # Dimension-level gating: reject if ANY dimension scored <= 2
    dim_floor = config.get("skill_dimension_floor", 2)
    dimension_scores = details.get("dimension_scores", {}) or {}
    weak_dims = {
        dim: val for dim, val in dimension_scores.items()
        if val is not None and float(val) <= dim_floor
    }
    if weak_dims:
        return {
            "pass": False,
            "reason": f"weak dimensions {weak_dims} <= floor {dim_floor}",
            "weak_dimensions": weak_dims,
        }

    return {
        "pass": True,
        "reason": f"score {score} >= {threshold}, all dimensions > {dim_floor}",
        "score": score,
        "dimension_scores": dimension_scores,
    }


# ---------------------------------------------------------------------------
# Cross-task workflow induction (AWM-style)
# ---------------------------------------------------------------------------

CROSS_TASK_INDUCTION_PROMPT = """\
You are extracting reusable workflow sub-routines from a batch of completed \
agent tasks. Your goal is to find COMMON PATTERNS that appear across multiple \
tasks, abstract them into reusable workflows with placeholder variables, and \
note what evaluators specifically check for.

This is inspired by the Agent Workflow Memory approach: extract frequently \
reused sub-routines (NOT full task strategies), use abstract placeholders \
instead of concrete values, and focus on patterns that transfer across tasks.

Here are the completed task results:

{task_summaries}

INSTRUCTIONS:
1. Identify common SUB-ROUTINES that appear across 2+ tasks. Examples:
   - "Search for items on a specific platform with constraints"
   - "Save structured output in required format"
   - "Compare alternatives against a reference"
   Do NOT create one workflow per task. Find SHARED patterns.

2. For each workflow, use ABSTRACT PLACEHOLDERS instead of concrete values:
   - Use {{platform}} not "TripAdvisor" or "Xiaohongshu"
   - Use {{search-query}} not "NUS interview questions"
   - Use {{budget-constraint}} not "$75"
   - Use {{output-format}} not "result.txt"

3. For each workflow, include the EVALUATOR CRITERIA that judges check for \
   this type of sub-routine. This is critical — the agent needs to know what \
   it will be scored on.

4. Include FAILURE PATTERNS from tasks that scored poorly on specific \
   dimensions. These are as valuable as success patterns.

Return a series of workflows separated by "===WORKFLOW===" markers.
Each workflow should follow this format:

# Workflow: <short descriptive title with no specific task details>

## Pattern
When this workflow applies (generalized, using placeholders).

## Sub-routine Steps
1. Step one (abstract, reusable)
2. Step two
...

## Evaluator Criteria
What the judge specifically checks for this type of sub-routine, and the \
relative importance of each dimension. Flag dimensions that are commonly weak.

## Placeholders
- {{variable-name}}: description of what this represents

## Failure Patterns
What NOT to do — common mistakes observed across tasks that led to score \
deductions. Be specific about what evaluators penalize.

Return ONLY the workflows, no extra text.
"""


def _summarize_task_for_induction(
    task_name: str,
    result: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]],
    quality_gate: Dict[str, Any],
) -> str:
    """Build a compact summary of one task for cross-task induction."""
    instruction = result.get("instruction", "")[:500]
    task_result = result.get("task_result", "")[:800]

    details = (eval_result.get("details", {}) or {}) if eval_result else {}
    score = details.get("overall_score", "?")
    dim_scores = details.get("dimension_scores", {}) or {}
    dim_reasoning = details.get("dimension_reasoning", {}) or {}
    evidence = details.get("evidence_summary", "")[:400]
    passed = quality_gate.get("pass", False)

    lines = [
        f"### Task: {task_name}  (score: {score}/5, quality_gate: {'PASS' if passed else 'FAIL'})",
        f"**Instruction (truncated):** {instruction}",
        f"**Dimension scores:** {json.dumps(dim_scores)}",
    ]

    # Include dimension reasoning — this tells the LLM what the judge cares about
    if dim_reasoning:
        lines.append("**Dimension reasoning:**")
        for dim, reasoning in dim_reasoning.items():
            lines.append(f"  - {dim}: {str(reasoning)[:200]}")

    if evidence:
        lines.append(f"**Evidence:** {evidence}")

    # Include weak dimensions from quality gate
    weak = quality_gate.get("weak_dimensions", {})
    if weak:
        lines.append(f"**WEAK DIMENSIONS (caused quality gate failure):** {weak}")

    lines.append(f"**Agent output (truncated):** {task_result}")
    lines.append("")

    return "\n".join(lines)


def induce_workflows_cross_task(
    task_results: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    max_workflows: int = 10,
) -> List[str]:
    """AWM-style cross-task workflow induction.

    Takes ALL phase 1 results (both passed and failed), finds common
    sub-routines, and returns a list of abstract workflow markdown documents.

    Args:
        task_results: list of dicts, each with keys:
            task_name, result, eval_result, quality_gate
        model: LLM model for induction
        max_workflows: maximum number of workflows to extract

    Returns:
        List of workflow markdown strings
    """
    if not task_results:
        return []

    # Build summaries for all tasks (include failures — they inform pitfalls)
    summaries = []
    for tr in task_results:
        summary = _summarize_task_for_induction(
            tr["task_name"],
            tr["result"],
            tr.get("eval_result"),
            tr.get("quality_gate", {}),
        )
        summaries.append(summary)

    combined_summaries = "\n---\n\n".join(summaries)

    # Truncate if too long for context
    if len(combined_summaries) > 60000:
        combined_summaries = combined_summaries[:60000] + "\n... [truncated]"

    prompt = CROSS_TASK_INDUCTION_PROMPT.format(
        task_summaries=combined_summaries,
    )

    import openai
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"), base_url=base_url
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0,
    )
    content = response.choices[0].message.content.strip()

    # Split on the workflow separator
    raw_workflows = re.split(r"===WORKFLOW===", content)
    workflows = [w.strip() for w in raw_workflows if w.strip()]

    # Validate: each workflow should have a title
    valid = []
    for w in workflows[:max_workflows]:
        if "# Workflow:" in w or "# Workflow " in w:
            valid.append(w)
        elif w.startswith("#"):
            valid.append(w)

    return valid


# ---------------------------------------------------------------------------
# Per-task extraction (updated with AWM insights)
# ---------------------------------------------------------------------------

GENERALIZATION_PROMPT = """\
You are evaluating whether a task and its solution represent a *generalizable* skill \
that could help an AI agent solve similar tasks in the future.

Task instruction:
{instruction}

Agent's solution summary:
{execution_summary}

Evaluation score: {score}/5
Dimension scores: {dimension_scores}

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
Extract a reusable workflow sub-routine from this completed task. \
The workflow should help an AI agent solve SIMILAR tasks in the future — \
generalize the strategy into an ABSTRACT, REUSABLE pattern.

Task instruction:
{instruction}

Agent's execution summary (step-by-step actions and outcomes):
{execution_summary}

Agent's final answer:
{task_result}

Evaluation score: {score}/5

Evaluator feedback:
{eval_feedback}

IMPORTANT RULES:
1. Use ABSTRACT PLACEHOLDERS instead of concrete values:
   - Use {{platform}} not specific website names
   - Use {{search-query}} not the actual query used
   - Use {{constraint}} not specific numbers or requirements
   This reduces bias and makes the workflow transferable.

2. Include an EVALUATOR CRITERIA section listing the specific dimensions \
   the judge checks, their scores from this run, and which ones are critical. \
   The agent MUST know what it will be evaluated on.

3. Focus on SUB-ROUTINE level strategy, not full task reconstruction. \
   Extract the reusable pattern, not a transcript of what happened.

4. NEVER include specific BID values. Describe elements by their visible label, \
   role, or purpose instead.

5. Only hardcode URLs for canonical entry points (google.com, youtube.com). \
   For site-specific pages, describe the navigation pattern.

6. If the score is low (below 3.5/5), be conservative: extract only the general \
   approach pattern, and use the Failure Patterns section to document what went wrong.

7. Include a FAILURE PATTERNS section documenting what the evaluator penalized \
   and what to avoid. This is as important as the success strategy.

Write the workflow as a structured markdown document:
# Workflow: <descriptive title using placeholders, not specific task details>

## Pattern
When this workflow applies (generalized, with placeholders).

## Sub-routine Steps
Numbered high-level steps using {{placeholder}} variables.

## Evaluator Criteria
The specific dimensions the judge checks, which are critical, \
and how they scored in this run. Format:
- dimension_name: score/5 — what the judge checks for this

## Placeholders
- {{variable}}: what this represents

## Failure Patterns
What NOT to do, informed by evaluator feedback. Be specific.

Return ONLY the markdown document, no extra text.
"""


def _format_eval_feedback(eval_result: Optional[Dict[str, Any]]) -> str:
    """Format evaluation feedback for the extraction prompt."""
    if not eval_result:
        return "(no evaluator feedback available)"

    parts = []
    feedback = eval_result.get("feedback", "")
    if feedback:
        parts.append(feedback)

    details = eval_result.get("details", {}) or {}

    dimension_scores = details.get("dimension_scores", {}) or {}
    if dimension_scores:
        parts.append("Dimension scores: " + ", ".join(
            f"{dim}={val}/5" for dim, val in dimension_scores.items()
        ))

    dimension_reasoning = details.get("dimension_reasoning", {}) or {}
    if dimension_reasoning:
        parts.append("Dimension reasoning:")
        for dim, reasoning in dimension_reasoning.items():
            parts.append(f"  {dim}: {str(reasoning)[:200]}")

    evidence = details.get("evidence_summary", "")
    if evidence:
        parts.append(f"Evidence: {evidence[:500]}")

    return "\n".join(parts) if parts else "(no evaluator feedback available)"


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
        """Decide whether a task result should be stored as a skill.

        Now uses dimension-level quality gating in addition to overall score.
        """
        gate = check_quality_gate(eval_result, config)
        if not gate["pass"]:
            return {"store": False, "reason": gate["reason"]}

        instruction = result.get("instruction", "")
        execution_summary = _format_execution_summary(result)
        details = (eval_result.get("details", {}) or {}) if eval_result else {}
        score = details.get("overall_score", "?")
        dimension_scores = details.get("dimension_scores", {}) or {}

        prompt = GENERALIZATION_PROMPT.format(
            instruction=instruction,
            execution_summary=execution_summary[:2000],
            score=score,
            dimension_scores=json.dumps(dimension_scores),
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
        """Generate a structured markdown workflow document from a task result.

        Updated to produce AWM-style abstract workflows with placeholders
        and evaluator criteria.
        """
        instruction = task.get("instruction", result.get("instruction", ""))
        execution_summary = _format_execution_summary(result)
        task_result = result.get("task_result", "")
        details = (eval_result.get("details", {}) or {}) if eval_result else {}
        score = details.get("overall_score", "?")

        eval_feedback = _format_eval_feedback(eval_result)

        prompt = EXTRACTION_PROMPT.format(
            instruction=instruction,
            execution_summary=execution_summary[:3000],
            task_result=task_result[:2000],
            score=score,
            eval_feedback=eval_feedback,
        )

        return self._call_llm(prompt, max_tokens=1500)
