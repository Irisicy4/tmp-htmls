"""
LLM-as-judge evaluator for EvolveBench task-49-go-to-demo-pocketbase-io-reset.

Category: System Configuration
Task: Go to https://demo.pocketbase.io and reset all settings to their default values. Navigate through all available settings menus and restore defaults wherever the option exists.
"""

import os, json, re

TASK_INSTRUCTION = """Go to https://demo.pocketbase.io and reset all settings to their default values. Navigate through all available settings menus and restore defaults wherever the option exists."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to navigate the PocketBase demo interface and accurately reset all settings to their default values. This includes identifying all settings menus, locating default restoration options, and ensuring no settings are left unchanged. The evaluation focuses on completeness, accuracy, and adherence to the task requirements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Access the PocketBase demo interface at https://demo.pocketbase.io.
- Navigate through all available settings menus systematically.
- Restore settings to their default values wherever the option exists.
- Ensure no settings are skipped or left unchanged.
- Avoid making any changes beyond restoring defaults.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access the PocketBase demo interface successfully?
- Did the agent navigate through all available settings menus?
- Did the agent restore settings to their default values wherever the option existed?
- Were all settings menus addressed without omissions?
- Did the agent avoid making changes beyond restoring defaults?

### Step 2: Dimension Scoring

#### A. Navigation Completeness
Measures whether the agent navigated through all settings menus.

5 — The agent navigated through all available settings menus without missing any.
4 — The agent navigated through most settings menus, missing only one or two.
3 — The agent navigated through some settings menus but missed several.
2 — The agent navigated through very few settings menus.
1 — The agent failed to navigate through any settings menus.

#### B. Default Restoration Accuracy
Measures the accuracy of restoring settings to their default values.

5 — The agent restored all settings to their default values accurately.
4 — The agent restored most settings to their default values, with minor inaccuracies.
3 — The agent restored some settings to their default values but made notable errors.
2 — The agent restored very few settings to their default values and made significant errors.
1 — The agent failed to restore any settings to their default values.

#### C. Task Adherence
Measures whether the agent avoided making changes beyond restoring defaults.

5 — The agent strictly adhered to the task and made no changes beyond restoring defaults.
4 — The agent mostly adhered to the task but made one or two minor changes beyond restoring defaults.
3 — The agent somewhat adhered to the task but made several changes beyond restoring defaults.
2 — The agent poorly adhered to the task and made many changes beyond restoring defaults.
1 — The agent completely disregarded the task and made numerous unrelated changes.

#### D. Efficiency
Measures the speed and organization of the agent's actions.

5 — The agent completed the task quickly and systematically without unnecessary delays.
4 — The agent completed the task efficiently but with minor delays or disorganization.
3 — The agent completed the task with moderate delays or disorganization.
2 — The agent completed the task with significant delays or disorganization.
1 — The agent failed to complete the task or was extremely slow and disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "navigation_completeness": <1-5>,
  "default_restoration_accuracy": <1-5>,
  "task_adherence": <1-5>,
  "efficiency": <1-5>,
  "dimension_reasoning": {{
    "navigation_completeness": "<one sentence citing specific evidence>",
    "default_restoration_accuracy": "<one sentence citing specific evidence>",
    "task_adherence": "<one sentence citing specific evidence>",
    "efficiency": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "navigation_completeness": 0.3,
    "default_restoration_accuracy": 0.3,
    "task_adherence":      0.2,
    "efficiency":          0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                           DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }