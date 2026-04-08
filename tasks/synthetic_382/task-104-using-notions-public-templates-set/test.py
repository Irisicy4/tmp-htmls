"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Set up a daily habit tracker in Notion for tracking exercise, water intake, and sleep, including checkboxes and a daily summary formula.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
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


TASK_INSTRUCTION = """Using Notion’s public templates, set up a daily habit tracker for tracking exercise, water intake, and sleep. Include checkboxes for each activity and provide a daily summary formula calculating total habits completed per day."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Notion's public templates to create a daily habit tracker. The tracker must include checkboxes for exercise, water intake, and sleep, and a formula to calculate the total habits completed per day. The agent must use Notion and demonstrate proper setup of the tracker.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Notion’s public templates, set up a daily habit tracker for tracking exercise, water intake, and sleep. Include checkboxes for each activity and provide a daily summary formula calculating total habits completed per day.

## Task-Specific Constraints
- Must use Notion's public templates to create the tracker.
- The tracker must include checkboxes for exercise, water intake, and sleep.
- The tracker must include a formula to calculate the total habits completed per day.
- The output must be a properly configured Notion page.
- The agent must demonstrate navigation to and use of Notion's template library.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Notion and use its public templates?
- Does the tracker include checkboxes for exercise, water intake, and sleep?
- Is there a formula to calculate the total habits completed per day?
- Is the output a properly configured Notion page?
- Are all required elements present and functional?

### Step 2: Dimension Scoring

#### A. Tracker Completeness (0.35)
Measures whether the tracker includes all required elements: checkboxes for exercise, water intake, and sleep, and a formula for total habits.

5 — Includes all required elements, fully functional.
4 — Includes all required elements, minor issues in functionality.
3 — Missing one required element or partially functional.
2 — Missing multiple required elements or mostly non-functional.
1 — Tracker is absent or completely incorrect.

#### B. Use of Notion Templates (0.30)
Measures whether the agent successfully navigated and used Notion's public templates.

5 — Clearly demonstrates use of Notion's public templates.
4 — Likely used Notion templates, but evidence is incomplete.
3 — Evidence of Notion usage, but not templates specifically.
2 — Minimal evidence of using Notion.
1 — No evidence of using Notion.

#### C. Formula Accuracy (0.20)
Measures whether the formula for calculating total habits completed per day is correct and functional.

5 — Formula is correct and functional.
4 — Formula is mostly correct, minor issues.
3 — Formula is partially correct or incomplete.
2 — Formula is mostly incorrect or non-functional.
1 — Formula is absent or completely incorrect.

#### D. Output Organization (0.15)
Measures whether the output is well-structured and easy to understand.

5 — Output is well-organized and easy to follow.
4 — Output is mostly organized, minor issues.
3 — Output is somewhat organized but confusing in places.
2 — Output is poorly organized and hard to follow.
1 — Output is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tracker_completeness": <1-5>,
  "use_of_notion_templates": <1-5>,
  "formula_accuracy": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "tracker_completeness": "<one sentence citing specific evidence>",
    "use_of_notion_templates": "<one sentence citing specific evidence>",
    "formula_accuracy": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tracker_completeness": 0.35,
    "use_of_notion_templates": 0.30,
    "formula_accuracy": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())