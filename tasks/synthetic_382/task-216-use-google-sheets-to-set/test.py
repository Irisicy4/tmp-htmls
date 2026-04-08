"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Set up a content calendar template for a YouTuber's weekly video schedule using Google Sheets.
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


TASK_INSTRUCTION = """Use Google Sheets to set up a content calendar template for a YouTuber's weekly video schedule. Include columns for video title, script completion status, filming date, editing date, and publishing date. Fill in placeholder data for one week and share the view-only link."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Google Sheets to create a content calendar template for a YouTuber's weekly video schedule. The calendar must include columns for video title, script completion status, filming date, editing date, and publishing date. Placeholder data for one week must be filled in, and the agent must share the view-only link to the sheet.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to set up a content calendar template for a YouTuber's weekly video schedule. Include columns for video title, script completion status, filming date, editing date, and publishing date. Fill in placeholder data for one week and share the view-only link.

## Task-Specific Constraints
- Must create a Google Sheet with the specified columns.
- Placeholder data must cover seven days (one week).
- The view-only link to the sheet must be shared in the response.
- The sheet must be well-organized and formatted for readability.
- The agent must use Google Sheets as the platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Google Sheets to create the content calendar?
- Are the required columns (video title, script completion status, filming date, editing date, publishing date) present in the sheet?
- Does the placeholder data cover seven days (one week)?
- Is the view-only link to the sheet included in the response?
- Is the sheet formatted and organized for readability?

### Step 2: Dimension Scoring

#### A. Template Accuracy (0.35)
Measures whether the content calendar template includes all required columns and placeholder data.

5 — All required columns are present, and placeholder data covers seven days.
4 — All required columns are present, but placeholder data is incomplete or partially incorrect.
3 — Some required columns are missing, but the template is partially usable.
2 — Most required columns are missing, and the template is barely usable.
1 — No usable template is provided.

#### B. Coverage of Requirements (0.30)
Measures whether the agent fulfilled all task-specific constraints.

5 — All constraints are fully satisfied (platform used, link shared, data complete).
4 — Most constraints are satisfied, but minor elements are missing.
3 — Some constraints are satisfied, but major elements are missing.
2 — Few constraints are satisfied, and the response is mostly incomplete.
1 — No constraints are satisfied.

#### C. Data Specificity (0.20)
Measures the quality and specificity of the placeholder data provided.

5 — Placeholder data is detailed, realistic, and covers all seven days.
4 — Placeholder data is mostly detailed but has minor issues.
3 — Placeholder data is generic or incomplete but usable.
2 — Placeholder data is sparse and barely usable.
1 — No placeholder data is provided.

#### D. Formatting and Organization (0.15)
Measures the readability and organization of the sheet.

5 — The sheet is well-formatted, visually clear, and easy to read.
4 — The sheet is mostly well-formatted but has minor readability issues.
3 — The sheet is somewhat organized but lacks clarity.
2 — The sheet is poorly formatted and hard to read.
1 — The sheet is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "data_specificity": <1-5>,
  "formatting_and_organization": <1-5>,
  "dimension_reasoning": {{
    "template_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "formatting_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "data_specificity": 0.20,
    "formatting_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())