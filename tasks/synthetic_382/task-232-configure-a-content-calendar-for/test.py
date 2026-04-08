"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Configure a content calendar for a creator specializing in AI tutorials using a Google Sheets template.
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


TASK_INSTRUCTION = """Configure a content calendar for a creator specializing in AI tutorials using a Google Sheets template. Include placeholder rows for 10 future videos, columns for publication dates, target keywords, and engagement metrics tracking."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a content calendar for a creator specializing in AI tutorials. The calendar must be configured in a Google Sheets template and include placeholder rows for 10 future videos, with columns for publication dates, target keywords, and engagement metrics tracking. A successful completion includes a well-structured and complete calendar that meets all specified requirements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Configure a content calendar for a creator specializing in AI tutorials using a Google Sheets template. Include placeholder rows for 10 future videos, columns for publication dates, target keywords, and engagement metrics tracking.

## Task-Specific Constraints
- Must use Google Sheets to configure the calendar.
- Must include exactly 10 placeholder rows for future videos.
- Each row must have columns for publication dates, target keywords, and engagement metrics tracking.
- The calendar must be well-structured and readable.
- The agent must demonstrate that it accessed and used the Google Sheets platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and configure the calendar as instructed?
- Are there exactly 10 placeholder rows in the calendar?
- Does each row include columns for publication dates, target keywords, and engagement metrics tracking?
- Is the calendar well-structured and readable?
- Did the agent provide evidence of using Google Sheets to complete the task?

### Step 2: Dimension Scoring

#### A. Calendar Configuration Accuracy (0.35)
Measures whether the calendar was configured correctly in Google Sheets with all required elements.

5 — Calendar is complete, with 10 rows and all required columns correctly configured.
4 — Calendar is mostly complete, with minor errors or omissions in rows or columns.
3 — Calendar is partially complete, missing some rows or columns but usable.
2 — Calendar is mostly incomplete or poorly configured.
1 — Calendar is absent or completely incorrect.

#### B. Coverage of Required Elements (0.30)
Measures whether all required elements (10 rows, publication dates, target keywords, engagement metrics) are included.

5 — All required elements are present and correctly implemented.
4 — Most required elements are present, with minor omissions.
3 — Some required elements are present, but others are missing.
2 — Few required elements are present.
1 — No required elements are present.

#### C. Detail and Specificity (0.20)
Measures the level of detail and specificity in the calendar content.

5 — Each row includes specific and meaningful placeholder data for all columns.
4 — Most rows include specific placeholder data, with minor gaps.
3 — Some rows include placeholder data, but many are generic or missing.
2 — Placeholder data is mostly generic or absent.
1 — No placeholder data is provided.

#### D. Output Structure and Readability (0.15)
Measures how well-structured and readable the calendar is.

5 — The calendar is well-organized, easy to read, and visually clear.
4 — The calendar is mostly well-organized, with minor readability issues.
3 — The calendar is somewhat organized but has noticeable readability issues.
2 — The calendar is poorly organized and difficult to read.
1 — The calendar is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "calendar_configuration_accuracy": <1-5>,
  "coverage_of_required_elements": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_readability": <1-5>,
  "dimension_reasoning": {{
    "calendar_configuration_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_elements": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_readability": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "calendar_configuration_accuracy": 0.35,
    "coverage_of_required_elements": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_readability": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())