"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Configure a content calendar for a weekly podcast in Google Sheets with specified columns and sample entries.
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


TASK_INSTRUCTION = """Open Google Sheets and configure a content calendar for a weekly podcast. Create columns for episode titles, recording dates, publication dates, guest names, and promotion status. Add three sample entries to demonstrate its functionality."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to open Google Sheets and create a content calendar for a weekly podcast. The calendar must include columns for episode titles, recording dates, publication dates, guest names, and promotion status. Additionally, the agent must add three sample entries to demonstrate the calendar's functionality.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets and configure a content calendar for a weekly podcast. Create columns for episode titles, recording dates, publication dates, guest names, and promotion status. Add three sample entries to demonstrate its functionality.

## Task-Specific Constraints
- Must open Google Sheets and create a new spreadsheet.
- Must include columns for episode titles, recording dates, publication dates, guest names, and promotion status.
- Must add three sample entries in the calendar.
- The sample entries must demonstrate realistic data for a weekly podcast.
- The calendar must be organized and visually clear.
- The agent's response must describe the calendar structure and sample entries.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent open Google Sheets and create a new spreadsheet?
- Are all required columns (episode titles, recording dates, publication dates, guest names, promotion status) present?
- Are there three sample entries in the calendar, and do they contain realistic data?
- Is the calendar organized and visually clear in the description provided?
- Does the agent's response describe the calendar structure and sample entries accurately?

### Step 2: Dimension Scoring

#### A. Calendar Structure Accuracy (0.35)
Measures whether the calendar includes all required columns and is organized correctly.

5 — Includes all required columns with correct labels and clear organization.
4 — Includes all required columns but organization is slightly unclear.
3 — Includes most required columns with minor omissions or errors.
2 — Includes few required columns and lacks organization.
1 — No attempt or completely incorrect.

#### B. Sample Entry Completeness (0.30)
Measures whether the three sample entries are present and contain realistic data.

5 — All three entries are present and contain realistic, detailed data.
4 — All three entries are present but lack some detail or realism.
3 — At least two entries are present with partial data.
2 — Only one entry is present or entries are unrealistic.
1 — No entries provided.

#### C. Execution Trace Verification (0.20)
Measures whether the agent's tool-call trace confirms the use of Google Sheets.

5 — Tool-call trace confirms Google Sheets was opened and used correctly.
4 — Tool-call trace confirms Google Sheets was opened but usage is unclear.
3 — Tool-call trace partially confirms Google Sheets usage.
2 — Tool-call trace shows minimal or incorrect usage of Google Sheets.
1 — No evidence of Google Sheets usage.

#### D. Response Clarity and Detail (0.15)
Measures the clarity and detail of the agent's response describing the calendar and entries.

5 — Response is clear, detailed, and accurately describes the calendar and entries.
4 — Response is clear but lacks some detail or minor inaccuracies.
3 — Response is partially clear with noticeable omissions or errors.
2 — Response is unclear or mostly incorrect.
1 — Response is absent or completely wrong.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "calendar_structure_accuracy": <1-5>,
  "sample_entry_completeness": <1-5>,
  "execution_trace_verification": <1-5>,
  "response_clarity_and_detail": <1-5>,
  "dimension_reasoning": {{
    "calendar_structure_accuracy": "<one sentence citing specific evidence>",
    "sample_entry_completeness": "<one sentence citing specific evidence>",
    "execution_trace_verification": "<one sentence citing specific evidence>",
    "response_clarity_and_detail": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "calendar_structure_accuracy": 0.35,
    "sample_entry_completeness": 0.30,
    "execution_trace_verification": 0.20,
    "response_clarity_and_detail": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())