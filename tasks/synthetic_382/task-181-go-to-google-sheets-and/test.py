"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Set up an Agile sprint tracking sheet in Google Sheets with specific columns, dropdowns, and sample data.
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


TASK_INSTRUCTION = """Go to Google Sheets and set up an Agile sprint tracking sheet. Create columns for 'Task Name', 'Status', 'Assignee', 'Priority', and 'Estimated Hours', and add 10 sample rows with realistic task data. Format the 'Status' column as a dropdown with values 'To Do', 'In Progress', and 'Done'."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create an Agile sprint tracking sheet in Google Sheets. The sheet must include specific columns ('Task Name', 'Status', 'Assignee', 'Priority', 'Estimated Hours'), 10 rows of realistic sample data, and a dropdown menu in the 'Status' column with values ('To Do', 'In Progress', 'Done').

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Google Sheets and set up an Agile sprint tracking sheet. Create columns for 'Task Name', 'Status', 'Assignee', 'Priority', and 'Estimated Hours', and add 10 sample rows with realistic task data. Format the 'Status' column as a dropdown with values 'To Do', 'In Progress', and 'Done'.

## Task-Specific Constraints
- Must create a Google Sheet with the specified column names.
- Must populate the sheet with 10 rows of realistic sample data.
- Must format the 'Status' column as a dropdown with the specified values.
- Must ensure the data is relevant to Agile sprint tracking.
- Must use Google Sheets as the platform for the task.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent create a Google Sheet?
- Are the specified columns ('Task Name', 'Status', 'Assignee', 'Priority', 'Estimated Hours') present?
- Does the sheet include 10 rows of realistic sample data?
- Is the 'Status' column formatted as a dropdown with the specified values ('To Do', 'In Progress', 'Done')?
- Is the data relevant to Agile sprint tracking?

### Step 2: Dimension Scoring

#### A. Column and Dropdown Accuracy (0.35)
Measures whether the required columns and dropdown formatting are present and correct.

5 — All required columns are present, and the 'Status' column is correctly formatted as a dropdown with all specified values.
4 — All required columns are present, but the dropdown formatting is partially correct (e.g., missing values).
3 — Most required columns are present, but dropdown formatting is missing or incorrect.
2 — Few required columns are present, and dropdown formatting is missing or incorrect.
1 — No required columns or dropdown formatting is present.

#### B. Data Completeness (0.30)
Measures whether the sheet contains 10 rows of realistic sample data.

5 — Contains 10 rows of realistic and relevant sample data.
4 — Contains 8-9 rows of realistic and relevant sample data.
3 — Contains 6-7 rows of sample data, but some may be unrealistic or irrelevant.
2 — Contains fewer than 6 rows, or most data is unrealistic or irrelevant.
1 — Contains no sample data or completely irrelevant data.

#### C. Relevance to Agile Tracking (0.20)
Measures whether the data and structure align with Agile sprint tracking.

5 — All data is highly relevant to Agile sprint tracking (e.g., tasks, priorities, assignees).
4 — Most data is relevant, but some minor details are off.
3 — Some data is relevant, but there are noticeable gaps or irrelevant entries.
2 — Little data is relevant to Agile sprint tracking.
1 — No data is relevant to Agile sprint tracking.

#### D. Platform Usage and Organization (0.15)
Measures whether the agent used Google Sheets and structured the sheet properly.

5 — Used Google Sheets and structured the sheet professionally (e.g., proper alignment, formatting).
4 — Used Google Sheets but with minor structural issues (e.g., inconsistent formatting).
3 — Used Google Sheets but with noticeable structural issues (e.g., misaligned columns).
2 — Attempted to use Google Sheets but failed to structure the sheet properly.
1 — Did not use Google Sheets or structure the sheet.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "column_and_dropdown_accuracy": <1-5>,
  "data_completeness": <1-5>,
  "relevance_to_agile_tracking": <1-5>,
  "platform_usage_and_organization": <1-5>,
  "dimension_reasoning": {{
    "column_and_dropdown_accuracy": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "relevance_to_agile_tracking": "<one sentence citing specific evidence>",
    "platform_usage_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "column_and_dropdown_accuracy": 0.35,
    "data_completeness": 0.30,
    "relevance_to_agile_tracking": 0.20,
    "platform_usage_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())