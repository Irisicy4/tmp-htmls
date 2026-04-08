"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a content calendar for January 2024 for a fictional Instagram influencer using Google Sheets.
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


TASK_INSTRUCTION = """Use Google Sheets to set up a content calendar for January 2024 for a fictional Instagram influencer. Include columns for Post Date, Content Theme, Platform (Instagram Stories or Feed), Target Audience, and Primary Hashtags. Structure the calendar appropriately and share the final sheet as a viewable link."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a content calendar for January 2024 for a fictional Instagram influencer using Google Sheets. The calendar must include specific columns: Post Date, Content Theme, Platform (Instagram Stories or Feed), Target Audience, and Primary Hashtags. The agent must structure the calendar appropriately and provide a viewable link to the final sheet.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to set up a content calendar for January 2024 for a fictional Instagram influencer. Include columns for Post Date, Content Theme, Platform (Instagram Stories or Feed), Target Audience, and Primary Hashtags. Structure the calendar appropriately and share the final sheet as a viewable link.

## Task-Specific Constraints
- Must use Google Sheets to create the calendar.
- Must include all required columns: Post Date, Content Theme, Platform, Target Audience, Primary Hashtags.
- Must structure the calendar in a clear and organized format.
- Must provide a viewable link to the final sheet.
- Must include at least 10 entries in the calendar.
- Must ensure hashtags are relevant to the content themes.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Google Sheets to create the calendar?
- Are all required columns (Post Date, Content Theme, Platform, Target Audience, Primary Hashtags) present in the response?
- Is the calendar structured clearly and organized appropriately?
- Does the response include a viewable link to the final sheet?
- Are there at least 10 entries in the calendar, and are the hashtags relevant to the content themes?

### Step 2: Dimension Scoring

#### A. Calendar Accuracy (0.35)
Measures whether the calendar is correctly created with all required columns and structured appropriately.

5 — All required columns are present, and the calendar is structured clearly and appropriately.
4 — All required columns are present, but the structure is slightly unclear or disorganized.
3 — Most required columns are present, but the structure is incomplete or unclear.
2 — Few required columns are present, and the structure is poor.
1 — No required columns are present, or the calendar is completely absent.

#### B. Coverage of Entries (0.30)
Measures whether the calendar includes at least 10 entries with relevant hashtags.

5 — Includes 10 or more entries, and all hashtags are relevant to the content themes.
4 — Includes 10 or more entries, but some hashtags are slightly irrelevant.
3 — Includes 7-9 entries, and most hashtags are relevant.
2 — Includes 4-6 entries, with many irrelevant hashtags.
1 — Includes fewer than 4 entries, or hashtags are completely irrelevant.

#### C. Platform Usage (0.20)
Measures whether the agent correctly specifies the platform (Instagram Stories or Feed) for each entry.

5 — All entries specify the correct platform (Stories or Feed) based on the content theme.
4 — Most entries specify the correct platform, with minor errors.
3 — Some entries specify the correct platform, but there are notable errors.
2 — Few entries specify the correct platform, with significant errors.
1 — No entries specify the correct platform, or all are incorrect.

#### D. Link and Formatting Quality (0.15)
Measures whether the agent provides a viewable link and organizes the output well.

5 — Provides a viewable link and organizes the output in a clear, professional format.
4 — Provides a viewable link, but the format is slightly unclear or unprofessional.
3 — Provides a viewable link, but the format is incomplete or disorganized.
2 — Provides a link, but it is not viewable or the format is poor.
1 — Does not provide a link, or the format is completely absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent created a calendar using Google Sheets, included all required columns, provided a viewable link, and added 10 entries with mostly relevant hashtags.",
  "calendar_accuracy": 5,
  "coverage_of_entries": 5,
  "platform_usage": 5,
  "link_and_formatting_quality": 5,
  "dimension_reasoning": {
    "calendar_accuracy": "All required columns are present and the structure is clear.",
    "coverage_of_entries": "The calendar includes 10 entries with relevant hashtags.",
    "platform_usage": "Each entry specifies the correct platform based on the content theme.",
    "link_and_formatting_quality": "The agent provided a viewable link and organized the output professionally."
  },
  "overall_score": 5.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "calendar_accuracy": 0.35,
    "coverage_of_entries": 0.30,
    "platform_usage": 0.20,
    "link_and_formatting_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())