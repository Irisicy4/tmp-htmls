"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a 5-day itinerary template for a trip to London, UK using Google Sheets and data from TripAdvisor, Visit London, and Transport for London.
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


TASK_INSTRUCTION = """Using Google Sheets, create a 5-day itinerary template for a trip to London, UK. Include columns for date, location, activity, cost, and transportation notes. Populate the template with data gathered from TripAdvisor, Visit London, and Transport for London."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a 5-day itinerary template for a trip to London, UK using Google Sheets. The itinerary must include columns for date, location, activity, cost, and transportation notes, and be populated with data gathered from TripAdvisor, Visit London, and Transport for London. A successful completion involves accurate and structured data from the specified platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets, create a 5-day itinerary template for a trip to London, UK. Include columns for date, location, activity, cost, and transportation notes. Populate the template with data gathered from TripAdvisor, Visit London, and Transport for London.

## Task-Specific Constraints
- Must visit TripAdvisor, Visit London, and Transport for London.
- Must include cost data for all activities and transportation.
- Output must be organized as a table with the specified columns.
- Must include at least one activity per day.
- Transportation notes must be specific and relevant to the activity/location.
- The itinerary must be for 5 distinct days.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TripAdvisor, Visit London, and Transport for London? Which ones were actually visited?
- Are all required columns (date, location, activity, cost, transportation notes) present in the response?
- Is the output organized as a table with structured rows for each day?
- Are cost data and transportation notes specific and relevant to the activities/locations?
- Does the itinerary cover 5 distinct days with at least one activity per day?

### Step 2: Dimension Scoring

#### A. Itinerary Accuracy (0.35)
Measures whether the itinerary is complete, accurate, and adheres to the required format.

5 — All columns are present, data is accurate, and the itinerary covers 5 distinct days with specific activities.
4 — Minor inaccuracies or missing details, but the itinerary is mostly complete and usable.
3 — Partial completion with some missing columns or inaccurate data.
2 — Significant inaccuracies or missing data; the itinerary is barely usable.
1 — No usable itinerary provided.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms and gathered data from them.

5 — Data is gathered from TripAdvisor, Visit London, and Transport for London.
4 — Data is gathered from at least 2 platforms, but one is missing.
3 — Data is gathered from only 1 platform.
2 — No evidence of platform usage or data gathering.
1 — No attempt to use the required platforms.

#### C. Detail Specificity (0.20)
Measures the depth and specificity of the itinerary, including cost and transportation notes.

5 — Cost and transportation notes are specific and relevant for all activities/locations.
4 — Minor omissions or vague details, but most cost and transportation notes are specific.
3 — Partial specificity; some cost or transportation notes are missing or vague.
2 — Significant lack of detail; most cost or transportation notes are missing.
1 — No specific details provided.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and adheres to the required table format.

5 — Output is fully structured as a table with clear rows and columns.
4 — Minor formatting issues, but the table is mostly organized.
3 — Partial structure; the table is incomplete or poorly formatted.
2 — Significant formatting issues; the output is barely readable.
1 — No structured output provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "itinerary_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "itinerary_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "itinerary_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())