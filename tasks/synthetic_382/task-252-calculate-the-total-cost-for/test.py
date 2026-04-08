"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Calculate the total cost for a 7-day trip to Rome, Italy, including flights, hotels, and meals, and recommend a total budget.
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


TASK_INSTRUCTION = """Calculate the total cost for a 7-day trip to Rome, Italy. Include live flight costs from Google Flights, average hotel rates from Hotels.com, and daily meal expenses averaged from TripAdvisor (assume 3 meals/day). Recommend a total budget for the trip, breaking down the cost sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost for a 7-day trip to Rome, Italy. The agent must gather live flight prices from Google Flights, average hotel rates from Hotels.com, and daily meal costs from TripAdvisor. The final output must include a breakdown of costs and a recommended total budget.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost for a 7-day trip to Rome, Italy. Include live flight costs from Google Flights, average hotel rates from Hotels.com, and daily meal expenses averaged from TripAdvisor (assume 3 meals/day). Recommend a total budget for the trip, breaking down the cost sources.

## Task-Specific Constraints
- Must retrieve live flight prices from Google Flights.
- Must retrieve average hotel rates from Hotels.com.
- Must retrieve daily meal costs (3 meals/day) from TripAdvisor.
- Must include a clear breakdown of costs (flights, hotels, meals).
- Must recommend a total budget for the trip.
- Output must be structured as a table or clear list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Flights, Hotels.com, and TripAdvisor? Which platforms were actually used?
- Did the agent retrieve live flight prices, average hotel rates, and daily meal costs?
- Is the output structured as a table or clear list with a cost breakdown?
- Are the flight, hotel, and meal costs accurate and sourced from the correct platforms?
- Does the response include a recommended total budget for the trip?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the total budget and cost breakdown are correct and complete.

5 — Includes accurate costs for flights, hotels, and meals, and a correct total budget.
4 — Includes most costs accurately but with minor errors or omissions.
3 — Includes partial costs or an incomplete total budget.
2 — Includes significant errors or omissions in costs or budget.
1 — Does not include a meaningful budget or cost breakdown.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent used all required platforms and retrieved the necessary data.

5 — Uses Google Flights, Hotels.com, and TripAdvisor, retrieving all required data.
4 — Uses at least 2 platforms and retrieves most required data.
3 — Uses at least 1 platform and retrieves partial data.
2 — Attempts to use platforms but retrieves little or no data.
1 — Does not use any required platforms or retrieve data.

#### C. Detail and Specificity (0.20)
Measures the level of detail in the cost breakdown and budget recommendation.

5 — Provides detailed cost breakdowns with specific prices and calculations.
4 — Provides mostly detailed breakdowns with minor gaps.
3 — Provides partial breakdowns with limited detail.
2 — Provides vague or incomplete breakdowns.
1 — Provides no meaningful detail or specificity.

#### D. Output Structure and Clarity (0.15)
Measures how well-organized and clear the output is.

5 — Output is well-structured, easy to read, and logically organized.
4 — Output is mostly clear with minor formatting issues.
3 — Output is somewhat clear but lacks organization.
2 — Output is poorly structured or difficult to follow.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())