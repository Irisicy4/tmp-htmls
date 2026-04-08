"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Set up a travel budget tracker in Google Sheets for a 7-day trip to Bali for two people, using estimated values from Skyscanner, Booking.com, and TripAdvisor.
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


TASK_INSTRUCTION = """Set up a travel budget tracker in Google Sheets for a 7-day trip to Bali for two people. Include categories for flights, accommodation, meals, activities, and miscellaneous expenses. Use estimated values from Skyscanner, Booking.com, and TripAdvisor."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a travel budget tracker in Google Sheets for a 7-day trip to Bali for two people. The tracker must include categories for flights, accommodation, meals, activities, and miscellaneous expenses. The agent must use estimated values sourced from Skyscanner, Booking.com, and TripAdvisor.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a travel budget tracker in Google Sheets for a 7-day trip to Bali for two people. Include categories for flights, accommodation, meals, activities, and miscellaneous expenses. Use estimated values from Skyscanner, Booking.com, and TripAdvisor.

## Task-Specific Constraints
- Must visit Skyscanner, Booking.com, and TripAdvisor to gather price estimates.
- Must include price data for flights, accommodation, meals, activities, and miscellaneous expenses.
- Output must be organized as a structured table in Google Sheets.
- Must include estimated values for each category based on platform data.
- Must ensure the tracker is formatted clearly and is usable for budgeting purposes.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Skyscanner, Booking.com, and TripAdvisor? Which platforms were visited?
- Are price estimates for flights, accommodation, meals, activities, and miscellaneous expenses present in the response?
- Is the output organized as a structured table in Google Sheets?
- Are the estimated values sourced from the required platforms?
- Is the tracker formatted clearly and usable for budgeting purposes?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the travel budget tracker is complete and accurate.

5 — Includes all required categories with accurate price estimates sourced from the specified platforms.
4 — Includes most categories with reasonably accurate estimates.
3 — Includes some categories but lacks accuracy or sourcing.
2 — Includes few categories, with significant omissions or inaccuracies.
1 — No usable budget tracker provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and used their data.

5 — Uses data from all three platforms (Skyscanner, Booking.com, TripAdvisor).
4 — Uses data from two platforms.
3 — Uses data from one platform.
2 — Attempts to use platforms but fails to gather usable data.
1 — No platform data used.

#### C. Detail and Specificity (0.20)
Measures the level of detail in the budget tracker.

5 — Provides detailed estimates for each category, with clear breakdowns.
4 — Provides reasonably detailed estimates but lacks breakdowns for some categories.
3 — Provides basic estimates with minimal detail.
2 — Provides vague or incomplete estimates.
1 — No meaningful estimates provided.

#### D. Output Structure and Formatting (0.15)
Measures the clarity and usability of the budget tracker.

5 — Tracker is well-organized, clearly formatted, and easy to use.
4 — Tracker is organized but formatting is slightly unclear.
3 — Tracker is usable but lacks clarity or formatting.
2 — Tracker is disorganized or difficult to use.
1 — No usable tracker provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_formatting": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_formatting": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_formatting": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())