"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Filter hotels in Kyoto, Japan on Booking.com based on specific criteria and extract top 5 options.
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


TASK_INSTRUCTION = """Go to Booking.com and filter for hotels in Kyoto, Japan that are within 1 km of the city center, have free Wi-Fi, and a rating above 8.5. Extract the top 5 options with their names, prices per night, and current reviews."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves filtering hotels in Kyoto, Japan on Booking.com based on specific criteria: proximity to the city center (within 1 km), free Wi-Fi availability, and a rating above 8.5. The agent must extract the top 5 options, including their names, prices per night, and current reviews.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Booking.com and filter for hotels in Kyoto, Japan that are within 1 km of the city center, have free Wi-Fi, and a rating above 8.5. Extract the top 5 options with their names, prices per night, and current reviews.

## Task-Specific Constraints
- Must use Booking.com as the platform for filtering and data extraction.
- Must include exactly 5 hotels in the final output.
- Each hotel entry must include name, price per night, and current reviews.
- Ratings must be above 8.5 for all hotels listed.
- Proximity to the city center must be within 1 km for all hotels listed.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Booking.com to perform the task?
- Are there exactly 5 hotels listed in the response?
- Does each hotel include name, price per night, and current reviews?
- Are all hotels rated above 8.5 and located within 1 km of the city center?
- Is the output structured in a clear and readable format?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted hotel data matches the task requirements.

5 — All 5 hotels meet the criteria (rating > 8.5, within 1 km, free Wi-Fi) with complete details.
4 — 4 hotels meet the criteria; minor errors in details for 1.
3 — At least 3 hotels meet the criteria; some missing or incorrect details.
2 — Fewer than 3 hotels meet the criteria; significant missing or incorrect details.
1 — No hotels meet the criteria or response is invalid.

#### B. Coverage of Requirements (0.30)
Measures whether all task-specific constraints are addressed.

5 — All constraints (platform, proximity, rating, Wi-Fi, structured output) are satisfied.
4 — Minor omissions or errors in addressing 1 constraint.
3 — At least 3 constraints are satisfied; others partially addressed.
2 — Fewer than 3 constraints are satisfied; major omissions.
1 — No constraints are satisfied.

#### C. Detail Specificity (0.25)
Measures the depth and specificity of the extracted data.

5 — Includes detailed names, prices, and reviews for all 5 hotels.
4 — Includes detailed data for 4 hotels; minor omissions for 1.
3 — Includes partial data for at least 3 hotels; lacks depth.
2 — Includes minimal or vague data for fewer than 3 hotels.
1 — No specific details provided.

#### D. Output Structure (0.10)
Measures the clarity and organization of the response.

5 — Output is well-structured, easy to read, and logically organized.
4 — Output is mostly clear but has minor formatting issues.
3 — Output is readable but lacks clear organization.
2 — Output is poorly structured or hard to interpret.
1 — Output is disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "detail_specificity": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())