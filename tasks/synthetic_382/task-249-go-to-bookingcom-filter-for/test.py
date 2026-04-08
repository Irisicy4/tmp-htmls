"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Filter and extract details of hotels in Venice, Italy on Booking.com based on specific criteria.
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


TASK_INSTRUCTION = """Go to Booking.com, filter for hotels in Venice, Italy with a check-in date of March 15th and check-out date of March 20th. Apply filters for 'free cancellation,' 'breakfast included,' and a guest rating of 8.0 or higher. Extract details of 5 matching hotels, including name, price per night, and distance to city center."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Booking.com to search for hotels in Venice, Italy, applying specific filters (free cancellation, breakfast included, guest rating of 8.0 or higher) and extracting details of 5 hotels (name, price per night, and distance to city center). A successful completion includes accurate filtering and structured output with all required details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Booking.com, filter for hotels in Venice, Italy with a check-in date of March 15th and check-out date of March 20th. Apply filters for 'free cancellation,' 'breakfast included,' and a guest rating of 8.0 or higher. Extract details of 5 matching hotels, including name, price per night, and distance to city center.

## Task-Specific Constraints
- Must navigate to Booking.com and perform the search there.
- Must apply all specified filters (free cancellation, breakfast included, guest rating of 8.0 or higher).
- Must extract details for exactly 5 hotels.
- Must include the name, price per night, and distance to city center for each hotel.
- Output must be structured as a list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Booking.com and perform the search there?
- Were all specified filters (free cancellation, breakfast included, guest rating of 8.0 or higher) applied?
- Did the agent extract details for exactly 5 hotels?
- Does the output include the name, price per night, and distance to city center for each hotel?
- Is the output structured as a list or table?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent correctly applied all specified filters on Booking.com.

5 — All filters were applied correctly and verified in the response.
4 — Most filters were applied, but one was missing or incorrect.
3 — Some filters were applied, but multiple were missing or incorrect.
2 — Few filters were applied correctly.
1 — No filters were applied or completely incorrect.

#### B. Data Completeness (0.30)
Measures whether the agent extracted all required details for 5 hotels.

5 — Extracted all required details (name, price per night, distance) for exactly 5 hotels.
4 — Extracted all required details for 4 hotels, or missed one detail for 1-2 hotels.
3 — Extracted partial details for 3-4 hotels.
2 — Extracted partial details for 1-2 hotels.
1 — Did not extract any relevant details.

#### C. Output Structure (0.20)
Measures whether the output is well-organized and presented in a clear, structured format.

5 — Output is fully structured as a table or clear list with all details easy to read.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but difficult to read or interpret.
2 — Output is minimally structured and hard to follow.
1 — Output is unstructured or completely disorganized.

#### D. Platform Use Verification (0.15)
Measures whether the agent used Booking.com as required.

5 — Clear evidence that Booking.com was used for the search.
4 — Likely used Booking.com, but evidence is indirect or incomplete.
3 — Unclear if Booking.com was used, but some evidence suggests it.
2 — Likely did not use Booking.com based on evidence.
1 — No evidence of Booking.com usage.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filtering_accuracy": <1-5>,
  "data_completeness": <1-5>,
  "output_structure": <1-5>,
  "platform_use_verification": <1-5>,
  "dimension_reasoning": {{
    "filtering_accuracy": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>",
    "platform_use_verification": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "data_completeness": 0.30,
    "output_structure": 0.20,
    "platform_use_verification": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())