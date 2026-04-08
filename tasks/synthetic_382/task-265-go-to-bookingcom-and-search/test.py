"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Search for hotels in Reykjavik, Iceland for specific dates and filters, and extract the top 5 ranked by guest ratings.
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


TASK_INSTRUCTION = """Go to Booking.com and search for hotels in Reykjavik, Iceland for a stay from June 15–20, applying filters for 'Free Wi-Fi' and 'Breakfast Included'. Extract the top 5 hotels ranked by guest ratings. Provide names, nightly rates, and guest ratings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to search for hotels in Reykjavik, Iceland on Booking.com for specific dates, applying filters for 'Free Wi-Fi' and 'Breakfast Included'. The agent must extract the top 5 hotels ranked by guest ratings and provide their names, nightly rates, and guest ratings in the output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Booking.com and search for hotels in Reykjavik, Iceland for a stay from June 15–20, applying filters for 'Free Wi-Fi' and 'Breakfast Included'. Extract the top 5 hotels ranked by guest ratings. Provide names, nightly rates, and guest ratings.

## Task-Specific Constraints
- Must navigate to Booking.com and apply the specified filters ('Free Wi-Fi' and 'Breakfast Included').
- Must extract data for exactly 5 hotels ranked by guest ratings.
- Must include nightly rates, guest ratings, and names for each hotel.
- Output must be organized as a structured list or table.
- Guest ratings must be sourced directly from Booking.com.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Booking.com and apply the required filters ('Free Wi-Fi' and 'Breakfast Included')?
- Are the names, nightly rates, and guest ratings of 5 hotels present in the response?
- Is the output organized as a structured list or table?
- Are the guest ratings sourced directly from Booking.com?
- Are there any missing or incorrect details in the extracted data?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and extracted the top 5 hotels ranked by guest ratings.

5 — Extracts names, nightly rates, and guest ratings for 5 hotels ranked by guest ratings.
4 — Extracts data for 4–5 hotels but with minor inaccuracies.
3 — Extracts data for at least 3 hotels but with noticeable errors or omissions.
2 — Extracts data for fewer than 3 hotels or with significant errors.
1 — Fails to extract any usable data.

#### B. Coverage of Filters and Platforms (0.30)
Measures whether the agent applied the required filters and used Booking.com.

5 — Applies both filters ('Free Wi-Fi' and 'Breakfast Included') and uses Booking.com.
4 — Applies one filter or uses Booking.com with minor omissions.
3 — Uses Booking.com but fails to apply filters correctly.
2 — Navigates to Booking.com but does not apply filters.
1 — Does not use Booking.com or apply filters.

#### C. Depth and Specificity (0.25)
Measures the level of detail and specificity in the extracted data.

5 — Provides accurate nightly rates and guest ratings for all 5 hotels.
4 — Provides accurate details for 4–5 hotels but with minor omissions.
3 — Provides details for at least 3 hotels but lacks specificity or accuracy.
2 — Provides vague or incomplete details for fewer than 3 hotels.
1 — Provides no specific details.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sourced credibly.

5 — Output is structured as a clear list or table and sources ratings from Booking.com.
4 — Output is mostly structured but with minor formatting issues.
3 — Output is partially structured or lacks clarity.
2 — Output is disorganized or unclear.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_filters_and_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_filters_and_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_filters_and_platforms": 0.30,
    "depth_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())