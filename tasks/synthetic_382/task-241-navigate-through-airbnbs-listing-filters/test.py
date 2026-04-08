"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Find accommodations in Kyoto, Japan, within a budget and specific amenities, and extract the top 5 listings with their nightly price and customer ratings.
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


TASK_INSTRUCTION = """Navigate through Airbnb's listing filters to find accommodations in Kyoto, Japan, with a budget of $100-$150 per night, availability in November, and amenities such as Wi-Fi and air conditioning. Extract the top 5 listings with their nightly price and customer ratings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating Airbnb's filters to find accommodations in Kyoto, Japan, within a budget of $100-$150 per night, available in November, and including amenities such as Wi-Fi and air conditioning. A successful completion requires extracting the top 5 listings with their nightly price and customer ratings, presented in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate through Airbnb's listing filters to find accommodations in Kyoto, Japan, with a budget of $100-$150 per night, availability in November, and amenities such as Wi-Fi and air conditioning. Extract the top 5 listings with their nightly price and customer ratings.

## Task-Specific Constraints
- Must navigate Airbnb's filters to specify location, budget, dates, and amenities.
- Must extract exactly 5 listings, no more or less.
- Each listing must include nightly price and customer rating.
- Output must be presented as a structured list or table.
- Listings must match all specified constraints (location, budget, amenities, and availability).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Airbnb's filters correctly to specify location, budget, dates, and amenities?
- Are exactly 5 listings extracted in the response?
- Does each listing include both nightly price and customer rating?
- Is the output presented in a structured format (e.g., list or table)?
- Do the listings match all specified constraints (location, budget, amenities, and availability)?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted listings are correct and meet all specified constraints.

5 — All 5 listings match the constraints (location, budget, amenities, availability) and include nightly price and customer rating.
4 — 4 listings match the constraints and include required details; 1 listing has minor errors.
3 — At least 3 listings match the constraints and include required details; others have errors or omissions.
2 — Fewer than 3 listings match the constraints; significant errors or omissions.
1 — No listings match the constraints or required details are missing.

#### B. Coverage of Filters (0.30)
Measures whether the agent correctly navigated Airbnb's filters to specify all required constraints.

5 — All filters (location, budget, dates, amenities) are correctly applied.
4 — Most filters are correctly applied; minor omissions or errors.
3 — Some filters are applied correctly; others are missing or incorrect.
2 — Few filters are applied correctly; significant omissions or errors.
1 — No filters are applied correctly.

#### C. Detail Specificity (0.20)
Measures whether the extracted listings include detailed and specific information.

5 — Listings include nightly price, customer rating, and additional details (e.g., property type, description).
4 — Listings include nightly price and customer rating; minor details are missing.
3 — Listings include nightly price and customer rating; no additional details.
2 — Listings lack either nightly price or customer rating; minimal detail provided.
1 — Listings lack both nightly price and customer rating; no detail provided.

#### D. Output Structure (0.15)
Measures whether the response is presented in a structured and readable format.

5 — Output is presented as a well-organized table or structured list.
4 — Output is structured but has minor formatting issues.
3 — Output is minimally structured; readability is acceptable but not ideal.
2 — Output is poorly structured; difficult to read or interpret.
1 — Output is unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_filters": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_filters": "<one sentence citing specific evidence>",
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
    "coverage_of_filters": 0.30,
    "detail_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())