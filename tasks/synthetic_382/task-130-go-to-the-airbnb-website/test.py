"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Search for Airbnb listings in Boston, MA for a weekend stay, apply filters, and extract details of the first five matching listings.
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


TASK_INSTRUCTION = """Go to the Airbnb website and search for listings in Boston, MA for a weekend stay (Friday to Sunday, two weeks from today). Apply filters for entire homes under $200 per night, that have at least 2 bedrooms and free Wi-Fi. Extract the first five matching listings along with their names, prices, and overall ratings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to search for Airbnb listings in Boston, MA for a weekend stay (Friday to Sunday, two weeks from today). The agent must apply filters for entire homes under $200 per night, that have at least 2 bedrooms and free Wi-Fi. The deliverable is a structured output containing the first five matching listings, including their names, prices, and overall ratings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the Airbnb website and search for listings in Boston, MA for a weekend stay (Friday to Sunday, two weeks from today). Apply filters for entire homes under $200 per night, that have at least 2 bedrooms and free Wi-Fi. Extract the first five matching listings along with their names, prices, and overall ratings.

## Task-Specific Constraints
- Must navigate to airbnb.com and perform a search for Boston, MA.
- Must apply filters for entire homes, under $200 per night, at least 2 bedrooms, and free Wi-Fi.
- Must extract exactly five listings with names, prices, and overall ratings.
- Output must be structured as a table or JSON list.
- Must ensure the extracted data matches the applied filters.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to airbnb.com and perform the required search?
- Were the specified filters applied correctly (entire homes, price, bedrooms, Wi-Fi)?
- Does the output contain exactly five listings with names, prices, and ratings?
- Is the output structured as a table or JSON list?
- Are the extracted details accurate and match the filters?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly extracted the required listings and details.

5 — Extracts exactly five listings with names, prices, and ratings, all matching filters.
4 — Extracts five listings but with minor inaccuracies in details or filters.
3 — Extracts fewer than five listings or with notable inaccuracies.
2 — Extracts listings but most details are incorrect or missing.
1 — Fails to extract any listings or details.

#### B. Filter Application Coverage (0.30)
Measures whether the agent applied all specified filters correctly.

5 — Applies all filters (entire homes, price, bedrooms, Wi-Fi) correctly.
4 — Applies most filters correctly but misses one minor filter.
3 — Applies some filters but misses key constraints (e.g., price or bedrooms).
2 — Applies few filters correctly.
1 — Fails to apply any filters.

#### C. Detail Specificity (0.20)
Measures the depth and specificity of extracted details.

5 — Provides accurate names, prices, and ratings for all listings.
4 — Provides mostly accurate details with minor omissions.
3 — Provides partially accurate details with notable omissions.
2 — Provides few accurate details or mostly incorrect data.
1 — Provides no accurate details.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and the data is credible.

5 — Output is structured as a table or JSON list, and data is credible.
4 — Output is structured but with minor formatting issues or unclear credibility.
3 — Output is partially structured or lacks clarity.
2 — Output is poorly structured or lacks credibility.
1 — Output is unstructured or completely unclear.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent navigated to airbnb.com, applied filters, and extracted listings. Output contains five listings with names, prices, and ratings, but some details are inaccurate.",
  "primary_deliverable_accuracy": 4,
  "filter_application_coverage": 4,
  "detail_specificity": 3,
  "output_structure_and_credibility": 4,
  "dimension_reasoning": {
    "primary_deliverable_accuracy": "Extracted five listings but with minor inaccuracies in details.",
    "filter_application_coverage": "Most filters were applied correctly, but one minor filter was missed.",
    "detail_specificity": "Details were partially accurate but lacked depth in some cases.",
    "output_structure_and_credibility": "Output was structured as a table and appeared credible."
  },
  "overall_score": 3.85,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "filter_application_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())