"""
LLM-as-judge evaluator for EvolveBench task-106.

Category: Real Estate
Task: Compare Airbnb and VRBO listings in Lisbon for vacation rental ROI, cross-reference
      with AirDNA market data, and produce a ROI analysis table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to Airbnb and search for entire-home listings in Lisbon, Portugal with capacity for at "
    "least 4 guests. Find three active listings with 10+ reviews and record: listing title, nightly "
    "rate, occupancy-implied availability (count available nights in the next 30 days), number of "
    "reviews, and average review score. Then search for the same property type on VRBO in the same "
    "city and find two comparable listings. Finally, visit AirDNA's free market overview for Lisbon "
    "to note the published average daily rate and occupancy rate. Produce a vacation rental ROI "
    "analysis table comparing each listing on: Platform, Nightly Rate, Estimated Monthly Revenue "
    "(nightly rate × occupancy × 30), Review Score, and an ROI feasibility comment."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a vacation rental market research task across Airbnb, VRBO, and AirDNA.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platforms: Airbnb (3 listings), VRBO (2 listings), AirDNA market data — all three required
- Location: Lisbon, Portugal specifically
- Airbnb filter: entire-home, 4+ guests, 10+ reviews
- Required listing fields: title/URL, nightly rate, availability (next 30 days), review count, review score
- AirDNA: average daily rate and occupancy rate for Lisbon market
- Revenue calculation: nightly rate × (occupancy rate from AirDNA or estimated) × 30 days
- Output: Comparison table with Platform, Nightly Rate, Est. Monthly Revenue, Review Score, ROI comment

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Airbnb and find Lisbon entire-home listings for 4+ guests with 10+ reviews?
- Did the agent navigate VRBO and find comparable Lisbon listings?
- Did the agent access AirDNA market overview for Lisbon and note ADR and occupancy rate?
- Are monthly revenue estimates calculated?
- Is there a comparison table with the required columns?

### Step 2: Dimension Scoring

#### A. Multi-Platform Listing Retrieval
Did the agent retrieve listings from both Airbnb and VRBO?

5 — Three Airbnb listings and two VRBO listings found in Lisbon with specific details (title, nightly rate, reviews).
4 — Three Airbnb listings found with good detail; VRBO listings found but one is incomplete.
3 — At least two Airbnb listings and one VRBO listing found; some details missing.
2 — Only one platform searched with complete data; the other platform attempted but no listings retrieved.
1 — No actual listing retrieval; data is fabricated or entirely from prior knowledge.

#### B. Listing Data Quality
Are the required fields (nightly rate, availability, review count, review score) present for most listings?

5 — All five listings (3 Airbnb + 2 VRBO) have nightly rate, review count, and review score; availability count present for Airbnb listings.
4 — Four of five listings have all required fields; one is missing one field.
3 — Three of five listings have most fields; availability counts are commonly missing.
2 — Fewer than three listings with complete data.
1 — No specific listing data; all fields are generic or estimated.

#### C. AirDNA Market Data
Did the agent access AirDNA and retrieve Lisbon market ADR and occupancy rate?

5 — AirDNA accessed; specific ADR (e.g., "€X/night") and occupancy rate (e.g., "X%") for Lisbon reported.
4 — AirDNA accessed; either ADR or occupancy rate retrieved but not both.
3 — AirDNA mentioned and market data referenced but values are vague or not clearly from the current AirDNA page.
2 — AirDNA mentioned but no specific market data retrieved; values estimated.
1 — AirDNA not accessed; no market-level data provided.

#### D. Revenue Calculation & ROI Table
Are monthly revenue estimates calculated and the output table complete?

5 — Monthly revenue calculated for all listings using nightly rate × occupancy × 30; complete table with all five columns including ROI feasibility comment per listing.
4 — Revenue calculated for four of five listings; table present with minor gaps.
3 — Revenue calculated for at least three listings; table present with two or more missing columns.
2 — Table present but revenue not calculated; only raw rates listed.
1 — No table and no revenue calculation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "listing_retrieval": <1-5>,
  "listing_data_quality": <1-5>,
  "airdna_market_data": <1-5>,
  "revenue_calc_and_table": <1-5>,
  "dimension_reasoning": {{
    "listing_retrieval": "<one sentence citing specific evidence>",
    "listing_data_quality": "<one sentence citing specific evidence>",
    "airdna_market_data": "<one sentence citing specific evidence>",
    "revenue_calc_and_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "listing_retrieval":       0.25,
    "listing_data_quality":    0.25,
    "airdna_market_data":      0.25,
    "revenue_calc_and_table":  0.25,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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
