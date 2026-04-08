"""
LLM-as-judge evaluator for EvolveBench task-105.

Category: Real Estate
Task: Search CommercialCafe for Miami office spaces, verify ownership on Miami-Dade
      Property Appraiser, and compile a commercial lease comparison table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to CommercialCafe and search for available office spaces for lease in downtown Miami, FL "
    "with at least 2,000 square feet. Identify three listings that are currently available. For each, "
    "record: property address, asking rent per square foot per year, total square footage, lease type "
    "(gross, NNN, modified gross), available date, and landlord or listing broker name. Then verify "
    "each property's ownership and assessed value on the Miami-Dade County Property Appraiser website. "
    "Compile a commercial lease comparison table with columns: Address, Asking Rent ($/sqft/yr), "
    "Size (sqft), Lease Type, Assessed Value, Owner of Record, Available Date, and a 'value score' "
    "commentary comparing asking rent to market norms."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a commercial real estate research task combining CommercialCafe listings with Miami-Dade property appraiser data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: CommercialCafe for listings AND Miami-Dade Property Appraiser for ownership/assessed value
- Location: Downtown Miami, FL specifically
- Size filter: 2,000+ square feet
- Required fields per listing: address, asking rent ($/sqft/yr), size, lease type, available date, broker/landlord
- Cross-reference: ownership and assessed value from miamidade.gov property appraiser
- Output: Table with all eight columns plus value score commentary

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate CommercialCafe and find downtown Miami office listings with 2,000+ sqft?
- Are three distinct property listings identified with specific addresses?
- Did the agent access Miami-Dade Property Appraiser for ownership and assessed values?
- Is there a complete table with all eight required columns?

### Step 2: Dimension Scoring

#### A. CommercialCafe Search & Listing Retrieval
Did the agent navigate CommercialCafe and find real listings?

5 — Navigated CommercialCafe, applied downtown Miami filter with 2,000+ sqft, and retrieved three distinct current listings with specific addresses.
4 — Navigated CommercialCafe and found listings but one listing is incomplete or filter not fully applied.
3 — Navigated to CommercialCafe but retrieved fewer than three listings or listings lack specific addresses.
2 — Mentioned CommercialCafe but listing details appear generic or not from current availability.
1 — No CommercialCafe navigation; listings are fabricated or from prior knowledge.

#### B. Listing Data Completeness
Are all required commercial lease fields present for all three listings?

5 — All three listings have asking rent ($/sqft/yr), size, lease type, available date, and broker/landlord name.
4 — Two of three listings are fully detailed; one is missing one field (typically lease type or broker).
3 — At least two listings have most fields; lease type or asking rent is commonly missing.
2 — Only one listing is fully detailed; the other two are incomplete.
1 — No complete listing data; generic or fabricated entries.

#### C. Property Appraiser Cross-Reference
Did the agent access Miami-Dade Property Appraiser for ownership and assessed value data?

5 — Owner of record and assessed value retrieved from miamidade.gov for all three properties with specific data.
4 — Ownership/assessed value found for two of three properties from the county appraiser.
3 — Appraiser data found for at least two properties; source is plausible but not clearly miamidade.gov.
2 — Ownership/assessed value cited but source is unclear or values appear estimated.
1 — No property appraiser data; ownership or assessed value absent or fabricated.

#### D. Output Quality & Value Score Commentary
Is the final table complete and does the value score commentary add analytical insight?

5 — Complete table with all eight columns and value score commentary for each listing that references specific market rate context (e.g., "above/below Miami Class A average of $X/sqft/yr").
4 — Table complete but value score comments are generic or one listing lacks commentary.
3 — Table present with six or seven columns; value score commentary is vague for most listings.
2 — Partial table (fewer than six columns); commentary absent or generic.
1 — No table; unstructured narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "commercialcafe_search": <1-5>,
  "listing_data_completeness": <1-5>,
  "appraiser_crossref": <1-5>,
  "output_quality": <1-5>,
  "dimension_reasoning": {{
    "commercialcafe_search": "<one sentence citing specific evidence>",
    "listing_data_completeness": "<one sentence citing specific evidence>",
    "appraiser_crossref": "<one sentence citing specific evidence>",
    "output_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "commercialcafe_search":      0.25,
    "listing_data_completeness":  0.25,
    "appraiser_crossref":         0.30,
    "output_quality":             0.20,
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
