"""
LLM-as-judge evaluator for EvolveBench task-104.

Category: Real Estate
Task: Search Redfin for Austin TX homes with high days-on-market, cross-reference assessed
      values from Travis County Appraisal District, and produce a buyer's market analysis table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to Redfin and search for single-family homes for sale in Austin, TX with 3+ bedrooms, "
    "2+ bathrooms, and a price between $400,000 and $600,000. Identify the five listings with the "
    "most days on market. For each property, record: address, list price, days on market, square "
    "footage, price per square foot, and the listing agent's name. Then look up each property's "
    "assessed value and tax history on the Travis County Appraisal District website. Calculate the "
    "list-price-to-assessed-value ratio for each property and produce a buyer's market analysis "
    "table with columns: Address, List Price, Assessed Value, L/A Ratio, Days on Market, "
    "Price/sqft, and a brief 'opportunity signal' note."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a real estate research task combining Redfin listings with county appraisal data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use Redfin for listings AND Travis County Appraisal District for assessed values
- Location: Austin, TX specifically
- Filters: 3+ bedrooms, 2+ bathrooms, $400k-$600k price range
- Selection criterion: Five listings with the MOST days on market (not just any five)
- Required fields: Address, List Price, Days on Market, Sqft, Price/sqft, Agent, Assessed Value, L/A Ratio
- Calculation: L/A Ratio = List Price ÷ Assessed Value (must be calculated)
- Output: Structured table with all seven columns plus opportunity signal note

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Redfin and apply the correct filters (Austin TX, 3+bd, 2+ba, $400k-$600k)?
- Did the agent identify properties with the highest days on market?
- Did the agent access Travis County Appraisal District for assessed values?
- Are L/A ratios calculated and present in the output?
- Is there a structured table with all required columns?

### Step 2: Dimension Scoring

#### A. Redfin Search Execution
Did the agent navigate Redfin with correct filters and find the right properties?

5 — Navigated Redfin, applied all correct filters (location, beds, baths, price range), and identified five properties ranked by days on market.
4 — Navigated Redfin with correct filters but selected properties not clearly sorted by days on market.
3 — Navigated Redfin but applied incomplete filters (e.g., missing price range or wrong location).
2 — Claimed to search Redfin but property details are generic or not clearly from current listings.
1 — No Redfin navigation; properties are invented or from internal knowledge.

#### B. Property Data Completeness
Are all required fields present for all five properties?

5 — All five properties have address, list price, DOM, sqft, price/sqft, and listing agent name.
4 — Four of five properties are fully detailed; one is missing one field.
3 — Three or four properties have most fields; agent names or sqft commonly missing.
2 — Fewer than three complete property records.
1 — No specific property data; generic or fabricated entries.

#### C. Assessed Value Cross-Reference
Did the agent look up assessed values from Travis County Appraisal District?

5 — Assessed values retrieved from traviscad.org for all five properties with specific values.
4 — Assessed values found for four of five properties from the county appraisal site.
3 — Assessed values found for at least three properties; source is plausible but not clearly traviscad.org.
2 — Assessed values cited but source is unclear or values appear estimated rather than looked up.
1 — No assessed values from the county appraisal district; data absent or fabricated.

#### D. L/A Ratio Calculation & Opportunity Signals
Are L/A ratios calculated correctly and are opportunity signal notes meaningful?

5 — L/A ratios calculated for all five properties (list price ÷ assessed value) and opportunity signals are specific and logical (e.g., "listed 18% above assessed value").
4 — L/A ratios present for four of five; opportunity signals present but some are generic.
3 — L/A ratios present for three or fewer; or ratios are present but opportunity signals are all generic.
2 — Opportunity signals present but no L/A ratios calculated.
1 — No L/A ratios or opportunity signals.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "redfin_search": <1-5>,
  "property_data_completeness": <1-5>,
  "assessed_value_crossref": <1-5>,
  "la_ratio_and_signals": <1-5>,
  "dimension_reasoning": {{
    "redfin_search": "<one sentence citing specific evidence>",
    "property_data_completeness": "<one sentence citing specific evidence>",
    "assessed_value_crossref": "<one sentence citing specific evidence>",
    "la_ratio_and_signals": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "redfin_search":              0.25,
    "property_data_completeness": 0.25,
    "assessed_value_crossref":    0.30,
    "la_ratio_and_signals":       0.20,
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
