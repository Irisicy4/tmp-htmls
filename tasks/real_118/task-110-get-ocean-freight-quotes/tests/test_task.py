"""
LLM-as-judge evaluator for EvolveBench task-110.

Category: Logistics & Supply Chain
Task: Get ocean freight quote from Freightos (Shanghai→Rotterdam), find ShipBob fulfillment rates,
      look up EU import duty on TARIC, find Dutch VAT rate, and compile a landed cost calculation.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to Freightos and get an ocean freight quote for shipping one 20-foot container (FCL) "
    "of general cargo from Shanghai, China to Rotterdam, Netherlands. Record the quoted freight "
    "rate, transit time, and carrier name. Then go to ShipBob's website to find their published "
    "per-unit fulfillment and storage rates for international orders. Next, look up the EU import "
    "duty rate for 'electrical machinery and equipment' (HS Chapter 85) using the TARIC database, "
    "and confirm the applicable VAT rate using the Dutch Tax Authority customs page. Compile a "
    "landed cost calculation table showing: Ocean Freight, Estimated Insurance (0.5% of €50,000 "
    "cargo value), Import Duty (% and € amount), VAT (% and € amount), ShipBob Fulfillment Cost "
    "(per 100 units), and Total Landed Cost per unit."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a multi-source landed cost calculation task for international shipment.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Freightos (ocean freight), ShipBob (fulfillment rates), TARIC (duty rate), Dutch Tax Authority (VAT)
- Route: Shanghai, China → Rotterdam, Netherlands; 20-foot FCL container
- Cargo value assumption: €50,000 (for insurance and duty/VAT calculations)
- Insurance: 0.5% × €50,000 = €250
- TARIC: HS Chapter 85 duty rate for electrical machinery
- Dutch VAT: standard rate on import (typically 21%)
- Output: Landed cost table with all six line items and a total landed cost per unit figure

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Freightos and retrieve a freight rate for the Shanghai→Rotterdam FCL route?
- Did the agent find ShipBob fulfillment/storage rates?
- Did the agent look up the HS Chapter 85 duty rate on TARIC?
- Did the agent find the Dutch import VAT rate?
- Is there a landed cost table with all six line items and a total?

### Step 2: Dimension Scoring

#### A. Freight Quote Retrieval
Did the agent navigate Freightos and retrieve a specific freight rate?

5 — Navigated Freightos, entered Shanghai→Rotterdam 20ft FCL, and retrieved a specific freight rate with carrier name and transit time.
4 — Navigated Freightos and retrieved a rate but carrier name or transit time missing.
3 — Navigated Freightos but quote was a range or estimation; specific rate unclear.
2 — Mentioned Freightos but rate appears to be a general market estimate rather than a live quote.
1 — No Freightos navigation; freight rate is fabricated or from prior knowledge only.

#### B. Duty & Tax Rate Lookup
Did the agent correctly find the EU import duty rate (TARIC) and Dutch VAT rate?

5 — TARIC database consulted and HS Chapter 85 duty rate found (specific percentage); Dutch import VAT rate confirmed from official source (e.g., 21%).
4 — Duty rate found from TARIC OR VAT rate confirmed from official source, but not both from primary sources.
3 — Both rates cited but one appears to be from memory rather than looked up (e.g., "standard EU rate" without TARIC source).
2 — Both rates cited but from general knowledge without any site navigation.
1 — No duty or VAT rate lookup; rates absent or clearly wrong.

#### C. ShipBob Fulfillment Rates
Did the agent find and report ShipBob's published fulfillment or storage rates?

5 — ShipBob website navigated; specific per-unit fulfillment rate and/or storage rate retrieved.
4 — ShipBob visited and rates referenced but figures are approximate or from a pricing page that requires contacting sales.
3 — ShipBob mentioned and general fulfillment cost structure described; no specific figures retrieved.
2 — ShipBob mentioned but rates not retrieved; agent used generic estimates.
1 — ShipBob not visited; fulfillment cost absent or fabricated.

#### D. Landed Cost Calculation & Table
Is the landed cost table complete with all six line items and a correctly computed total per unit?

5 — Table with all six items (freight, insurance, duty, VAT, fulfillment, total); insurance calculated as 0.5% × €50,000; duty and VAT applied to appropriate base; total landed cost per unit computed.
4 — Table with five of six items; total cost calculated but one line item is missing or estimated.
3 — Table with four items; total cost present but calculation steps not shown.
2 — Partial table (fewer than four items); no total per-unit cost.
1 — No table; unstructured narrative with no calculations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "freight_quote": <1-5>,
  "duty_tax_lookup": <1-5>,
  "shipbob_rates": <1-5>,
  "landed_cost_table": <1-5>,
  "dimension_reasoning": {{
    "freight_quote": "<one sentence citing specific evidence>",
    "duty_tax_lookup": "<one sentence citing specific evidence>",
    "shipbob_rates": "<one sentence citing specific evidence>",
    "landed_cost_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "freight_quote":    0.30,
    "duty_tax_lookup":  0.30,
    "shipbob_rates":    0.15,
    "landed_cost_table": 0.25,
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
