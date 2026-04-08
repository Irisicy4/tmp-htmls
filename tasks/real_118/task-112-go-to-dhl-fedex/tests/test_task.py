"""
LLM-as-judge evaluator for EvolveBench task-112.

Category: Logistics & Supply Chain
Task: Get shipping quotes from DHL, FedEx, and UPS for a 5kg NYC→London package, check service
      alerts, and produce a carrier comparison table with a recommendation.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to the DHL Express rate and transit time tool, the FedEx Rate Finder, and the UPS shipping "
    "calculator and get quotes for shipping a 5 kg package measuring 30×20×15 cm from New York, NY "
    "to London, UK with delivery required within 3 business days. For each carrier, record: service "
    "name, quoted price (USD), estimated delivery date, and any fuel surcharge shown. Then check "
    "each carrier's service alert or network status page for any current delays on the transatlantic "
    "route. Produce a carrier comparison table with columns: Carrier, Service, Price (USD), Delivery "
    "Date, Fuel Surcharge, Current Service Alert, and a final 'recommended choice' row."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a multi-carrier shipping rate comparison task including service alert checks.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Carriers: DHL Express, FedEx, AND UPS — all three required
- Package: 5 kg, 30×20×15 cm, New York NY → London UK, within 3 business days
- Required per carrier: service name, quoted price (USD), delivery date, fuel surcharge
- Service alerts: Must check each carrier's network status/alert page for transatlantic delays
- Output: Six-column table plus recommendation row with justification

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate rate calculators for all three carriers (DHL, FedEx, UPS)?
- Are specific quoted prices retrieved for each carrier's expedited service?
- Did the agent check service alert pages for each carrier?
- Is there a complete comparison table with a recommendation?

### Step 2: Dimension Scoring

#### A. Rate Calculator Navigation
Did the agent navigate all three carrier rate calculators and retrieve quotes?

5 — Navigated DHL, FedEx, and UPS rate tools; retrieved quoted prices for a qualifying service (3-business-day delivery) from each, with specific USD amounts.
4 — Navigated two of three carrier sites and retrieved prices; the third was attempted but blocked or returned no quote.
3 — Navigated at least two carrier sites and retrieved at least one specific price; the third carrier has an estimated or missing price.
2 — Navigated one carrier site with a specific price; the other two are estimated or from prior knowledge.
1 — No carrier rate tool navigation; all prices are fabricated or generic estimates.

#### B. Quote Data Completeness
Are service name, price, estimated delivery date, and fuel surcharge present for all three carriers?

5 — All three carriers have service name, USD price, estimated delivery date, and fuel surcharge (or explicit note that surcharge was not shown).
4 — Two of three carriers have all four fields; one is missing the fuel surcharge or exact delivery date.
3 — All three carriers have service name and price; delivery dates or fuel surcharges commonly absent.
2 — Service names and prices present for at least two carriers; other fields missing.
1 — No complete quote data for any carrier.

#### C. Service Alert Check
Did the agent check service alert pages for each carrier?

5 — Service alert or network status page checked for all three carriers; specific alert status (or "no current delays") reported for transatlantic route.
4 — Service alerts checked for two of three carriers with specific findings.
3 — Service alerts mentioned and at least one carrier's status reported; the other two are assumed or generic.
2 — Service alert pages referenced but no specific findings reported for any carrier.
1 — Service alerts not checked; alert column is blank or "not checked."

#### D. Output Table & Recommendation
Is the comparison table complete and the recommendation specific and justified?

5 — Complete six-column table for all three carriers plus a recommendation row that names the carrier, service, and justification (e.g., "FedEx International Priority — lowest price with no service alerts").
4 — Table with five columns and a recommendation present but justification is generic.
3 — Table with four or fewer columns; or recommendation is missing.
2 — Partial table without a recommendation.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "rate_calculator_navigation": <1-5>,
  "quote_data_completeness": <1-5>,
  "service_alert_check": <1-5>,
  "output_and_recommendation": <1-5>,
  "dimension_reasoning": {{
    "rate_calculator_navigation": "<one sentence citing specific evidence>",
    "quote_data_completeness": "<one sentence citing specific evidence>",
    "service_alert_check": "<one sentence citing specific evidence>",
    "output_and_recommendation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "rate_calculator_navigation": 0.30,
    "quote_data_completeness":    0.25,
    "service_alert_check":        0.25,
    "output_and_recommendation":  0.20,
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
