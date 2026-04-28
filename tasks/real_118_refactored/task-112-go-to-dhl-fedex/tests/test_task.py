import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to the DHL Express rate and transit time tool, the FedEx Rate Finder, and the UPS shipping "
    "calculator and get quotes for shipping a 5 kg package measuring 30×20×15 cm from New York, NY "
    "to London, UK with delivery required within 3 business days. For each carrier, record: service "
    "name, quoted price (USD), estimated delivery date, and any fuel surcharge shown. Then check "
    "each carrier's service alert or network status page for any current delays on the transatlantic "
    "route. Produce a carrier comparison table with columns: Carrier, Service, Price (USD), Delivery "
    "Date, Fuel Surcharge, Current Service Alert, and a final 'recommended choice' row.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
  "dimension_reasoning": {{
    "rate_calculator_navigation": "<one sentence citing specific evidence>",
    "quote_data_completeness": "<one sentence citing specific evidence>",
    "service_alert_check": "<one sentence citing specific evidence>",
    "output_and_recommendation": "<one sentence citing specific evidence>"
  }},
  "rate_calculator_navigation": <1-5>,
  "quote_data_completeness": <1-5>,
  "service_alert_check": <1-5>,
  "output_and_recommendation": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "rate_calculator_navigation": 0.35,
    "quote_data_completeness": 0.23,
    "service_alert_check": 0.23,
    "output_and_recommendation": 0.19,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
