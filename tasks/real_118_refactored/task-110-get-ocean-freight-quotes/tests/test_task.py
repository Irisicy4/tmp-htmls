import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to Freightos and get an ocean freight quote for shipping one 20-foot container (FCL) "
    "of general cargo from Shanghai, China to Rotterdam, Netherlands. Record the quoted freight "
    "rate, transit time, and carrier name. Then go to ShipBob's website to find their published "
    "per-unit fulfillment and storage rates for international orders. Next, look up the EU import "
    "duty rate for 'electrical machinery and equipment' (HS Chapter 85) using the TARIC database, "
    "and confirm the applicable VAT rate using the Dutch Tax Authority customs page. Compile a "
    "landed cost calculation table showing: Ocean Freight, Estimated Insurance (0.5% of €50,000 "
    "cargo value), Import Duty (% and € amount), VAT (% and € amount), ShipBob Fulfillment Cost "
    "(per 100 units), and Total Landed Cost per unit.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
""")

DIMENSION_WEIGHTS = {
    "freight_quote":    0.30,
    "duty_tax_lookup":  0.30,
    "shipbob_rates":    0.15,
    "landed_cost_table": 0.25,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
