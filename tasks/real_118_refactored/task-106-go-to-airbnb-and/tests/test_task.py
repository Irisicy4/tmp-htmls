import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to Airbnb and search for entire-home listings in Lisbon, Portugal with capacity for at "
    "least 4 guests. Find three active listings with 10+ reviews and record: listing title, nightly "
    "rate, occupancy-implied availability (count available nights in the next 30 days), number of "
    "reviews, and average review score. Then search for the same property type on VRBO in the same "
    "city and find two comparable listings. Finally, visit AirDNA's free market overview for Lisbon "
    "to note the published average daily rate and occupancy rate. Produce a vacation rental ROI "
    "analysis table comparing each listing on: Platform, Nightly Rate, Estimated Monthly Revenue "
    "(nightly rate × occupancy × 30), Review Score, and an ROI feasibility comment.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
""")

DIMENSION_WEIGHTS = {
    "listing_retrieval":       0.25,
    "listing_data_quality":    0.25,
    "airdna_market_data":      0.25,
    "revenue_calc_and_table":  0.25,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
