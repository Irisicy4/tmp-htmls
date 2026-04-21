import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to CommercialCafe and search for available office spaces for lease in downtown Miami, FL "
    "with at least 2,000 square feet. Identify three listings that are currently available. For each, "
    "record: property address, asking rent per square foot per year, total square footage, lease type "
    "(gross, NNN, modified gross), available date, and landlord or listing broker name. Then verify "
    "each property's ownership and assessed value on the Miami-Dade County Property Appraiser website. "
    "Compile a commercial lease comparison table with columns: Address, Asking Rent ($/sqft/yr), "
    "Size (sqft), Lease Type, Assessed Value, Owner of Record, Available Date, and a 'value score' "
    "commentary comparing asking rent to market norms.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
""")

DIMENSION_WEIGHTS = {
    "commercialcafe_search":      0.25,
    "listing_data_completeness":  0.25,
    "appraiser_crossref":         0.30,
    "output_quality":             0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
