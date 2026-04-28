import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to Redfin and search for single-family homes for sale in Austin, TX with 3+ bedrooms, "
    "2+ bathrooms, and a price between $400,000 and $600,000. Identify the five listings with the "
    "most days on market. For each property, record: address, list price, days on market, square "
    "footage, price per square foot, and the listing agent's name. Then look up each property's "
    "assessed value and tax history on the Travis County Appraisal District website. Calculate the "
    "list-price-to-assessed-value ratio for each property and produce a buyer's market analysis "
    "table with columns: Address, List Price, Assessed Value, L/A Ratio, Days on Market, "
    "Price/sqft, and a brief 'opportunity signal' note.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
  "dimension_reasoning": {{
    "redfin_search": "<one sentence citing specific evidence>",
    "property_data_completeness": "<one sentence citing specific evidence>",
    "assessed_value_crossref": "<one sentence citing specific evidence>",
    "la_ratio_and_signals": "<one sentence citing specific evidence>"
  }},
  "redfin_search": <1-5>,
  "property_data_completeness": <1-5>,
  "assessed_value_crossref": <1-5>,
  "la_ratio_and_signals": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "redfin_search":              0.25,
    "property_data_completeness": 0.25,
    "assessed_value_crossref":    0.30,
    "la_ratio_and_signals":       0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
