import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Run a vitamin price check on Vivomixx 112 food supplement across multiple vendor websites and compile a price comparison report.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Product: Vivomixx 112 capsules specifically (not other Vivomixx variants)
- Coverage: at least 4 vendor websites compared
- Report: structured with vendor name, price, shipping, and URL
- Best deal: must identify the lowest price

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Vivomixx 112 specifically?
- How many vendors were checked?
- What prices were found (with currency)?
- Was a comparison table or report produced?
- Was the best deal identified?

### Step 2: Dimension Scoring

#### A. Product Targeting (0.2)
Did the agent search for Vivomixx 112 specifically?

5 — Searched for Vivomixx 112 (capsule count) and found it on multiple sites.
4 — Found Vivomixx 112 on some sites; others show different variants.
3 — Found Vivomixx generally but not always the 112-count variant.
2 — Found similar products but not Vivomixx 112 specifically.
1 — Wrong product or no product found.

#### B. Vendor Coverage (0.25)
How many vendors were compared?

5 — 5 or more vendors with prices.
4 — 4 vendors.
3 — 3 vendors.
2 — 2 vendors.
1 — Only 1 vendor or none.

#### C. Price Accuracy (0.35)
Are prices accurate with clear sourcing?

5 — Prices from vendor pages with URLs, current date, correct product variant.
4 — Prices accurate but missing some URLs.
3 — Prices found but sourcing unclear.
2 — Prices estimated or from aggregators without direct verification.
1 — No reliable prices.

#### D. Report Structure (0.2)
Is the comparison report well-structured?

5 — Table format: vendor, price, shipping, total, URL; best deal highlighted.
4 — Structured list with most fields.
3 — Prices listed without full structure.
2 — Narrative without clear comparison structure.
1 — No report structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "product_targeting": "<one sentence citing specific evidence>",
    "vendor_coverage": "<one sentence citing specific evidence>",
    "price_accuracy": "<one sentence citing specific evidence>",
    "report_structure": "<one sentence citing specific evidence>",
  }},
  "product_targeting": <1-5>,
  "vendor_coverage": <1-5>,
  "price_accuracy": <1-5>,
  "report_structure": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "product_targeting": 0.2,
    "vendor_coverage": 0.25,
    "price_accuracy": 0.35,
    "report_structure": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
