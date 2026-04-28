import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please analyze the Amazon store at https://www.amazon.com/stores/Avery/page and help me find potential B2B customers to sell our products to.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must visit the exact Amazon store URL
- Store: Avery brand store
- Goal: identify B2B customer segments for the products in the store
- Output: list of potential B2B customer types with reasoning

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent visit the Avery Amazon store?
- What products/categories were identified in the store?
- What B2B customer segments were identified?
- Is the B2B identification grounded in the store's actual products?

### Step 2: Dimension Scoring

#### A. Store Analysis (0.3)
Did the agent analyze the Amazon store?

5 — Agent visited the store URL and identified product categories, price points, and brand positioning.
4 — Store visited but analysis is shallow.
3 — Visited store but only noted a few products.
2 — Described what an Amazon store looks like without visiting.
1 — No store visit.

#### B. B2B Identification (0.35)
Were relevant B2B customer segments identified?

5 — 4+ specific B2B segments identified that logically need these products (e.g. offices, schools, print shops, event planners if it's a label/stationery brand).
4 — 2-3 specific B2B segments.
3 — 1-2 segments identified.
2 — Vague 'businesses that buy products' without specificity.
1 — No B2B segments identified.

#### C. Reasoning Quality (0.25)
Is B2B identification grounded in the store's products?

5 — Each customer segment directly linked to specific products in the store with clear rationale.
4 — Good reasoning but not all segments linked to specific products.
3 — General reasoning without product-specific links.
2 — Generic B2B advice not tied to the store.
1 — No reasoning.

#### D. Actionability (0.1)
Is the output actionable for sales outreach?

5 — Includes how to reach each B2B segment (industry associations, LinkedIn, trade shows, etc.).
4 — Outreach channels mentioned for some segments.
3 — Segments identified without outreach guidance.
2 — Very generic.
1 — Not actionable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "store_analysis": "<one sentence citing specific evidence>",
    "b2b_identification": "<one sentence citing specific evidence>",
    "reasoning_quality": "<one sentence citing specific evidence>",
    "actionability": "<one sentence citing specific evidence>",
  }},
  "store_analysis": <1-5>,
  "b2b_identification": <1-5>,
  "reasoning_quality": <1-5>,
  "actionability": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "store_analysis": 0.3,
    "b2b_identification": 0.35,
    "reasoning_quality": 0.25,
    "actionability": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
