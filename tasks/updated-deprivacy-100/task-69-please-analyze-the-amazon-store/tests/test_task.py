"""LLM-as-judge evaluator for task-69-please-analyze-the-amazon-store.

Category: Marketing & Analytics
Task: Please analyze the Amazon store at https://www.amazon.com/stores/Avery/page and help me find potential B2B customers to sell our products to.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please analyze the Amazon store at https://www.amazon.com/stores/Avery/page and help me find potential B2B customers to sell our products to.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves analyzing the Avery Amazon brand store to identify the types of products sold, then researching potential B2B customers who might buy those products wholesale or in bulk.'

CONSTRAINTS = """- URL: must visit the exact Amazon store URL
- Store: Avery brand store
- Goal: identify B2B customer segments for the products in the store
- Output: list of potential B2B customer types with reasoning"""

EVIDENCE_QUESTIONS = """- Did the agent visit the Avery Amazon store?
- What products/categories were identified in the store?
- What B2B customer segments were identified?
- Is the B2B identification grounded in the store's actual products?"""

DIMENSION_RUBRICS = """#### A. Store Analysis (0.3)
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
1 — Not actionable."""

DIMENSION_WEIGHTS = {
    'store_analysis': 0.3,
    'b2b_identification': 0.35,
    'reasoning_quality': 0.25,
    'actionability': 0.1,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
