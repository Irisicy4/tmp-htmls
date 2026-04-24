"""LLM-as-judge evaluator for task-60-check-product-on-amazon-australia-brunnings.

Category: Shopping
Task: Please check this product on Amazon Australia: https://www.amazon.com.au/Brunnings-Ant-Killer-Powder-500/dp/B0DR94VX18/ and tell me if it is effective for killing ants, cockroaches, and bull ants.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please check this product on Amazon Australia: https://www.amazon.com.au/Brunnings-Ant-Killer-Powder-500/dp/B0DR94VX18/ and tell me if it is effective for killing ants, cockroaches, and bull ants.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves checking a specific Amazon Australia product listing and evaluating its effectiveness against three pest types based on product details, reviews, and active ingredients.'

CONSTRAINTS = """- URL: must visit the exact Amazon Australia product page provided
- Pests: must specifically address ants, cockroaches, AND bull ants (three separate assessments)
- Evidence: effectiveness claims must be grounded in product description, active ingredients, or customer reviews
- Not just star rating — must assess actual pest-control efficacy"""

EVIDENCE_QUESTIONS = """- Did the agent visit the exact Amazon Australia URL?
- What is the product name and active ingredient?
- Does the product claim to kill ants, cockroaches, and bull ants?
- What evidence (ingredients, reviews, product description) supports the effectiveness assessment?
- Was each pest type addressed individually?"""

DIMENSION_RUBRICS = """#### A. Product Page Access (0.2)
Did the agent visit the correct product page?

5 — Agent navigated to the exact Amazon.com.au URL and retrieved product details.
4 — Agent reached Amazon Australia but had trouble loading the full page.
3 — Agent found the product via search rather than the direct URL.
2 — Agent used Amazon.com (US) instead of Amazon.com.au.
1 — No product page accessed.

#### B. Pest Coverage (0.3)
Were all three pest types addressed?

5 — Ants, cockroaches, and bull ants all assessed individually with specific evidence.
4 — Two pest types assessed with evidence; one is missing or vague.
3 — All three mentioned but without individual assessment.
2 — Only one or two pest types mentioned.
1 — Pest effectiveness not addressed.

#### C. Evidence Quality (0.35)
Is the effectiveness assessment grounded in product evidence?

5 — Assessment based on active ingredients (e.g. permethrin, deltamethrin) AND product claims AND reviews.
4 — Based on two of three evidence types.
3 — Based on product claims alone or reviews alone.
2 — Based only on star rating or vague impressions.
1 — No evidence cited.

#### D. Recommendation Clarity (0.15)
Is the final assessment clear and actionable?

5 — Clear yes/no/partially effective for each pest with reasoning.
4 — Clear overall recommendation but individual pest breakdown is weak.
3 — Recommendation present but hedged without clear conclusion.
2 — Inconclusive or contradictory.
1 — No recommendation."""

DIMENSION_WEIGHTS = {
    'product_page_access': 0.2,
    'pest_coverage': 0.3,
    'evidence_quality': 0.35,
    'recommendation_clarity': 0.15,
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
