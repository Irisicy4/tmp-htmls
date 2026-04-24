"""LLM-as-judge evaluator for task-26-find-me-large-size-hip.

Category: Shopping
Task: Find me large-size hip support and thigh compression sleeves on Amazon. I weigh 280 lbs and I am 50 years old.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Find me large-size hip support and thigh compression sleeves on Amazon. I weigh 280 lbs and I am 50 years old.'

SYSTEM_PROMPT_EXTRA = 'You are an expert evaluator for AI agent benchmarks assessing an Amazon product search task with specific user constraints.'

CONSTRAINTS = """- Platform: Amazon specifically
- Product type: hip support AND/OR thigh compression sleeves (both are acceptable; either alone is partial)
- Size constraint: must fit a 280 lb user — agent must filter for plus-size/bariatric/XXL options
- Age context: 50 years old — implies recovery/joint support context, not athletic performance
- Agent must search Amazon and retrieve actual current listings with prices/ratings"""

EVIDENCE_QUESTIONS = """- Did the agent search Amazon? Cite evidence.
- Were products found for hip support AND thigh compression, or only one?
- Are the products suitable for a 280 lb user (plus-size, XXL, or explicitly rated for heavy users)?
- Are specific product names/ASINs/prices provided?
- Did the agent consider the user's age/context in recommendations?"""

DIMENSION_RUBRICS = """#### A. Amazon Search Execution
Did the agent actively search Amazon for current listings?

5 — Agent searched Amazon and retrieved current product listings with names, prices, and ratings.
4 — Agent searched Amazon but retrieved limited results or only product names without prices/ratings.
3 — Agent used general web search referencing Amazon rather than navigating Amazon directly.
2 — Agent described what to search for without performing the search.
1 — No search; response from prior knowledge or refused.

#### B. Size Appropriateness
Are the recommended products suitable for a 280 lb user?

5 — All products explicitly accommodate 280 lb users; size charts or weight limits mentioned; XXL/2XL/bariatric options specified.
4 — Products appear to be plus-size or large range; weight limit not explicitly verified but plausibly suitable.
3 — Agent mentioned "large size" without verifying 280 lb suitability; standard large may not fit.
2 — Products are standard sizes without considering the user's weight.
1 — Size constraint ignored entirely.

#### C. Product Coverage
Did the agent find both hip support AND thigh compression products?

5 — Both product types found with 2+ options each; clear distinction between hip support and thigh compression.
4 — Both types found but only 1 option for one type, or types are conflated.
3 — Only one product type found; the other is described but not searched.
2 — Only one product type found with no mention of the other.
1 — Neither product type found as a specific Amazon listing.

#### D. Recommendation Quality
Are the recommendations specific, actionable, and suitable for the user's context?

5 — Specific products with name, ASIN or link, price, rating, and brief explanation of why suitable for this user (280 lb, joint support context).
4 — Specific products with most details; user context (age/weight) briefly considered.
3 — Product names provided but lacking price/rating or user-context consideration.
2 — Very generic recommendations without specific product details.
1 — No actionable recommendations."""

DIMENSION_WEIGHTS = {
    'amazon_search_execution': 0.25,
    'size_appropriateness': 0.35,
    'product_coverage': 0.2,
    'recommendation_quality': 0.2,
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
