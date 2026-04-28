"""LLM-as-judge evaluator for task-31-choose-bag-suitable-for-20.

Category: Shopping
Task: Choose a bag suitable for a 20-year-old. From these four brands (OUTDOOR, GREGORY, COLEMAN, MARIMEKKO) find backpacks under 15,000 yen. Search on Amazon, pay attention to price, and after searching re
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Choose a bag suitable for a 20-year-old. From these four brands (OUTDOOR, GREGORY, COLEMAN, MARIMEKKO) find backpacks under 15,000 yen. Search on Amazon, pay attention to price, and after searching report the product names.'

SYSTEM_PROMPT_EXTRA = 'You are an expert evaluator for AI agent benchmarks assessing a constrained Amazon product search task.'

CONSTRAINTS = """- Platform: Amazon (amazon.co.jp implied given yen pricing)
- Brands: must search all four — OUTDOOR, GREGORY, COLEMAN, MARIMEKKO
- Price: under ¥15,000 hard limit
- Output: product names (not just brand names); specific model names required
- Age suitability: 20-year-old context (style/functionality appropriate for young adult)"""

EVIDENCE_QUESTIONS = """- Did the agent search Amazon for backpacks from these 4 brands?
- Were products found under ¥15,000 from each brand?
- Are specific product names (model names) listed?
- Did the agent apply the price filter correctly?"""

DIMENSION_RUBRICS = """#### AMAZON SEARCH (weight 0.25)
Did the agent search Amazon and retrieve current listings?

5 — Agent navigated Amazon, searched each of the 4 brands, and retrieved current product listings with prices.
4 — Agent searched Amazon for most brands; one brand missing or prices not verified.
3 — Agent used general web search referencing Amazon rather than navigating directly.
2 — Agent described what to search without performing the search.
1 — No search performed.

#### BRAND COVERAGE (weight 0.25)
Were backpacks found from all 4 brands?

5 — At least one backpack under ¥15,000 found and named for each of the 4 brands.
4 — 3 of 4 brands have qualifying products listed; one brand had no qualifying product (noted).
3 — 2 of 4 brands covered with products.
2 — Only 1 brand covered.
1 — No brand-specific products found.

#### PRICE COMPLIANCE (weight 0.3)
Are all listed products genuinely under ¥15,000?

5 — All products explicitly priced and confirmed under ¥15,000; prices cited from Amazon.
4 — All products stated to be under ¥15,000 but prices not explicitly cited for all.
3 — Some products priced; a few are above ¥15,000 or price unclear.
2 — Price filter applied loosely; several products may exceed ¥15,000.
1 — Price constraint ignored entirely.

#### PRODUCT SPECIFICITY (weight 0.2)
Are specific product model names reported (not just brand names)?

5 — Full model names listed for each product (e.g. "Gregory Nano 16 Backpack").
4 — Model names listed for most; 1–2 are brand+type only (e.g. "COLEMAN hiking bag").
3 — Mix of model names and generic descriptions.
2 — Only brand names listed without model names.
1 — No product names at all."""

DIMENSION_WEIGHTS = {
    'amazon_search': 0.25,
    'brand_coverage': 0.25,
    'price_compliance': 0.3,
    'product_specificity': 0.2,
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
