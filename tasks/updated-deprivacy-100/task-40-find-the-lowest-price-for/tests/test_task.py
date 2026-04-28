"""LLM-as-judge evaluator for task-40-find-the-lowest-price-for.

Category: Shopping
Task: Find the lowest price for Numi organic tea flower tea gift set across the web.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Find the lowest price for Numi organic tea flower tea gift set across the web.'

SYSTEM_PROMPT_EXTRA = 'You are an expert evaluator for AI agent benchmarks assessing a cross-web lowest-price search task for a specific product.'

CONSTRAINTS = """- Product: Numi Organic Tea Flowering Tea Gift Set specifically (not generic Numi tea)
- Scope: across the web — must check multiple retailers (Amazon, Numi website, specialty tea retailers, etc.)
- Goal: lowest price — must compare and identify the cheapest option
- Shipping: ideally should consider total cost including shipping"""

EVIDENCE_QUESTIONS = """- Did the agent search for the specific Numi flowering tea gift set?
- How many retailers were checked?
- What is the lowest price found? From which retailer?
- Were prices verified from actual current listings?
- Was shipping cost considered?"""

DIMENSION_RUBRICS = """#### A. Search Breadth
Did the agent check multiple retailers?

5 — 3+ retailers checked (e.g. Amazon, Numi website, iHerb, Walmart, specialty tea shops); prices compared.
4 — 2 retailers checked with actual prices.
3 — 1 retailer checked; others referenced but not verified.
2 — General web search only; no specific retailer navigation.
1 — No search; response from prior knowledge.

#### B. Price Verification
Are the prices current and from actual listings?

5 — Prices cited with retailer source; prices are plausibly current (not clearly outdated); specific product variant identified.
4 — Prices from actual listings; one may be slightly outdated or product variant not fully specified.
3 — Prices given but source unclear or agent did not verify they are current.
2 — Prices estimated or from prior knowledge without verification.
1 — No prices; or fabricated prices.

#### C. Lowest Price Identified
Did the agent clearly identify the lowest price?

5 — Lowest price explicitly stated with retailer name; comparison to other options shown.
4 — Lowest price stated; comparison to alternatives not fully shown.
3 — Multiple prices listed but lowest not explicitly called out.
2 — Prices listed in no particular order without identifying lowest.
1 — No price comparison; just one price from one source.

#### D. Result Actionability
Can the user immediately act on the recommendation?

5 — Retailer name, price, and direct URL or product name for easy search; shipping cost noted.
4 — Retailer and price provided; URL missing but product findable.
3 — Retailer named but price or product specifics unclear.
2 — Vague guidance ("check Amazon for best price") without specific finding.
1 — No actionable information."""

DIMENSION_WEIGHTS = {
    'search_breadth': 0.25,
    'price_verification': 0.35,
    'lowest_price_identified': 0.25,
    'result_actionability': 0.15,
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
