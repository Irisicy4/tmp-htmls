"""LLM-as-judge evaluator for task-32-help-me-choose-the-perfect.

Category: Shopping
Task: Help me choose the perfect birthday gift for my 28-year-old male friend. He likes gaming, computers and gadgets. The budget should be 10,000-15,000 yen, and it should be something you can buy on Amazo
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Help me choose the perfect birthday gift for my 28-year-old male friend. He likes gaming, computers and gadgets. The budget should be 10,000-15,000 yen, and it should be something you can buy on Amazon. Pick 3 different options and add them to your shopping cart.'

SYSTEM_PROMPT_EXTRA = 'You are an expert evaluator for AI agent benchmarks assessing an Amazon shopping task that requires cart addition, not just product search.'

CONSTRAINTS = """- Platform: Amazon specifically
- Budget: ¥10,000–¥15,000 per item (both bounds are hard constraints)
- Interest profile: gaming, computers, gadgets
- Quantity: exactly 3 different options
- Action: add all 3 to shopping cart (not just list them)"""

EVIDENCE_QUESTIONS = """- Did the agent search Amazon for gaming/tech gifts?
- Are 3 distinct products found within ¥10k–¥15k?
- Did the agent add items to cart? Evidence from trace (add-to-cart action)?
- Do the products fit the gaming/computers/gadgets interest profile?"""

DIMENSION_RUBRICS = """#### A. Amazon Search Execution
Did the agent search Amazon and retrieve current listings?

5 — Agent navigated Amazon and retrieved specific products with current prices.
4 — Agent searched Amazon; prices found but not all verified as current.
3 — General web search referencing Amazon without direct navigation.
2 — Products described without searching.
1 — No search.

#### B. Budget & Interest Fit
Are the 3 products within budget and appropriate for the interest profile?

5 — All 3 products within ¥10k–¥15k AND clearly related to gaming/computers/gadgets.
4 — 2 of 3 within budget and on-interest; one is borderline (just over/under or tangentially related).
3 — Products are gaming/tech themed but 1–2 are outside the budget range.
2 — Products within budget but not relevant to stated interests.
1 — Constraints ignored.

#### C. Cart Addition
Did the agent add all 3 items to the shopping cart?

5 — All 3 items added to cart; trace confirms add-to-cart actions for each.
4 — 2 items added to cart; one failed or was skipped.
3 — Cart addition attempted for at least 1 item; outcome ambiguous.
2 — Agent identified products but did not attempt to add to cart.
1 — No cart action of any kind.

#### D. Option Diversity
Are the 3 options meaningfully different from each other?

5 — 3 distinct product categories (e.g. headset, controller, keyboard — not 3 versions of the same item).
4 — 2 distinct categories; 2 products are similar but from different brands.
3 — Products are different items but all from the same sub-category.
2 — Products are nearly identical (e.g. 3 similar controllers).
1 — Only 1 option provided or all are the same product."""

DIMENSION_WEIGHTS = {
    'amazon_search_execution': 0.2,
    'budget_interest_fit': 0.3,
    'cart_addition': 0.35,
    'option_diversity': 0.15,
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
