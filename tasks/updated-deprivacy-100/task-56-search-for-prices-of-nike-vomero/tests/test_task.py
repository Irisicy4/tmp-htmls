"""LLM-as-judge evaluator for task-56-search-for-prices-of-nike-vomero.

Category: Shopping
Task: Search for prices of the Nike Vomero Plus sneaker in Brazil and compare prices across stores to find the best deal.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Search for prices of the Nike Vomero Plus sneaker in Brazil and compare prices across stores to find the best deal.'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent successfully found and compared Nike Vomero Plus prices across Brazilian retailers.'

CONSTRAINTS = """- Product: Nike Vomero Plus specifically (not Air Vomero or other models)
- Region: Brazil — prices should be in BRL
- Multi-store comparison: at least 3 stores should be compared
- Best deal: agent must identify which store has the lowest price"""

EVIDENCE_QUESTIONS = """- Did the agent search for Nike Vomero Plus in Brazil?
- How many stores were compared?
- What prices were found and in what currency?
- Which store was identified as having the best deal?
- Were prices from Google Shopping or direct retailer pages?"""

DIMENSION_RUBRICS = """#### A. Search Execution (0.2)
Did the agent search for Nike Vomero Plus in Brazil?

5 — Agent used Google Shopping Brazil or visited Brazilian retailer sites directly.
4 — Agent searched but used general web search rather than shopping platforms.
3 — Agent found some prices but search was not Brazil-specific.
2 — Agent described what to search without actually searching.
1 — No search performed.

#### B. Store Coverage (0.25)
How many stores were compared?

5 — 4 or more Brazilian stores compared with specific prices.
4 — 3 stores compared.
3 — 2 stores compared.
2 — Only 1 store found.
1 — No stores compared.

#### C. Price Accuracy (0.35)
Are prices accurate and in BRL?

5 — Prices in BRL from credible Brazilian retailers (Netshoes, Nike.com.br, Centauro, etc.) with specific amounts.
4 — Prices in BRL but from less authoritative sources.
3 — Prices found but currency or accuracy uncertain.
2 — Prices found but in USD or without clear source.
1 — No actual prices found.

#### D. Recommendation Quality (0.2)
Did the agent identify the best deal clearly?

5 — Clear winner identified with price, store name, and any relevant conditions (shipping, installments).
4 — Best deal identified but missing one detail.
3 — Best deal mentioned vaguely.
2 — Prices listed without identifying best deal.
1 — No recommendation."""

DIMENSION_WEIGHTS = {
    'search_execution': 0.2,
    'store_coverage': 0.25,
    'price_accuracy': 0.35,
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
