"""LLM-as-judge evaluator for task-81-research-cheapest-shops-iphone-15-16-plus.

Category: Shopping
Task: Research the cheapest shops where you can buy the iPhone 15 Plus and iPhone 16 Plus under a 2-year return program.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Research the cheapest shops where you can buy the iPhone 15 Plus and iPhone 16 Plus under a 2-year return program.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves finding the cheapest prices for iPhone 15 Plus and iPhone 16 Plus in Japan under a 2-year return (残価設定型) program across carrier and retail options.'

CONSTRAINTS = """- Products: iPhone 15 Plus AND iPhone 16 Plus — both required
- Program: 2-year return/残価設定型 program (不要になったら返却できるプログラム)
- Market: Japan (based on language context)
- Comparison: multiple shops/carriers compared for each device"""

EVIDENCE_QUESTIONS = """- Did the agent search for both iPhone models under 2-year return programs?
- Which shops/carriers were checked (docomo, au, SoftBank, Apple, etc.)?
- What are the monthly or total prices for each model?
- Which shop was identified as cheapest for each model?"""

DIMENSION_RUBRICS = """#### A. Product Coverage (0.2)
Were both iPhone 15 Plus and 16 Plus covered?

5 — Both models researched with prices under 2-year return program.
4 — Both models covered but one less thoroughly.
3 — Only one model researched.
2 — Prices found but not specifically for 2-year return program.
1 — No product research.

#### B. Shop Comparison (0.3)
Were multiple shops compared?

5 — 4+ shops/carriers compared for each model (docomo, au, SoftBank, Apple, etc.).
4 — 3 shops compared.
3 — 2 shops compared.
2 — Only 1 shop.
1 — No comparison.

#### C. Price Accuracy (0.35)
Are prices accurate and program-specific?

5 — Monthly installment and total program cost clearly stated; 2-year return program confirmed at each carrier.
4 — Prices present but program type not always confirmed.
3 — Approximate prices without program-specific detail.
2 — General iPhone prices without return program context.
1 — No reliable prices.

#### D. Best Deal (0.15)
Was the cheapest option identified?

5 — Clear winner for each model with total cost breakdown.
4 — Best deal identified but without full cost breakdown.
3 — Best deal suggested without clear comparison basis.
2 — Prices listed without identifying best deal.
1 — No recommendation."""

DIMENSION_WEIGHTS = {
    'product_coverage': 0.2,
    'shop_comparison': 0.3,
    'price_accuracy': 0.35,
    'best_deal': 0.15,
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
