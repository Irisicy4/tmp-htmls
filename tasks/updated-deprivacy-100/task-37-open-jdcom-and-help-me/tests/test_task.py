"""LLM-as-judge evaluator for task-37-open-jdcom-and-help-me.

Category: Shopping
Task: Open JD.com and help me find the latest phones from Xiaomi, Huawei, vivo, and OPPO in the 3000–4000 yuan price range, and compare their performance.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Open JD.com and help me find the latest phones from Xiaomi, Huawei, vivo, and OPPO in the 3000–4000 yuan price range, and compare their performance.'

SYSTEM_PROMPT_EXTRA = 'You are an expert evaluator for AI agent benchmarks assessing a JD.com multi-brand phone search and comparison task.'

CONSTRAINTS = """- Platform: JD.com (京东) specifically
- Brands: Xiaomi, Huawei, vivo, OPPO — all four required
- Price range: ¥3,000–¥4,000 (both bounds hard constraints)
- Recency: "latest phones" — must be current/recent models, not discontinued
- Comparison: performance comparison required (CPU, camera, battery, display)"""

EVIDENCE_QUESTIONS = """- Did the agent navigate JD.com? Cite evidence.
- Were phones from all 4 brands found within ¥3k–¥4k?
- Are the models current/latest (not discontinued)?
- Is a performance comparison provided?"""

DIMENSION_RUBRICS = """#### A. JD.com Execution
Did the agent navigate JD.com as instructed?

5 — Agent navigated JD.com, searched by brand within price range, retrieved current listings.
4 — Agent accessed JD.com but search was partial (1–2 brands searched directly).
3 — Agent referenced JD.com from general search without navigating the site.
2 — Agent used a different platform (e.g. Tmall, Amazon) without acknowledgement.
1 — No search performed.

#### B. Brand & Price Compliance
Were phones from all 4 brands found within ¥3k–¥4k?

5 — Phones from all 4 brands found within ¥3k–¥4k with confirmed prices from JD.com.
4 — 3 brands covered within price range; one brand not found or price not verified.
3 — 2 brands covered; prices approximately correct.
2 — Only 1 brand covered, or price range not respected.
1 — Brand/price constraints ignored.

#### C. Performance Comparison
Is a meaningful performance comparison provided?

5 — Structured comparison across 3+ dimensions (CPU, camera, battery, display) for all 4 phones; clear winner or trade-off analysis.
4 — Comparison across 2 dimensions for most phones; trade-offs discussed.
3 — Specs listed per phone but no explicit comparison or ranking.
2 — Specs mentioned briefly without structured comparison.
1 — No performance comparison; just product names listed.

#### D. Result Currency
Are the models genuinely the latest available in that price range?

5 — Models are identifiably current-generation (e.g. 2024–2025 release); model numbers provided.
4 — Models appear current but release year not verified.
3 — Models may be current but agent did not verify recency.
2 — Models appear to be older generations.
1 — Models are clearly discontinued or fabricated."""

DIMENSION_WEIGHTS = {
    'jd_execution': 0.25,
    'brand_price_compliance': 0.3,
    'performance_comparison': 0.3,
    'result_currency': 0.15,
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
