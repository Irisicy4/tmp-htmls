"""LLM-as-judge evaluator for task-90-sneaker-reseller-stockx-or-goat-compare.

Category: Shopping
Task: I am a sneaker reseller. Should I choose StockX or GOAT? Please research and compare the two platforms and give me a recommendation.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'I am a sneaker reseller. Should I choose StockX or GOAT? Please research and compare the two platforms and give me a recommendation.'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves researching and comparing StockX and GOAT sneaker resale platforms from a seller's perspective to give a concrete recommendation."

CONSTRAINTS = """- Perspective: sneaker reseller (seller, not buyer) — fees, payout speed, authentication, seller protections
- Platforms: StockX AND GOAT — both must be covered
- Recommendation: must give a clear recommendation with reasoning
- Data: current fee structures and platform features"""

EVIDENCE_QUESTIONS = """- Did the agent research both platforms?
- What seller fees were found for each platform?
- Were payout timelines, authentication processes, and seller protections compared?
- Was a clear recommendation made with reasoning?"""

DIMENSION_RUBRICS = """#### A. Platform Research (0.25)
Did the agent research both platforms?

5 — Both StockX and GOAT researched with current seller fee data from official sources.
4 — Both researched but one less thoroughly.
3 — Both covered but data may be outdated.
2 — Only one platform researched.
1 — No research.

#### B. Comparison Depth (0.35)
Were relevant seller factors compared?

5 — Fees, payout speed, authentication process, seller protections, geographic availability, and user base all compared.
4 — 4-5 factors compared.
3 — 2-3 factors compared.
2 — Only fees compared.
1 — No comparison.

#### C. Data Accuracy (0.25)
Is the comparison data accurate?

5 — Fee percentages match current published rates (StockX ~9-10%, GOAT ~15-25% depending on tier); other facts verifiable.
4 — Mostly accurate with minor discrepancies.
3 — Approximately correct but imprecise.
2 — Significant inaccuracies.
1 — Clearly wrong data.

#### D. Recommendation Quality (0.15)
Is the recommendation clear and justified?

5 — Clear recommendation with specific reasoning tailored to a reseller's priorities.
4 — Recommendation made but reasoning is somewhat generic.
3 — Balanced 'it depends' without a clear winner.
2 — Recommendation without justification.
1 — No recommendation."""

DIMENSION_WEIGHTS = {
    'platform_research': 0.25,
    'comparison_depth': 0.35,
    'data_accuracy': 0.25,
    'recommendation_quality': 0.15,
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
