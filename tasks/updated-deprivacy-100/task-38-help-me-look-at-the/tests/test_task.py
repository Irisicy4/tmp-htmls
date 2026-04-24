"""LLM-as-judge evaluator for task-38-help-me-look-at-the.

Category: Shopping
Task: Help me look at the price and configuration of Mac Mini, Mac Studio, MacBook Pro and MacBook Air, and give a trade-off for choosing. I want to compare their price and performance and find the one with
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Help me look at the price and configuration of Mac Mini, Mac Studio, MacBook Pro and MacBook Air, and give a trade-off for choosing. I want to compare their price and performance and find the one with the best value.'

SYSTEM_PROMPT_EXTRA = 'This is an Apple product comparison and value assessment task covering 4 product lines.'

CONSTRAINTS = """- Products: Mac Mini, Mac Studio, MacBook Pro, MacBook Air — all 4 required
- Data needed: current prices AND key specs (chip, RAM, storage, GPU, display)
- Analysis required: trade-off comparison AND best-value recommendation
- Source: should reference apple.com or other authoritative pricing"""

EVIDENCE_QUESTIONS = """- Were all 4 Apple products covered?
- Are current prices and specs provided for each?
- Is a trade-off analysis (not just specs listing) present?
- Is a best-value recommendation made with justification?"""

DIMENSION_RUBRICS = """#### A. Product Coverage
Are all 4 products covered with price and spec data?

5 — All 4 products covered with current base price and key specs (chip, RAM, storage minimum).
4 — 3 of 4 covered fully; one has partial info.
3 — All 4 mentioned but 1–2 lack specific prices or specs.
2 — Only 2–3 products covered with data.
1 — Fewer than 2 products or no data.

#### B. Spec & Price Accuracy
Is the information accurate and current?

5 — Prices and specs are consistent with current Apple lineup (M4/M4 Pro/M4 Max chips as of 2025–2026); no clearly wrong claims.
4 — Mostly accurate; 1–2 minor errors or slightly outdated specs.
3 — Generally correct direction; some specs are vague or approximate.
2 — Significant inaccuracies (wrong chip generation, wrong price tier).
1 — Fabricated or clearly wrong.

#### C. Trade-off Analysis
Does the agent provide genuine trade-off analysis (not just specs listing)?

5 — Explicit trade-offs discussed: portability vs performance, desktop vs laptop, value for money per use case (e.g. creative professionals vs students vs home users).
4 — Trade-offs discussed for 2–3 dimension pairs; one use-case segment not addressed.
3 — Implicit comparison (e.g. "Mac Studio is faster but costs more") without structured trade-off framework.
2 — Products listed side by side without any trade-off commentary.
1 — No trade-off analysis.

#### D. Value Recommendation
Does the agent make a specific best-value recommendation with justification?

5 — Clear recommendation with stated assumption (e.g. "for most users, Mac Mini M4 offers best value because..."); alternative for power users noted.
4 — Recommendation made with brief justification; alternatives not discussed.
3 — Agent suggests a product without clear justification.
2 — Multiple options suggested without a clear recommendation.
1 — No recommendation."""

DIMENSION_WEIGHTS = {
    'product_coverage': 0.2,
    'spec_price_accuracy': 0.25,
    'tradeoff_analysis': 0.35,
    'value_recommendation': 0.2,
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
