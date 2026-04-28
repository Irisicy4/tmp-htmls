"""LLM-as-judge evaluator for task-43-bought-crate-of-clemenules-clementines.

Category: Daily Activities
Task: I bought a crate of Clemenules clementines and they all had seeds—shouldn't this variety be seedless? Can someone confirm whether Clemenules are normally seedless?
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "I bought a crate of Clemenules clementines and they all had seeds—shouldn't this variety be seedless? Can someone confirm whether Clemenules are normally seedless?"

SYSTEM_PROMPT_EXTRA = 'This is a factual horticultural Q&A task. There is a clear correct answer: Clemenules are normally seedless BUT can develop seeds through cross-pollination with compatible citrus nearby. The agent should confirm seedlessness as the norm and explain the exception.'

CONSTRAINTS = """- This has a verifiable answer: Clemenules (Clementine Nules) are normally seedless/parthenocarpic
- Exception: seeds develop when Clemenules are grown near compatible citrus (bees cross-pollinate)
- The agent should: (1) confirm seedlessness as normal, (2) explain why seeds may have occurred, (3) cite credible horticultural source
- This does not require browser navigation — the agent can answer from knowledge, but should verify or cite a source"""

EVIDENCE_QUESTIONS = """- Did the agent confirm Clemenules are normally seedless?
- Did the agent explain the cross-pollination exception?
- Was a credible source cited (agriculture extension, citrus grower association, horticultural database)?
- Is the answer clear and directly responsive to the user's concern?"""

DIMENSION_RUBRICS = """#### A. Factual Accuracy
Is the core factual answer correct?

5 — Agent correctly states: (a) Clemenules are normally seedless, AND (b) seeds occur due to cross-pollination with nearby compatible citrus.
4 — Agent correctly states seedlessness as normal; cross-pollination explanation is vague or partial.
3 — Agent states Clemenules are normally seedless without explaining why the user's batch had seeds.
2 — Agent is uncertain or gives a hedged answer without confirming seedlessness.
1 — Agent gives wrong answer (e.g. states Clemenules do have seeds normally) or refuses to answer.

#### B. Explanation Quality
Is the explanation of the seeded exception clear and accurate?

5 — Clear explanation: Clemenules are parthenocarpic; seeds develop when bees cross-pollinate with mandarin, tangerine, or other compatible citrus planted nearby; commercial orchards control for this.
4 — Cross-pollination mentioned as cause; parthenocarpy or commercial control not explained.
3 — Agent mentions that conditions can cause seeds but explanation is vague.
2 — Agent says "sometimes they have seeds" without mechanistic explanation.
1 — No explanation of the seeded exception.

#### C. Source Credibility
Did the agent cite or reference a credible horticultural source?

5 — Specific source cited: citrus variety database, agricultural extension service (e.g. UC Riverside Citrus Variety Collection), grower association, or peer-reviewed horticulture reference.
4 — Source type mentioned but not specifically identified.
3 — Agent indicated it checked a source but no citation given.
2 — No source; answer from prior knowledge only.
1 — Fabricated or irrelevant source.

#### D. Practical Guidance
Did the agent address the user's underlying concern (were their clementines normal or defective)?

5 — Agent directly addresses user's concern: confirms their experience is a known exception (not fraud or mislabelling); may suggest checking whether the orchard had other citrus nearby.
4 — User's concern addressed; suggestion for next steps not provided.
3 — Factual answer given but user's specific situation not directly addressed.
2 — Generic information without addressing whether the user's purchase was normal.
1 — No practical guidance."""

DIMENSION_WEIGHTS = {
    'factual_accuracy': 0.4,
    'explanation_quality': 0.3,
    'source_credibility': 0.15,
    'practical_guidance': 0.15,
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
