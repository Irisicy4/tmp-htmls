"""LLM-as-judge evaluator for task-01-im-looking-for-backpack-under.

Category: Shopping
Task: im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices,
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices, comparing them to the original.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based shopping research task.'

CONSTRAINTS = """- Budget ceiling: $75 per item (hard constraint — any option above this is a failure)
- Required quantity: 3 to 5 backpack options (not fewer, not more)
- Reference product: https://www.amazon.com/dp/B09YRC9Y3G (agent must have researched this product's features)
- Required output components: (a) key features per option, (b) price per option, (c) comparison to the original product"""

EVIDENCE_QUESTIONS = """Carefully read the agent response and trace. Then answer each of the following questions with a direct observation from the response — do not infer or assume:

- How many backpack options did the agent present? List them briefly.
- What prices were mentioned? List all dollar amounts found.
- Did any price exceed $75? If so, which?
- Did the agent reference or research the original Amazon product (B09YRC9Y3G)?
- Did the agent compare candidates to the original? Quote or describe the comparison language used.
- Were key features (e.g. capacity, material, straps, pockets) described for each option?"""

DIMENSION_RUBRICS = """Score each dimension from 1 to 5 using the rubrics below. Each score must be justified by specific evidence from Step 1.

#### A. Constraint Satisfaction
Did the agent respect all hard constraints (budget and quantity)?

5 — All 3–5 options are explicitly priced at or below $75; quantity is exactly 3–5.
4 — All prices are at or below $75; quantity is 3–5 but one option is missing a price.
3 — Quantity is 3–5 but 1 option has an ambiguous or missing price; no option clearly exceeds $75.
2 — Quantity is outside 3–5 (e.g. only 2 options, or 6+), OR 1 option exceeds $75.
1 — Multiple options exceed $75, OR fewer than 2 options presented, OR task was abandoned.

#### B. Result Specificity
Are the options real, specific products (with names, links, or identifiable details) rather than vague suggestions?

5 — All 3–5 options have: product name, specific price, and at least one verifiable detail (ASIN, URL, brand+model).
4 — Most options (3+) have name and price; 1–2 are missing a verifiable identifier.
3 — Options are named but lack prices or identifiers for 2+ of them; still clearly distinct products.
2 — Options are described generically (e.g. "a hiking backpack from Amazon") without specific names or prices.
1 — No specific products identified; agent only described a search or listed categories.

#### C. Comparison Quality
Did the agent meaningfully compare candidates to the original product?

5 — Each option is explicitly compared to the original on 2+ specific features (e.g. capacity, material, pockets); comparison is structured and actionable.
4 — Most options (3+) are compared to the original on at least 1 specific feature; comparison is present but uneven.
3 — A general comparison summary exists (e.g. "similar features to the original") but feature-by-feature breakdown is missing or shallow.
2 — Comparison language is present (e.g. "vs", "similar") but no actual feature comparison is made.
1 — No comparison to the original product at all.

#### D. Task Completeness
Did the agent address every component of the request?

5 — All four required components present for all options: key features, price, comparison, and the original product was researched.
4 — All components present for most options; 1 minor gap (e.g. one option missing a feature description).
3 — 2–3 components present consistently; one component (e.g. original product research, or comparison) is missing or superficial.
2 — Only 1–2 components addressed; significant parts of the task are skipped.
1 — Agent failed to complete the task or provided an off-topic response."""

DIMENSION_WEIGHTS = {
    'constraint_satisfaction': 0.35,
    'result_specificity': 0.25,
    'comparison_quality': 0.25,
    'task_completeness': 0.15,
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
