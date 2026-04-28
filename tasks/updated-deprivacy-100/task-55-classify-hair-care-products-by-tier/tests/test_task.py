"""LLM-as-judge evaluator for task-55-classify-hair-care-products-by-tier.

Category: Shopping
Task: Classify hair care products on the market by their ingredient lists and rank them by tier. Then analyze the tier of the shampoo shown in the image based on its ingredients.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Classify hair care products on the market by their ingredient lists and rank them by tier. Then analyze the tier of the shampoo shown in the image based on its ingredients.'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent produced a credible ingredient-based tier classification of hair care products and correctly analyzed a specific shampoo.'

CONSTRAINTS = """- Tier classification must be based on actual ingredient analysis (not brand prestige)
- Must produce at least 3 tiers (e.g. Basic, Mid-range, Premium)
- Must analyze the specific shampoo from the image
- Analysis must reference specific ingredients and explain their tier implications"""

EVIDENCE_QUESTIONS = """- Did the agent research ingredient-based tier classification systems?
- How many tiers were defined and what criteria distinguish them?
- Did the agent analyze the specific shampoo from the image?
- What ingredients were identified and what tier was assigned?"""

DIMENSION_RUBRICS = """#### A. Tier Framework (0.3)
Did the agent produce a credible ingredient-based tier classification?

5 — Clear 3+ tier framework with specific ingredient criteria for each tier (e.g. sulfate-free, silicone types, active ingredients).
4 — Good framework but one tier is poorly defined.
3 — Basic tiers present but criteria are vague or brand-based rather than ingredient-based.
2 — Only 2 tiers or very simplistic classification.
1 — No tier framework.

#### B. Ingredient Knowledge (0.25)
Does the agent demonstrate knowledge of hair care ingredients?

5 — References specific ingredients by name (surfactants, conditioning agents, actives) and explains their quality indicators.
4 — Good ingredient knowledge but incomplete coverage.
3 — Basic ingredient knowledge — knows good vs bad but lacks specifics.
2 — Very generic ingredient discussion.
1 — No ingredient knowledge demonstrated.

#### C. Product Analysis (0.35)
Did the agent analyze the specific shampoo and assign it a tier?

5 — Specific shampoo analyzed with ingredient-by-ingredient breakdown and clear tier assignment with reasoning.
4 — Tier assigned with some ingredient analysis but not comprehensive.
3 — Tier assigned but reasoning is thin.
2 — Tier mentioned without specific ingredient analysis.
1 — No product analysis.

#### D. Output Structure (0.1)
Is the output well-organized?

5 — Clear structure: tier framework first, then product analysis, with consistent formatting.
4 — Good structure but minor organizational issues.
3 — Content present but hard to navigate.
2 — Wall of text.
1 — No structure."""

DIMENSION_WEIGHTS = {
    'tier_framework': 0.3,
    'ingredient_knowledge': 0.25,
    'product_analysis': 0.35,
    'output_structure': 0.1,
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
