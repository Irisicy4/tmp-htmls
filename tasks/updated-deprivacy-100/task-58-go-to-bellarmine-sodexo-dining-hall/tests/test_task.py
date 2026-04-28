"""LLM-as-judge evaluator for task-58-go-to-bellarmine-sodexo-dining-hall.

Category: Daily Activities
Task: Go to https://bellarmine.sodexomyway.com/en-us/locations/university-dining-hall and tell me what is on the menu today. List all available food items for breakfast, lunch, and dinner.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Go to https://bellarmine.sodexomyway.com/en-us/locations/university-dining-hall and tell me what is on the menu today. List all available food items for breakfast, lunch, and dinner.'

SYSTEM_PROMPT_EXTRA = "Assess whether an AI agent successfully retrieved and reported today's cafeteria menu from the Bellarmine University dining hall website."

CONSTRAINTS = """- Platform: bellarmine.sodexomyway.com specifically
- Meals: breakfast, lunch, and dinner should all be covered
- Currency: today's menu — not a generic or sample menu
- Completeness: all food items listed, not just categories"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to the Bellarmine dining hall website?
- Was today's date menu retrieved (not a sample or generic menu)?
- Are breakfast, lunch, and dinner all covered?
- Are specific food items listed (not just categories)?"""

DIMENSION_RUBRICS = """#### A. Platform Navigation (0.25)
Did the agent navigate to the correct website?

5 — Agent navigated to bellarmine.sodexomyway.com and accessed today's menu.
4 — Agent reached the site but had difficulty loading the menu.
3 — Agent found Bellarmine dining info from a different source.
2 — Agent described what the menu site looks like without navigating.
1 — No navigation attempted.

#### B. Menu Currency (0.3)
Is the menu for today specifically?

5 — Menu clearly identified as today's date with specific items.
4 — Menu retrieved but date not explicitly confirmed.
3 — Menu present but may be a sample or generic menu.
2 — Generic dining hall info without today's specific items.
1 — No menu retrieved.

#### C. Meal Coverage (0.25)
Are all three meal periods covered?

5 — Breakfast, lunch, and dinner all listed with specific items.
4 — Two meal periods covered.
3 — One meal period covered.
2 — Meal periods mentioned but no specific items.
1 — No meal coverage.

#### D. Item Specificity (0.2)
Are specific food items listed?

5 — Specific named dishes listed for each meal (e.g. 'scrambled eggs, bacon, oatmeal').
4 — Most items specific but some categories without detail.
3 — Mix of specific items and categories.
2 — Only food categories listed (e.g. 'breakfast bar, grill').
1 — No specific items."""

DIMENSION_WEIGHTS = {
    'platform_navigation': 0.25,
    'menu_currency': 0.3,
    'meal_coverage': 0.25,
    'item_specificity': 0.2,
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
