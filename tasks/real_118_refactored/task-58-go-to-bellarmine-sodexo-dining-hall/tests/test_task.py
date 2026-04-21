import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Go to https://bellarmine.sodexomyway.com/en-us/locations/university-dining-hall and tell me what is on the menu today. List all available food items for breakfast, lunch, and dinner.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: bellarmine.sodexomyway.com specifically
- Meals: breakfast, lunch, and dinner should all be covered
- Currency: today's menu — not a generic or sample menu
- Completeness: all food items listed, not just categories

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Bellarmine dining hall website?
- Was today's date menu retrieved (not a sample or generic menu)?
- Are breakfast, lunch, and dinner all covered?
- Are specific food items listed (not just categories)?

### Step 2: Dimension Scoring

#### A. Platform Navigation (0.25)
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
1 — No specific items.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_navigation": <1-5>,
  "menu_currency": <1-5>,
  "meal_coverage": <1-5>,
  "item_specificity": <1-5>,
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "menu_currency": "<one sentence citing specific evidence>",
    "meal_coverage": "<one sentence citing specific evidence>",
    "item_specificity": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.25,
    "menu_currency": 0.3,
    "meal_coverage": 0.25,
    "item_specificity": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
