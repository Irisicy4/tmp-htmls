"""LLM-as-judge evaluator for task-65-find-recommended-dinner-restaurants-in.

Category: Travel & Planning
Task: Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves finding dinner restaurant recommendations in Tuscany, Italy, with confirmed availability on specific dates (December 26-27).'

CONSTRAINTS = """- Location: Tuscany, Italy
- Meal type: dinner specifically
- Dates: both December 26th AND 27th — restaurants must be confirmed open on both days
- Source: must use a credible restaurant discovery platform (TripAdvisor, Google Maps, TheFork, etc.)"""

EVIDENCE_QUESTIONS = """- Did the agent search for restaurants in Tuscany, Italy?
- Were dinner-specific results returned?
- Were December 26 and 27 opening hours verified?
- How many restaurants were recommended?
- What platform was used?"""

DIMENSION_RUBRICS = """#### A. Search Execution (0.2)
Did the agent search for Tuscany restaurants on a credible platform?

5 — Used TripAdvisor, Google Maps, TheFork, or equivalent to search Tuscany, Italy dinner restaurants.
4 — Used a credible platform but search was less targeted.
3 — Found results via general web search without a restaurant platform.
2 — Used an unsuitable platform.
1 — No restaurant search performed.

#### B. Date Verification (0.35)
Were December 26 and 27 opening hours verified?

5 — Both dates explicitly verified as open for each recommended restaurant.
4 — Opening hours checked but one date not explicitly confirmed.
3 — Regular weekly hours checked without confirming those specific dates.
2 — Restaurants recommended without any date verification.
1 — No date verification.

#### C. Recommendation Quality (0.3)
Are the restaurant recommendations high-quality?

5 — 3+ restaurants with name, cuisine type, address, hours, and why recommended.
4 — 2-3 restaurants with most details.
3 — Restaurants named but details thin.
2 — Only one restaurant or very generic recommendations.
1 — No specific recommendations.

#### D. Dinner Focus (0.15)
Are recommendations specifically for dinner?

5 — All restaurants confirmed open for dinner with dinner service hours.
4 — Dinner service implied but not explicitly stated.
3 — Restaurants are general (lunch+dinner) without dinner-specific confirmation.
2 — Mix of dinner and non-dinner options.
1 — Not dinner-focused."""

DIMENSION_WEIGHTS = {
    'search_execution': 0.2,
    'date_verification': 0.35,
    'recommendation_quality': 0.3,
    'dinner_focus': 0.15,
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
