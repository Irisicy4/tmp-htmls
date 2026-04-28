import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: Tuscany, Italy
- Meal type: dinner specifically
- Dates: both December 26th AND 27th — restaurants must be confirmed open on both days
- Source: must use a credible restaurant discovery platform (TripAdvisor, Google Maps, TheFork, etc.)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for restaurants in Tuscany, Italy?
- Were dinner-specific results returned?
- Were December 26 and 27 opening hours verified?
- How many restaurants were recommended?
- What platform was used?

### Step 2: Dimension Scoring

#### A. Search Execution (0.35)
Did the agent search for Tuscany restaurants on a credible platform?

5 — Used TripAdvisor, Google Maps, TheFork, or equivalent to search Tuscany, Italy dinner restaurants.
4 — Used a credible platform but search was less targeted.
3 — Found results via general web search without a restaurant platform.
2 — Used an unsuitable platform.
1 — No restaurant search performed.

#### B. Date Verification (0.28)
Were December 26 and 27 opening hours verified?

5 — Both dates explicitly verified as open for each recommended restaurant.
4 — Opening hours checked but one date not explicitly confirmed.
3 — Regular weekly hours checked without confirming those specific dates.
2 — Restaurants recommended without any date verification.
1 — No date verification.

#### C. Recommendation Quality (0.24)
Are the restaurant recommendations high-quality?

5 — 3+ restaurants with name, cuisine type, address, hours, and why recommended.
4 — 2-3 restaurants with most details.
3 — Restaurants named but details thin.
2 — Only one restaurant or very generic recommendations.
1 — No specific recommendations.

#### D. Dinner Focus (0.13)
Are recommendations specifically for dinner?

5 — All restaurants confirmed open for dinner with dinner service hours.
4 — Dinner service implied but not explicitly stated.
3 — Restaurants are general (lunch+dinner) without dinner-specific confirmation.
2 — Mix of dinner and non-dinner options.
1 — Not dinner-focused.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "date_verification": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
    "dinner_focus": "<one sentence citing specific evidence>",
  }},
  "search_execution": <1-5>,
  "date_verification": <1-5>,
  "recommendation_quality": <1-5>,
  "dinner_focus": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "search_execution": 0.35,
    "date_verification": 0.28,
    "recommendation_quality": 0.24,
    "dinner_focus": 0.13,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
