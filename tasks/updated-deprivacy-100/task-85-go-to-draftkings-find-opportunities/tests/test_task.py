"""LLM-as-judge evaluator for task-85-go-to-draftkings-find-opportunities.

Category: Finance & Economics
Task: Go to DraftKings and find me opportunities to win money.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Go to DraftKings and find me opportunities to win money.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves navigating DraftKings sportsbook to identify current betting opportunities, favorable odds, or promotions that offer value to a bettor.'

CONSTRAINTS = """- Platform: DraftKings specifically (draftkings.com)
- Goal: identify actionable opportunities — not just explain how betting works
- Must be based on current live data (current odds, promotions, events)
- Should highlight specific bets or promotions with reasoning"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to DraftKings?
- What current sports events and odds were found?
- Were specific betting opportunities or promotions identified?
- Is the reasoning for each opportunity clear?
- Are the opportunities based on current data?"""

DIMENSION_RUBRICS = """#### A. Platform Navigation (0.2)
Did the agent navigate to DraftKings?

5 — Agent navigated to draftkings.com and accessed current odds/events.
4 — Agent reached DraftKings but with access difficulty.
3 — Agent found DraftKings odds via third-party aggregator.
2 — Agent described DraftKings without navigating.
1 — No DraftKings navigation.

#### B. Opportunity Identification (0.35)
Were specific betting opportunities identified?

5 — 3+ specific opportunities: sport, event, bet type, odds, and reasoning for value.
4 — 2-3 opportunities with most details.
3 — 1-2 opportunities with basic info.
2 — General categories of bets without specific events.
1 — No specific opportunities.

#### C. Data Currency (0.25)
Are opportunities based on current data?

5 — Odds and events are clearly current (today's games, live lines).
4 — Mostly current but some data may be slightly outdated.
3 — Current events but odds not confirmed current.
2 — Historical or generic odds used.
1 — No current data.

#### D. Reasoning Quality (0.2)
Is reasoning for each opportunity provided?

5 — Clear reasoning: why this bet has value (edge, line movement, promo, injury news).
4 — Reasoning present but shallow for some bets.
3 — Opportunities listed without reasoning.
2 — Generic 'this sport is popular' reasoning.
1 — No reasoning."""

DIMENSION_WEIGHTS = {
    'platform_navigation': 0.2,
    'opportunity_identification': 0.35,
    'data_currency': 0.25,
    'reasoning_quality': 0.2,
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
