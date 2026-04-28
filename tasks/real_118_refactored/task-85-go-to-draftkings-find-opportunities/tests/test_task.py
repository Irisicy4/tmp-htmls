import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Go to DraftKings and find me opportunities to win money.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: DraftKings specifically (draftkings.com)
- Goal: identify actionable opportunities — not just explain how betting works
- Must be based on current live data (current odds, promotions, events)
- Should highlight specific bets or promotions with reasoning

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to DraftKings?
- What current sports events and odds were found?
- Were specific betting opportunities or promotions identified?
- Is the reasoning for each opportunity clear?
- Are the opportunities based on current data?

### Step 2: Dimension Scoring

#### A. Platform Navigation (0.35)
Did the agent navigate to DraftKings?

5 — Agent navigated to draftkings.com and accessed current odds/events.
4 — Agent reached DraftKings but with access difficulty.
3 — Agent found DraftKings odds via third-party aggregator.
2 — Agent described DraftKings without navigating.
1 — No DraftKings navigation.

#### B. Opportunity Identification (0.30)
Were specific betting opportunities identified?

5 — 3+ specific opportunities: sport, event, bet type, odds, and reasoning for value.
4 — 2-3 opportunities with most details.
3 — 1-2 opportunities with basic info.
2 — General categories of bets without specific events.
1 — No specific opportunities.

#### C. Data Currency (0.20)
Are opportunities based on current data?

5 — Odds and events are clearly current (today's games, live lines).
4 — Mostly current but some data may be slightly outdated.
3 — Current events but odds not confirmed current.
2 — Historical or generic odds used.
1 — No current data.

#### D. Reasoning Quality (0.15)
Is reasoning for each opportunity provided?

5 — Clear reasoning: why this bet has value (edge, line movement, promo, injury news).
4 — Reasoning present but shallow for some bets.
3 — Opportunities listed without reasoning.
2 — Generic 'this sport is popular' reasoning.
1 — No reasoning.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "opportunity_identification": "<one sentence citing specific evidence>",
    "data_currency": "<one sentence citing specific evidence>",
    "reasoning_quality": "<one sentence citing specific evidence>",
  }},
  "platform_navigation": <1-5>,
  "opportunity_identification": <1-5>,
  "data_currency": <1-5>,
  "reasoning_quality": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.35,
    "opportunity_identification": 0.30,
    "data_currency": 0.20,
    "reasoning_quality": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
