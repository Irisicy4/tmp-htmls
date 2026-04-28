import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Collect and analyze the top YouTube Shorts accounts most watched by people aged 50 and above in Korea. I want to create a new channel, so please gather very detailed information.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: YouTube Shorts specifically (not general YouTube)
- Demographic: viewers aged 50+ in Korea
- Depth: detailed channel info — subscriber count, posting frequency, content style, engagement
- Goal: inform creation of a new channel — must include strategic insights

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Korean YouTube Shorts channels popular with 50+ viewers?
- What specific channels were identified (names, subscriber counts, content types)?
- Was engagement data (views, likes) collected?
- Are content strategy insights provided?
- Are recommendations for a new channel included?

### Step 2: Dimension Scoring

#### A. Channel Research (0.3)
Did the agent find specific, relevant Korean YouTube Shorts channels?

5 — 5+ specific channels identified with names, subscriber counts, and content focus targeting 50+ Koreans.
4 — 3-4 channels with most details.
3 — 1-2 channels or channels without demographic focus.
2 — General YouTube Shorts landscape without Korea/50+ focus.
1 — No channel research.

#### B. Data Depth (0.25)
Was detailed channel data collected?

5 — For each channel: subscribers, avg views per short, posting frequency, content themes, engagement rate.
4 — Most metrics present but one or two missing.
3 — Basic data (name + subscriber count) only.
2 — Very surface-level data.
1 — No channel data.

#### C. Demographic Relevance (0.25)
Is content specifically relevant to the 50+ Korean demographic?

5 — Clear evidence that identified channels target 50+ Koreans (content topics, language, cultural references).
4 — Channels plausibly relevant but demographic targeting not confirmed.
3 — Korean channels but not specifically 50+ focused.
2 — Generic popular channels without demographic filter.
1 — No demographic relevance.

#### D. Strategic Insights (0.2)
Are actionable insights for a new channel provided?

5 — Specific recommendations: content niches, posting schedule, style, hooks that work for the demographic.
4 — Some strategic guidance but not fully actionable.
3 — Generic advice not grounded in the research.
2 — Research presented without strategic conclusions.
1 — No strategic insights.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "channel_research": "<one sentence citing specific evidence>",
    "data_depth": "<one sentence citing specific evidence>",
    "demographic_relevance": "<one sentence citing specific evidence>",
    "strategic_insights": "<one sentence citing specific evidence>",
  }},
  "channel_research": <1-5>,
  "data_depth": <1-5>,
  "demographic_relevance": <1-5>,
  "strategic_insights": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "channel_research": 0.3,
    "data_depth": 0.25,
    "demographic_relevance": 0.25,
    "strategic_insights": 0.2,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
