import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Search online to find information about the current foot traffic and popularity of Jewel Changi Airport shopping mall in Singapore. Look for recent visitor reviews, social media posts, and any available foot traffic data. Summarize your findings.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: Jewel Changi Airport, Singapore
- Data: foot traffic, visitor numbers, or crowd levels — current or very recent
- Sources: search across multiple platforms (social media, news, reviews, data sites)
- Output: clear assessment of current crowd levels

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Jewel Changi Airport foot traffic information?
- What sources were used (Google reviews, TripAdvisor, social media, news, etc.)?
- What specific data or impressions were found?
- Is the information recent?
- Is a clear assessment of crowd level provided?

### Step 2: Dimension Scoring

#### A. Search Execution (0.35)
Did the agent search for foot traffic at Jewel Changi Airport?

5 — Searched multiple platforms (Google reviews, TripAdvisor, Instagram, news) for Jewel Changi visitor info.
4 — Searched 2-3 sources.
3 — Only general web search without platform-specific sources.
2 — Described what to search without searching.
1 — No search.

#### B. Data Recency (0.26)
Is the information current or recent?

5 — Data from within the past week or clearly current (real-time crowd indicator or recent post).
4 — Data from past month.
3 — Data is recent but exact date unclear.
2 — Historical data without recency context.
1 — No temporal context.

#### C. Crowd Assessment (0.3)
Is a clear foot traffic assessment provided?

5 — Specific crowd level given (e.g. 'very busy, estimated X visitors on weekends', 'peak hours are...') with evidence.
4 — Good assessment with some evidence.
3 — General impression without specific data.
2 — 'It may be busy' without evidence.
1 — No assessment.

#### D. Source Quality (0.09)
Are credible sources cited?

5 — Sources named (Google reviews, TripAdvisor posts, news articles) with recency.
4 — Sources mentioned but vaguely.
3 — Single source only.
2 — Sources not cited.
1 — No sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "data_recency": "<one sentence citing specific evidence>",
    "crowd_assessment": "<one sentence citing specific evidence>",
    "source_quality": "<one sentence citing specific evidence>",
  }},
  "search_execution": <1-5>,
  "data_recency": <1-5>,
  "crowd_assessment": <1-5>,
  "source_quality": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "search_execution": 0.35,
    "data_recency": 0.26,
    "crowd_assessment": 0.3,
    "source_quality": 0.09,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
