"""LLM-as-judge evaluator for task-88-search-online-to-find-information.

Category: Daily Activities
Task: Search online to find information about the current foot traffic and popularity of Jewel Changi Airport shopping mall in Singapore. Look for recent visitor reviews, social media posts, and any availab
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Search online to find information about the current foot traffic and popularity of Jewel Changi Airport shopping mall in Singapore. Look for recent visitor reviews, social media posts, and any available foot traffic data. Summarize your findings.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves searching for current or recent foot traffic information about Jewel Changi Airport shopping mall in Singapore.'

CONSTRAINTS = """- Location: Jewel Changi Airport, Singapore
- Data: foot traffic, visitor numbers, or crowd levels — current or very recent
- Sources: search across multiple platforms (social media, news, reviews, data sites)
- Output: clear assessment of current crowd levels"""

EVIDENCE_QUESTIONS = """- Did the agent search for Jewel Changi Airport foot traffic information?
- What sources were used (Google reviews, TripAdvisor, social media, news, etc.)?
- What specific data or impressions were found?
- Is the information recent?
- Is a clear assessment of crowd level provided?"""

DIMENSION_RUBRICS = """#### A. Search Execution (0.25)
Did the agent search for foot traffic at Jewel Changi Airport?

5 — Searched multiple platforms (Google reviews, TripAdvisor, Instagram, news) for Jewel Changi visitor info.
4 — Searched 2-3 sources.
3 — Only general web search without platform-specific sources.
2 — Described what to search without searching.
1 — No search.

#### B. Data Recency (0.3)
Is the information current or recent?

5 — Data from within the past week or clearly current (real-time crowd indicator or recent post).
4 — Data from past month.
3 — Data is recent but exact date unclear.
2 — Historical data without recency context.
1 — No temporal context.

#### C. Crowd Assessment (0.35)
Is a clear foot traffic assessment provided?

5 — Specific crowd level given (e.g. 'very busy, estimated X visitors on weekends', 'peak hours are...') with evidence.
4 — Good assessment with some evidence.
3 — General impression without specific data.
2 — 'It may be busy' without evidence.
1 — No assessment.

#### D. Source Quality (0.1)
Are credible sources cited?

5 — Sources named (Google reviews, TripAdvisor posts, news articles) with recency.
4 — Sources mentioned but vaguely.
3 — Single source only.
2 — Sources not cited.
1 — No sourcing."""

DIMENSION_WEIGHTS = {
    'search_execution': 0.25,
    'data_recency': 0.3,
    'crowd_assessment': 0.35,
    'source_quality': 0.1,
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
