"""LLM-as-judge evaluator for task-75-github-project-claims-5000-stars-star-inflation.

Category: Software Engineering
Task: This GitHub project claims to have reached 5,000 stars in 48 hours. I'm certain the data is fake. Please compile a record of star inflation and also analyze the forks from the same angle.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "This GitHub project claims to have reached 5,000 stars in 48 hours. I'm certain the data is fake. Please compile a record of star inflation and also analyze the forks from the same angle."

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves investigating a GitHub repository for signs of artificial star and fork inflation, using GitHub's API or tools like Star History to detect suspicious growth patterns."

CONSTRAINTS = """- Target: a specific GitHub repository (must be identified from context or searched)
- Analysis: star growth timeline — look for unnatural spikes, bot-like patterns
- Forks: same analysis applied to fork count
- Evidence: must use data (star history charts, API data, timestamps) not just assertion"""

EVIDENCE_QUESTIONS = """- Did the agent access GitHub star history data for the repository?
- What growth patterns were found (spikes, timing, geographic concentration)?
- Was fork analysis also performed?
- What specific evidence of inflation was identified?
- Were external tools (star-history.com, GitHub API) used?"""

DIMENSION_RUBRICS = """#### A. Star History Analysis (0.35)
Did the agent analyze star growth timeline?

5 — Star growth timeline retrieved (via star-history.com or GitHub API) showing specific dates, volumes, and suspicious spikes.
4 — Star growth analyzed but with less precise data.
3 — Growth pattern described without timeline data.
2 — Agent asserted inflation without data.
1 — No star history analysis.

#### B. Inflation Evidence (0.3)
Was specific evidence of star inflation identified?

5 — Specific red flags cited: unnatural spike timing, accounts with no other activity, geographic clustering, similar account creation dates.
4 — 2-3 specific indicators identified.
3 — 1 specific indicator identified.
2 — Vague inflation claims without evidence.
1 — No inflation evidence.

#### C. Fork Analysis (0.25)
Was fork analysis also performed?

5 — Fork count and growth pattern analyzed; empty/inactive forks identified as additional evidence.
4 — Forks analyzed but less thoroughly than stars.
3 — Forks mentioned but not analyzed.
2 — Fork analysis described without data.
1 — No fork analysis.

#### D. Report Quality (0.1)
Is the report well-organized and evidence-based?

5 — Structured report with timeline, evidence, and conclusion; data cited.
4 — Good report but organization could be clearer.
3 — Information present but loosely organized.
2 — Disorganized or hard to follow.
1 — No report structure."""

DIMENSION_WEIGHTS = {
    'star_history_analysis': 0.35,
    'inflation_evidence': 0.3,
    'fork_analysis': 0.25,
    'report_quality': 0.1,
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
