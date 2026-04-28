"""LLM-as-judge evaluator for task-35-on-linkedin-find-the-top.

Category: Marketing & Analytics
Task: On LinkedIn, find the top 30 US design agencies and browse the latest 10 posts each has published. Provide a report that summarizes their posted content and engagement on those posts.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'On LinkedIn, find the top 30 US design agencies and browse the latest 10 posts each has published. Provide a report that summarizes their posted content and engagement on those posts.'

SYSTEM_PROMPT_EXTRA = 'This is a large-scale LinkedIn research task: 30 agencies × 10 posts = 300 posts. Full completion is exceptional. Apply graduated credit based on actual coverage achieved.'

CONSTRAINTS = """- Platform: LinkedIn specifically
- Scope: top 30 US design agencies (agent must identify and justify which 30)
- Per-agency: latest 10 posts each
- Report must include: (1) summary of posted content per agency, (2) engagement metrics (likes, comments, shares) per post or per agency
- Scale: 300 posts total; partial completion should receive graduated credit"""

EVIDENCE_QUESTIONS = """- Did the agent navigate LinkedIn? Cite trace evidence.
- How many agencies were identified? Are they recognisable US design firms?
- For how many agencies were posts actually browsed?
- Are engagement metrics (likes/comments) reported for any posts?
- Is a structured report produced?"""

DIMENSION_RUBRICS = """#### A. LinkedIn Execution
Did the agent actively navigate LinkedIn as instructed?

5 — Agent navigated LinkedIn, searched for design agencies, and browsed company pages/posts.
4 — Agent accessed LinkedIn but was blocked by login walls or rate limits; navigated partially.
3 — Agent described LinkedIn approach but only retrieved limited data (e.g. from LinkedIn search snippets).
2 — Agent used non-LinkedIn sources (e.g. agency websites, general web) instead.
1 — No LinkedIn navigation performed.

#### B. Agency Coverage
How many of the target 30 US design agencies are covered?

5 — 20+ recognisable US design agencies identified and included in report.
4 — 10–19 agencies covered with specific post data.
3 — 5–9 agencies covered.
2 — 2–4 agencies covered.
1 — Fewer than 2 agencies or agencies are not specifically US design firms.

#### C. Post Content Summary
Are the agencies' post topics and themes summarised?

5 — For 10+ agencies: content themes summarised per agency (e.g. "Agency X posts about brand identity projects and case studies").
4 — Content summarised for 5–9 agencies with themes identified.
3 — Content summarised for 2–4 agencies; others described generically.
2 — Only 1 agency's content described; or summaries are too vague to be useful.
1 — No content summary; just agency names listed.

#### D. Engagement Reporting
Are engagement metrics reported?

5 — Likes/comments/shares reported for posts from 5+ agencies; average or total engagement noted.
4 — Engagement reported for 2–4 agencies with specific numbers.
3 — Engagement mentioned for 1–2 agencies, or described qualitatively ("high engagement") without numbers.
2 — Engagement not reported but acknowledged as a gap.
1 — Engagement completely absent from report."""

DIMENSION_WEIGHTS = {
    'linkedin_execution': 0.25,
    'agency_coverage': 0.3,
    'post_content_summary': 0.25,
    'engagement_reporting': 0.2,
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
