"""LLM-as-judge evaluator for task-63-please-search-the-internet-for.

Category: Marketing & Analytics
Task: Please search the internet for the company at https://www.peloton.com and find all their weak points.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please search the internet for the company at https://www.peloton.com and find all their weak points.'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves analyzing a company's website and online presence to identify their weaknesses for competitive intelligence purposes. The target company is Peloton (peloton.com), a fitness equipment and subscription service company."

CONSTRAINTS = """- Target: https://www.peloton.com — must visit this site
- Scope: web presence, reviews, SEO, product/service gaps, customer complaints
- Depth: multiple angles (website quality, social proof, pricing, complaints)
- Output: actionable list of competitor weaknesses"""

EVIDENCE_QUESTIONS = """- Did the agent visit peloton.com?
- What aspects of the site were analyzed (design, content, SEO, offerings)?
- Were external sources checked (Google reviews, social media, complaints)?
- Are specific weaknesses identified with evidence?"""

DIMENSION_RUBRICS = """#### A. Site Analysis (0.3)
Did the agent analyze the company's website?

5 — Agent visited the site and analyzed: design quality, content, product/service offering, UX, and value proposition.
4 — Visited site but analysis is shallow.
3 — Visited site but only described it without identifying weaknesses.
2 — Described what to look for without visiting.
1 — No site visit.

#### B. External Research (0.25)
Did the agent research external sources?

5 — Checked Google reviews, social media, and/or complaint platforms for customer sentiment.
4 — Checked one external source.
3 — Brief external search without substantive findings.
2 — Only searched the company's own site.
1 — No external research.

#### C. Weakness Identification (0.35)
Are specific, actionable weaknesses identified?

5 — 5+ specific weaknesses with evidence (e.g. high price point, subscription churn, hardware reliability issues, intense competition, thin profit margins).
4 — 3-4 specific weaknesses.
3 — 1-2 specific weaknesses.
2 — Vague weaknesses without evidence.
1 — No weaknesses identified.

#### D. Competitive Framing (0.1)
Are weaknesses framed as opportunities?

5 — Each weakness explicitly linked to how a competitor can exploit it.
4 — Some competitive framing.
3 — Weaknesses listed without competitive context.
2 — Analysis from wrong perspective.
1 — No competitive framing."""

DIMENSION_WEIGHTS = {
    'site_analysis': 0.3,
    'external_research': 0.25,
    'weakness_identification': 0.35,
    'competitive_framing': 0.1,
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
