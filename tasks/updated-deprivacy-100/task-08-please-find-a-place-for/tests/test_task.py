"""LLM-as-judge evaluator for task-08-please-find-a-place-for.

Category: Travel & Planning
Task: Please find a place for a date in Sydney, Australia on December 25, 2025. I wish there was a place with easy parking.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please find a place for a date in Sydney, Australia on December 25, 2025. I wish there was a place with easy parking.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully found suitable date spots in Sydney, Australia for Christmas Day (December 25, 2025).'

CONSTRAINTS = """- Location: Sydney, Australia — recommendations must be in Sydney or Greater Sydney area
- Date: December 25, 2025 (Christmas Day) — venues must be open or accessible on a public holiday
- Parking: easy parking is explicitly requested — recommendations should include parking availability or nearby parking options
- Output: the agent should present specific venue recommendations, not just generic advice"""

EVIDENCE_QUESTIONS = """- Did the agent search for date spots or navigate a relevant platform? Cite evidence.
- What specific venues did the agent recommend? List names and locations if mentioned.
- Are the venues in Sydney or Greater Sydney area?
- Did the agent address Christmas Day availability (December 25 is a public holiday in Australia)?
- Did the agent address parking availability at or near the recommended venues?"""

DIMENSION_RUBRICS = """#### A. Location Relevance
Are the recommended venues actually in Sydney or Greater Sydney area?

5 — All recommendations are clearly in Sydney or Greater Sydney, with specific suburb or address details.
4 — Recommendations are Sydney-based but location details are vague (e.g. no suburb or address).
3 — Most recommendations are Sydney-based but one or more are unclear or outside the area.
2 — Recommendations are for Australia generally without Sydney specificity.
1 — Recommendations are not in Sydney or are entirely generic with no location detail.

#### B. Christmas Day Availability
Did the agent address whether venues are open or suitable on December 25?

5 — Agent explicitly verified or noted Christmas Day operating hours or suitability for each venue.
4 — Agent acknowledged Christmas Day context and noted most venues should be accessible, with some specifics.
3 — Agent mentioned it is Christmas Day but did not verify individual venue availability.
2 — Date noted without any holiday context or availability check.
1 — Christmas Day context ignored entirely.

#### C. Parking Coverage
Did the agent address the easy parking requirement?

5 — Parking availability or nearby parking options explicitly noted for recommended venues.
4 — Parking mentioned for most venues; one or two lack parking detail.
3 — Parking mentioned in general terms but not tied to specific venues.
2 — Parking briefly mentioned without actionable detail.
1 — Parking requirement not addressed.

#### D. Recommendation Quality
Are the recommendations specific, useful, and suitable for a date?

5 — 3+ specific venue recommendations with name, location, why it suits a date, and parking info.
4 — 3+ venues named with most key details; one element missing per venue.
3 — 2–3 venues mentioned with partial details; user would need to do follow-up research.
2 — 1–2 venues mentioned or recommendations are vague.
1 — No specific venues; only generic advice or refusal."""

DIMENSION_WEIGHTS = {
    'location_relevance': 0.3,
    'christmas_availability': 0.25,
    'parking_coverage': 0.25,
    'recommendation_quality': 0.2,
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
