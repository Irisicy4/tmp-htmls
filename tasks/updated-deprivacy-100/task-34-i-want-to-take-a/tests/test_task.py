"""LLM-as-judge evaluator for task-34-i-want-to-take-a.

Category: Travel & Planning
Task: I want to take a train from Kanazawa to Hakone. Please help me plan how to buy tickets for travel on February 12, 2026.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'I want to take a train from Kanazawa to Hakone. Please help me plan how to buy tickets for travel on February 12, 2026.'

SYSTEM_PROMPT_EXTRA = 'This is a Japanese rail ticket planning task. The agent should identify the appropriate route from Kanazawa to Hakone, which typically involves the Shinkansen (bullet train) and possibly transfers. Relevant platforms include JR Tickets, Hyperdia, ekitan.com, and the JR Pass may be applicable.'

CONSTRAINTS = """- Origin: Kanazawa, Ishikawa Prefecture
- Destination: Hakone, Kanagawa Prefecture
- Date: February 12, 2026
- Platforms: JR Tickets, Hyperdia, ekitan.com are common Japanese rail booking and search platforms; JR Pass may be relevant
- Agent should identify the route (likely via Shinkansen with transfers), check train options, and advise on ticket purchase process"""

EVIDENCE_QUESTIONS = """- Did the agent identify the route from Kanazawa to Hakone (including any transfers)?
- Did the agent search a Japanese rail platform (JR Tickets, Hyperdia, ekitan.com, etc.) for trains on Feb 12?
- Were specific train options (train number, departure time, duration, price) provided?
- Was the JR Pass or ticket purchase process discussed?
- Were ticket purchase steps outlined?"""

DIMENSION_RUBRICS = """#### A. Route Clarity
Did the agent correctly identify the full route (origin station → destination station, including transfers)?

5 — Full route clearly identified (e.g. Kanazawa → Nagoya/Tokyo via Shinkansen → Hakone-Yumoto); transfer requirements and stations noted.
4 — Route identified but transfer details incomplete or one leg vague.
3 — General direction correct but route details vague.
2 — Only one endpoint identified; agent asked for clarification without providing useful info.
1 — Route not identified; response is generic travel advice.

#### B. Ticket Platform Guidance
Did the agent guide on how to buy tickets via Japanese rail platforms?

5 — Agent navigated JR Tickets, Hyperdia, ekitan.com, or equivalent and provided specific steps or found actual available trains on Feb 12.
4 — Agent described a Japanese rail booking platform process accurately without actually searching.
3 — Agent mentioned a Japanese rail platform without detailed guidance.
2 — Agent mentioned buying tickets without specifying a platform.
1 — No platform guidance.

#### C. Date Context Awareness
Did the agent address the February 12, 2026 travel date appropriately?

5 — Agent acknowledged the specific travel date, checked or advised on availability, and noted any relevant booking lead times.
4 — Agent noted the date and provided general booking timing advice.
3 — Agent referenced the date but did not address availability or booking implications.
2 — Date noted without any contextual guidance.
1 — Date context ignored.

#### D. Actionability
Can the user immediately act on the information provided?

5 — Complete action plan: full route identified, platform specified, train options or search steps provided, ticket purchase guidance included.
4 — Most elements present; one step (e.g. specific train options) missing.
3 — Partial guidance; user would need to do significant additional research.
2 — High-level overview only; not actionable.
1 — No actionable information."""

DIMENSION_WEIGHTS = {
    'route_clarity': 0.25,
    'ticket_platform_guidance': 0.3,
    'date_context_awareness': 0.25,
    'actionability': 0.2,
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
