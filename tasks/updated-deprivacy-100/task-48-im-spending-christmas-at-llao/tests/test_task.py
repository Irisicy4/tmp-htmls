"""LLM-as-judge evaluator for task-48-im-spending-christmas-at-llao.

Category: General
Task: I'm spending Christmas at Llao Llao Hotel & Resort in Argentine Patagonia. Please write an efficient itinerary for how to move around inside the resort and nearby area.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "I'm spending Christmas at Llao Llao Hotel & Resort in Argentine Patagonia. Please write an efficient itinerary for how to move around inside the resort and nearby area."

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent produced a high-quality, efficient itinerary for spending Christmas at Llao Llao Hotel & Resort in Argentine Patagonia.'

CONSTRAINTS = """- Venue: Llao Llao Hotel & Resort in Argentine Patagonia specifically — content must be resort-specific, not generic
- Occasion: Christmas — seasonal events and activities should be included if available
- Focus: efficiency of movement inside the resort and nearby area — routing and timing matter
- Nearby area includes: Nahuel Huapi Lake boat trips, hiking trails (e.g. Cerro Lopez, Circuito Chico), and the town of Bariloche
- Output: a structured itinerary with times, locations, and movement guidance"""

EVIDENCE_QUESTIONS = """- Did the agent research Llao Llao Hotel & Resort in Argentine Patagonia specifically?
- Does the itinerary include resort-specific locations (hotel facilities, grounds, nearby Patagonian attractions)?
- Are Christmas-specific events or activities mentioned?
- Is movement efficiency addressed (routing, travel times between venues)?
- Is the itinerary structured with times?"""

DIMENSION_RUBRICS = """#### A. Resort Specificity (0.30)
5 — Itinerary references specific Llao Llao Hotel & Resort facilities, grounds, and nearby Patagonian attractions (e.g. Nahuel Huapi Lake, hiking trails, Bariloche).
4 — Mostly resort-specific but some generic content.
3 — Mix of resort-specific and generic travel advice.
2 — Mostly generic resort itinerary not tailored to Llao Llao.
1 — No resort-specific content.

#### B. Christmas Relevance (0.20)
5 — Christmas events, decorations, or seasonal programming at Llao Llao or in the Patagonia/Bariloche area specifically mentioned.
4 — Christmas theme addressed but events are generic rather than resort-specific.
3 — Christmas mentioned but itinerary is not seasonally adapted.
2 — Itinerary could apply to any time of year.
1 — Christmas not addressed at all.

#### C. Movement Efficiency (0.30)
5 — Itinerary explicitly optimises routing (e.g. geographically grouped, minimises backtracking, accounts for travel times between resort and nearby areas such as the lake or Bariloche).
4 — Good sequencing but efficiency rationale not explicitly stated.
3 — Itinerary is sequential but not clearly optimised for movement.
2 — Activities listed without regard to location or routing.
1 — No movement guidance.

#### D. Structure & Completeness (0.20)
5 — Full day itinerary with times, locations, durations, and practical tips.
4 — Times and locations present but missing some practical details.
3 — Structured list but missing times or durations.
2 — Paragraph description without clear schedule.
1 — No structure."""

DIMENSION_WEIGHTS = {
    'resort_specificity': 0.3,
    'christmas_relevance': 0.2,
    'movement_efficiency': 0.3,
    'structure_completeness': 0.2,
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
