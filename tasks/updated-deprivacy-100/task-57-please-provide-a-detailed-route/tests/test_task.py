"""LLM-as-judge evaluator for task-57-please-provide-a-detailed-route.

Category: Travel & Planning
Task: Please provide a detailed route guide for the Tongariro Alpine Crossing combining the Mangatepopo Valley approach and the Emerald Lakes descent route.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please provide a detailed route guide for the Tongariro Alpine Crossing combining the Mangatepopo Valley approach and the Emerald Lakes descent route.'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent produced a comprehensive, specific hiking route guide for the Tongariro Alpine Crossing in New Zealand, specifically covering the Mangatepopo Valley approach and the Emerald Lakes descent route.'

CONSTRAINTS = """- Specific route: must cover the Mangatepopo Valley approach AND the Emerald Lakes descent
- Location: Tongariro National Park, New Zealand (Tongariro Alpine Crossing)
- Key waypoints: Mangatepopo Valley, Soda Springs, South Crater, Red Crater, Emerald Lakes, Blue Lake, Ketetahi area
- Guide must include: distance (~19.4 km), elevation gain (~1,600 m), duration (typically 7–9 hours), key waypoints, difficulty level
- Practical info: transportation (shuttle required — one-way track), best season (Oct–Apr), gear recommendations, volcanic hazard and weather change warnings
- Safety considerations: active volcanic area, NZ DOC alerts, sudden weather changes, appropriate footwear and layers"""

EVIDENCE_QUESTIONS = """- Did the agent research this specific crossing, including both the Mangatepopo Valley approach and Emerald Lakes descent?
- Are key waypoints (South Crater, Red Crater, Emerald Lakes) identified and described?
- What specific distances, durations, and elevation figures are mentioned?
- Is practical information (shuttle transport, season, gear, volcanic hazards) included?"""

DIMENSION_RUBRICS = """#### A. Route Specificity (0.35)
Does the guide cover the specific Tongariro Alpine Crossing route combination?

5 — Both Mangatepopo Valley approach and Emerald Lakes descent explicitly covered with specific waypoints (e.g. South Crater, Red Crater, Blue Lake) and route description.
4 — Both elements covered but one is less detailed.
3 — One element covered well; the other mentioned briefly.
2 — Generic Tongariro Alpine Crossing guide without specific mention of both named route sections.
1 — No route-specific content.

#### B. Technical Detail (0.3)
Does the guide include distance, elevation, duration, and difficulty?

5 — All four metrics present with specific numbers (e.g. ~19.4 km, ~1,600 m elevation gain, 7–9 hours, Grade: Demanding).
4 — Three of four metrics present.
3 — Two of four metrics present.
2 — Only one metric present.
1 — No technical details.

#### C. Waypoint Coverage (0.2)
Are key waypoints along the route described?

5 — 5 or more named waypoints described (e.g. Mangatepopo car park, Soda Springs, South Crater, Red Crater summit, Emerald Lakes, Blue Lake, Ketetahi Shelter).
4 — 3–4 named waypoints.
3 — 1–2 named waypoints.
2 — Waypoints mentioned but not named or described.
1 — No waypoints.

#### D. Practical Information (0.15)
Is practical travel information included?

5 — Transportation (shuttle logistics), best season, gear recommendations, and volcanic/weather safety warnings all addressed.
4 — Three of four practical elements addressed.
3 — Two practical elements.
2 — One practical element.
1 — No practical information."""

DIMENSION_WEIGHTS = {
    'route_specificity': 0.35,
    'technical_detail': 0.3,
    'waypoint_coverage': 0.2,
    'practical_information': 0.15,
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
