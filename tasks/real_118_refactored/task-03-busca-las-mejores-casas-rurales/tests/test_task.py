import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("busca las mejores casas rurales donde hospedarse en Catalonia. "
    "Que tengan al menos 4 estrellas. Buscalo en Tripadvisor\n\n"
    "(Translation: Search for the best rural houses to stay in Catalonia. "
    "They must have at least 4 stars. Search on TripAdvisor.)")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: TripAdvisor specifically — not Booking.com, Google Hotels, or general web search
- Location: Catalonia, Spain — results must be in this specific region
- Property type: casas rurales (rural houses / countryside accommodation) — not hotels, hostels, or urban rentals
- Rating constraint: minimum 4 stars (hard constraint — any property below 4 stars is a failure)
- Language note: the task is in Spanish; the agent may respond in Spanish or English — both are acceptable

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace. Then answer each of the following questions with a direct observation — do not infer or assume:

- Did the agent navigate to or search on TripAdvisor? Cite evidence from the trace or response.
- How many rural houses did the agent present? List their names briefly.
- What star ratings were mentioned for each property? List them explicitly.
- Did any property have fewer than 4 stars, or was a rating missing?
- Are the properties located in Catalonia specifically?
- Are the properties rural houses (casas rurales), or a mix of property types?
- Did the agent provide useful details beyond just names (e.g. location within Catalonia, price, amenities, reviews)?

### Step 2: Dimension Scoring
Score each dimension from 1 to 5 using the rubrics below. Each score must be justified by specific evidence from Step 1.

#### A. Platform Execution
Did the agent actually use TripAdvisor as instructed?

5 — Clear evidence in trace and/or response that the agent searched and retrieved results from TripAdvisor; multiple listings accessed.
4 — Agent used TripAdvisor but accessed only 1–2 listings, or trace confirms platform but response doesn't explicitly cite it.
3 — Ambiguous: response mentions TripAdvisor but trace shows no navigation, or agent used TripAdvisor alongside other platforms without distinguishing sources.
2 — Agent used a different platform (e.g. Booking.com, Google) and did not use TripAdvisor at all.
1 — Agent did not perform any search; response is hallucinated or generated from prior knowledge only.

#### B. Constraint Adherence
Do all presented properties meet the 4+ star requirement and match the property type and location?

5 — All properties are: (a) rated 4 stars or above, (b) rural houses (casas rurales), (c) located in Catalonia.
4 — All properties meet the star rating and location constraints; 1 property is ambiguously typed (not clearly a casa rural).
3 — Most properties meet all constraints; 1 property is below 4 stars or has no rating stated.
2 — Multiple properties are missing ratings, below 4 stars, or are not casas rurales.
1 — Constraints were ignored entirely; results are hotels, urban rentals, or unrated properties.

#### C. Result Specificity
Are the results real, named properties with enough detail to be actionable?

5 — 3+ named casas rurales with: TripAdvisor rating, location within Catalonia, and at least one additional detail (price, standout amenity, or review highlight).
4 — 3+ named properties with ratings and location, but additional details are sparse for 1–2 of them.
3 — 2–3 named properties with ratings; location within Catalonia is vague or missing for some.
2 — Properties are named but lack ratings, or only 1–2 properties are presented.
1 — No specific properties named; agent only described a search or listed generic accommodation categories.

#### D. Presentation Quality
Is the output well-organised and useful for someone planning a trip?

5 — Results are clearly structured (e.g. one section per property); each entry includes name, rating, location, and at least one useful detail; easy to compare options.
4 — Results are organised but minor gaps exist (e.g. one property missing a detail, or formatting is slightly inconsistent).
3 — Results are present but presented as prose or a loosely structured list; harder to scan and compare.
2 — Output is a wall of text or a very sparse list; properties are mentioned but not presented in a usable format.
1 — Output is a refusal, error message, or contains no usable accommodation information.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags, exactly as shown:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "platform_execution": <1-5>,
  "constraint_adherence": <1-5>,
  "result_specificity": <1-5>,
  "presentation_quality": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "constraint_adherence": "<one sentence citing specific evidence>",
    "result_specificity": "<one sentence citing specific evidence>",
    "presentation_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_execution":   0.25,
    "constraint_adherence": 0.35,
    "result_specificity":   0.25,
    "presentation_quality": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
