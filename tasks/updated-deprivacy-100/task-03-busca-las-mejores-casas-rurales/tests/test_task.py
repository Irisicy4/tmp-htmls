"""
LLM-as-judge evaluator for EvolveBench task-03.

Category: Travel & Planning
Approach: Hybrid — category-level rubric axes (Travel & Planning) + task-specific constraint injection.
Design grounded in:
  - AutoMetrics: axes-from-examples pipeline, observable 5-point rubric anchors
  - SWE-ABS: forced analysis before verdict, orthogonal sub-dimensions, structured output tag

Task (Spanish original):
  "busca las mejores casas rurales donde hospedarse en Catalonia.
   Que tengan al menos 4 estrellas. Buscalo en Tripadvisor"

Translation:
  "Search for the best rural houses to stay in Catalonia.
   They must have at least 4 stars. Search on TripAdvisor."
"""

import os
import json
import re

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_INSTRUCTION = (
    "busca las mejores casas rurales donde hospedarse en Catalonia. "
    "Que tengan al menos 4 estrellas. Buscalo en Tripadvisor\n\n"
    "(Translation: Search for the best rural houses to stay in Catalonia. "
    "They must have at least 4 stars. Search on TripAdvisor.)"
)

PASS_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Judge prompt
#
# Structure (from SWE-ABS pattern):
#   1. Forced analysis section — prevents jumping to verdict
#   2. Per-dimension scoring — orthogonal axes, each with observable anchors
#   3. Structured output tag — enables reliable parsing + majority voting
#
# Axes (Travel & Planning category):
#   A. Platform Execution     — did the agent actually use TripAdvisor?
#   B. Constraint Adherence   — do all results meet the 4+ star requirement?
#   C. Result Specificity     — are real, named properties returned with details?
#   D. Presentation Quality   — is the output useful and well-organised for a traveller?
#
# Task-specific constraints injected:
#   - Platform: TripAdvisor specifically
#   - Location: Catalonia (region in Spain)
#   - Property type: casas rurales (rural houses/countryside accommodation)
#   - Rating: minimum 4 stars
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based travel research task.

You will be given:
1. The task instruction the agent was given (in Spanish with English translation)
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

You must evaluate the agent's performance across four dimensions. Follow the evaluation steps exactly in order."""


USER_PROMPT_TEMPLATE = """## Task Instruction
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
"""

# Dimension weights — Constraint Adherence is highest because the star rating
# is the only hard, verifiable filter. Platform Execution is second because
# TripAdvisor is explicitly required.
DIMENSION_WEIGHTS = {
    "platform_execution":   0.25,
    "constraint_adherence": 0.35,
    "result_specificity":   0.25,
    "presentation_quality": 0.15,
}

DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response(result: dict) -> str:
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""


def _parse_answer_tag(text):
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def _call_judge_once(agent_response: str, execution_summary: str) -> dict | None:
    """Single judge call. Returns parsed dict or None on failure."""
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1024,
        )
        raw = completion.choices[0].message.content
        return _parse_answer_tag(raw)
    except Exception as e:
        return {"error": str(e)}


def _majority_vote(votes: list[dict]) -> dict:
    """
    Aggregate up to 3 judge calls via majority vote on each dimension.
    Overall score is recomputed from voted dimension scores using weights.
    Used for borderline cases (overall_score within 0.5 of threshold).
    """
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid:
        return votes[0] if votes else {"error": "All judge calls failed"}

    aggregated = {}
    for dim in DIMENSIONS:
        scores = [v[dim] for v in valid]
        aggregated[dim] = sorted(scores)[len(scores) // 2]

    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD

    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)

    return aggregated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test(result: dict) -> dict:
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")

    if not agent_response.strip():
        return {
            "passed": False,
            "feedback": "No response found from agent.",
            "details": {"task_completed": result.get("status") == "success"},
        }

    first_call = _call_judge_once(agent_response, execution_summary)

    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        borderline = abs(float(overall) - PASS_THRESHOLD) <= 0.5

        if borderline:
            call2 = _call_judge_once(agent_response, execution_summary)
            call3 = _call_judge_once(agent_response, execution_summary)
            scores = _majority_vote([first_call, call2, call3])
        else:
            scores = first_call
            scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}

    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)

    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores:
            feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"):
        feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items():
            feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")

    return {
        "passed": bool(passed),
        "feedback": "\n".join(feedback_lines),
        "details": {
            "task_completed": result.get("status") == "success",
            "overall_score": overall,
            "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
            "evidence_summary": scores.get("evidence_summary"),
            "dimension_reasoning": scores.get("dimension_reasoning"),
            "pass_threshold": PASS_THRESHOLD,
            "votes_used": scores.get("_votes_used", 1),
        },
    }