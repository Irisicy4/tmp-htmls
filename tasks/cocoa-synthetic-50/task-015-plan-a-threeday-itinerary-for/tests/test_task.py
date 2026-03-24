"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Plan a three-day itinerary for a visit to Rome, Italy, using public websites and populate a Google Sheet with activities, durations, and costs.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }


TASK_INSTRUCTION = """Plan a three-day itinerary for a visit to Rome, Italy, using public websites like Google Maps for locations, Lonely Planet for recommendations, and Rome's official tourism page for events. Populate a Google Sheet template with daily activities, including landmarks, dining options, activity durations, and estimated costs for each day."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves planning a three-day itinerary for a visit to Rome, Italy. The agent must use public websites like Google Maps, Lonely Planet, and Rome's official tourism page to gather information. A successful completion includes a structured Google Sheet with daily activities, landmarks, dining options, activity durations, and estimated costs.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Plan a three-day itinerary for a visit to Rome, Italy, using public websites like Google Maps for locations, Lonely Planet for recommendations, and Rome's official tourism page for events. Populate a Google Sheet template with daily activities, including landmarks, dining options, activity durations, and estimated costs for each day.

## Task-Specific Constraints
- Must use at least three of the specified platforms (Google Maps, Lonely Planet, turismoroma.it).
- Must include at least three landmarks per day.
- Must provide estimated costs for each activity or meal.
- Output must be organized in a structured table format.
- Must include activity durations for all listed activities.
- Must ensure the itinerary is feasible based on travel times.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are there at least three landmarks per day in the itinerary?
- Are estimated costs provided for all activities and meals?
- Is the output organized in a structured table format?
- Are activity durations and travel times included and feasible?

### Step 2: Dimension Scoring

#### A. Itinerary Completeness (0.35)
Measures whether the itinerary includes all required elements (landmarks, dining options, durations, costs).

5 — Includes all required elements for all three days, with no omissions.
4 — Includes most required elements, but minor omissions are present.
3 — Includes some required elements, but significant omissions exist.
2 — Includes very few required elements, with major gaps.
1 — Includes almost none of the required elements.

#### B. Platform Usage and Coverage (0.30)
Measures whether the agent used the required platforms and gathered sufficient information.

5 — Uses all three specified platforms and gathers comprehensive information.
4 — Uses at least two platforms and gathers sufficient information.
3 — Uses at least one platform but with limited information.
2 — Minimal platform usage with insufficient information.
1 — No evidence of platform usage.

#### C. Detail and Feasibility (0.20)
Measures whether the itinerary is detailed and feasible based on travel times and durations.

5 — Highly detailed and feasible, with accurate travel times and durations.
4 — Mostly detailed and feasible, with minor inaccuracies.
3 — Somewhat detailed but with notable feasibility issues.
2 — Lacks detail and has major feasibility issues.
1 — No detail or feasibility considerations.

#### D. Output Structure and Organization (0.15)
Measures the clarity and organization of the output.

5 — Output is well-structured, clear, and easy to follow.
4 — Output is mostly clear but with minor formatting issues.
3 — Output is somewhat clear but has significant formatting issues.
2 — Output is poorly structured and difficult to follow.
1 — Output is unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "itinerary_completeness": <1-5>,
  "platform_usage_and_coverage": <1-5>,
  "detail_and_feasibility": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "itinerary_completeness": "<one sentence citing specific evidence>",
    "platform_usage_and_coverage": "<one sentence citing specific evidence>",
    "detail_and_feasibility": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "itinerary_completeness": 0.35,
    "platform_usage_and_coverage": 0.30,
    "detail_and_feasibility": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())