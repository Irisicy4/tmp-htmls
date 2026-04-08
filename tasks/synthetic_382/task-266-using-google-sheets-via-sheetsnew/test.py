"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a detailed 5-day travel itinerary for Paris using specified platforms and tools.
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


TASK_INSTRUCTION = """Using Google Sheets (via sheets.new), create a travel itinerary for a 5-day trip to Paris. Include activities for each day, estimated costs, and travel times between locations. Use TripAdvisor, Google Maps, and Paris tourism sites to gather information."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a detailed 5-day travel itinerary for Paris, including daily activities, estimated costs, and travel times between locations. The agent must use TripAdvisor, Google Maps, and Paris tourism sites to gather information. A successful completion requires a well-structured itinerary with accurate and relevant data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets (via sheets.new), create a travel itinerary for a 5-day trip to Paris. Include activities for each day, estimated costs, and travel times between locations. Use TripAdvisor, Google Maps, and Paris tourism sites to gather information.

## Task-Specific Constraints
- Must use TripAdvisor, Google Maps, and Paris tourism sites to gather information.
- Must include estimated costs for all activities and travel.
- Must provide travel times between locations for each day.
- Output must be organized as a structured table or list in Google Sheets.
- Must include at least 3 activities per day.
- Must ensure activities are relevant to Paris tourism.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TripAdvisor, Google Maps, and Paris tourism sites? Which ones were actually visited?
- Are estimated costs for all activities and travel present in the response?
- Is the output organized as a structured table or list in Google Sheets?
- Are travel times between locations provided for each day?
- Are there at least 3 activities per day, and are they relevant to Paris tourism?

### Step 2: Dimension Scoring

#### A. Itinerary Completeness (0.35)
Measures whether the itinerary includes all required elements: daily activities, estimated costs, and travel times.

5 — Includes all required elements for all 5 days with no missing data.
4 — Includes most required elements, with minor omissions or inaccuracies.
3 — Includes some required elements, but several are incomplete or missing.
2 — Includes few required elements, with significant omissions.
1 — Does not include any required elements or is completely incorrect.

#### B. Platform Usage Coverage (0.30)
Measures whether the agent used all required platforms (TripAdvisor, Google Maps, Paris tourism sites).

5 — Uses all three platforms and integrates data from each effectively.
4 — Uses two platforms and integrates data from them effectively.
3 — Uses one platform and integrates data from it effectively.
2 — Navigates to platforms but fails to integrate data.
1 — Does not use any required platforms.

#### C. Detail and Specificity (0.25)
Measures the level of detail in the itinerary, including costs, travel times, and activity descriptions.

5 — Provides highly detailed costs, travel times, and activity descriptions for all days.
4 — Provides detailed costs, travel times, and activity descriptions for most days.
3 — Provides basic costs, travel times, and activity descriptions, but lacks detail.
2 — Provides minimal or vague details, with significant gaps.
1 — Provides no meaningful details.

#### D. Output Structure and Organization (0.10)
Measures whether the output is well-organized and presented in a structured format.

5 — Output is fully organized as a structured table or list in Google Sheets.
4 — Output is mostly organized, with minor formatting issues.
3 — Output is partially organized, with significant formatting issues.
2 — Output is disorganized and hard to interpret.
1 — Output is completely unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent used TripAdvisor and Google Maps but did not visit Paris tourism sites. Estimated costs and travel times are present but incomplete. The output is structured as a table in Google Sheets, and most days include 3 activities.",
  "itinerary_completeness": 3,
  "platform_usage_coverage": 3,
  "detail_and_specificity": 3,
  "output_structure_and_organization": 4,
  "dimension_reasoning": {
    "itinerary_completeness": "Several required elements are incomplete, such as travel times for some days.",
    "platform_usage_coverage": "Only two of the three required platforms were used.",
    "detail_and_specificity": "Details are present but lack depth in activity descriptions and cost breakdowns.",
    "output_structure_and_organization": "The output is organized as a structured table in Google Sheets."
  },
  "overall_score": 3.15,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "itinerary_completeness": 0.35,
    "platform_usage_coverage": 0.30,
    "detail_and_specificity": 0.25,
    "output_structure_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())