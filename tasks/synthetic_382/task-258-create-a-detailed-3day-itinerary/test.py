"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a detailed 3-day itinerary for Tokyo using a Google Sheets template, including activities, location names, and estimated costs based on data from TripAdvisor, Japan-Guide, and Google Maps.
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


TASK_INSTRUCTION = """Create a detailed 3-day itinerary for Tokyo using a Google Sheets template. Include activities, location names, and estimated costs based on data from TripAdvisor, Japan-Guide, and Google Maps."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a detailed 3-day itinerary for Tokyo using a Google Sheets template. The itinerary must include activities, location names, and estimated costs, and the agent must gather data from TripAdvisor, Japan-Guide, and Google Maps. A successful completion requires a structured and complete itinerary with accurate and sourced information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a detailed 3-day itinerary for Tokyo using a Google Sheets template. Include activities, location names, and estimated costs based on data from TripAdvisor, Japan-Guide, and Google Maps.

## Task-Specific Constraints
- Must visit and extract data from TripAdvisor, Japan-Guide, and Google Maps.
- Must include estimated costs for all activities and locations.
- Must organize the output as a structured table in the Google Sheets template.
- Must include at least 3 activities per day.
- Must provide location names and activity descriptions for each entry.
- Must ensure data accuracy and source credibility.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TripAdvisor, Japan-Guide, and Google Maps? Which platforms were actually visited?
- Are estimated costs included for all activities and locations?
- Is the output organized as a structured table in the Google Sheets template?
- Are there at least 3 activities per day in the itinerary?
- Are location names and activity descriptions accurate and sourced?

### Step 2: Dimension Scoring

#### A. Itinerary Completeness (0.35)
Measures whether the 3-day itinerary includes all required components: activities, location names, and estimated costs.

5 — Includes all required components for all 3 days, with no missing entries.
4 — Includes most required components, with minor omissions.
3 — Includes some required components, but several entries are incomplete.
2 — Includes few required components, with major omissions.
1 — Includes none of the required components.

#### B. Platform Coverage (0.30)
Measures whether the agent used TripAdvisor, Japan-Guide, and Google Maps effectively to gather data.

5 — Data is sourced from all three platforms with clear evidence.
4 — Data is sourced from at least two platforms, with minor gaps.
3 — Data is sourced from one platform, or evidence is unclear.
2 — Minimal platform usage, with significant gaps.
1 — No evidence of platform usage.

#### C. Detail and Specificity (0.20)
Measures the level of detail in the itinerary, including activity descriptions, location names, and cost estimates.

5 — Highly detailed itinerary with accurate descriptions and cost estimates for all entries.
4 — Detailed itinerary with minor inaccuracies or missing details.
3 — Moderately detailed itinerary, with several missing or vague entries.
2 — Poorly detailed itinerary, with most entries incomplete.
1 — No meaningful details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and uses credible sources.

5 — Output is structured as a clear table and uses credible sources for all data.
4 — Output is mostly well-organized, with minor formatting or credibility issues.
3 — Output is partially organized, with noticeable formatting or credibility issues.
2 — Output is poorly organized, with major formatting or credibility issues.
1 — Output is unstructured and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "itinerary_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "itinerary_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "itinerary_completeness": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())