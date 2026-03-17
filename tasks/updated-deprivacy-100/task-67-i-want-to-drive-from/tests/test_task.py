"""
LLM-as-judge evaluator for EvolveBench task-67.

Category: Travel & Planning
Task: I want to drive from Sydney to Melbourne tomorrow. Please plan a route for me with light traffic. I need to drive at night at around 100 km/h and arrive within one day.
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


TASK_INSTRUCTION = """I want to drive from Sydney to Melbourne tomorrow. Please plan a route for me with light traffic. I need to drive at night at around 100 km/h and arrive within one day."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves planning a long-distance driving route in Australia from Sydney, NSW to Melbourne, VIC, optimizing for light traffic and nighttime driving at highway speeds.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Origin: Sydney, NSW
- Destination: Melbourne, VIC
- Speed: ~100 km/h (highway driving)
- Constraint: arrive within one day
- Preference: light traffic route
- Must include: estimated distance, duration, major roads/highways, rest stop recommendations

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent plan a specific route between the two cities?
- What highways and roads were identified?
- What is the estimated total distance and driving time?
- Does the route feasibility check out (can it be done in one day at 100 km/h)?
- Are rest stops or timing recommendations included?

### Step 2: Dimension Scoring

#### A. Route Planning (0.35)
Did the agent plan a specific, complete route?

5 — Specific route with named highways (Hume Highway, M31, etc.), major towns passed, and key junctions.
4 — Route planned but some segments vague.
3 — General direction planned without specific highways.
2 — Only origin/destination acknowledged without route detail.
1 — No route planned.

#### B. Feasibility Analysis (0.3)
Is the route feasible within the constraints?

5 — Distance and time calculated (approximately 870 km, ~9-10 hours), confirmed doable in one day.
4 — Feasibility confirmed but calculation less precise.
3 — Feasibility stated without calculation.
2 — Route planned but feasibility not addressed.
1 — Route infeasible or not analyzed.

#### C. Traffic Optimization (0.2)
Is the route optimized for light traffic?

5 — Alternative routes compared, nighttime driving advantages noted, peak congestion areas identified.
4 — Light traffic preference addressed but only partially.
3 — Traffic mentioned but no specific optimization.
2 — Generic route without traffic consideration.
1 — Traffic not addressed.

#### D. Practical Details (0.15)
Are practical driving details included?

5 — Rest stops, fuel stations, toll estimates, and timing recommendations included.
4 — Some practical details (e.g. rest stop timing) included.
3 — Distance and time only, no practical tips.
2 — Very limited practical info.
1 — No practical details.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "route_planning": <1-5>,
  "feasibility_analysis": <1-5>,
  "traffic_optimization": <1-5>,
  "practical_details": <1-5>,
  "dimension_reasoning": {{
    "route_planning": "<one sentence citing specific evidence>",
    "feasibility_analysis": "<one sentence citing specific evidence>",
    "traffic_optimization": "<one sentence citing specific evidence>",
    "practical_details": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "route_planning": 0.35,
    "feasibility_analysis": 0.3,
    "traffic_optimization": 0.2,
    "practical_details": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())