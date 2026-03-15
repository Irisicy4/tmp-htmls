"""
LLM-as-judge evaluator for EvolveBench task-72.

Category: Travel & Planning
Task: Find me date spots in the Suwon and Gyeonggi area for December 25th. Places with convenient parking ...
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


TASK_INSTRUCTION = """Find me date spots in the Suwon and Gyeonggi area for December 25th. Places with convenient parking would be great."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is similar to task-71 but with an added parking requirement and specific date (December 25th).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: Suwon and Gyeonggi-do area
- Date: December 25th specifically (Christmas day)
- Parking: venues with convenient/available parking preferred
- Type: date-appropriate activities and venues

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for venues in Suwon/Gyeonggi?
- Are December 25th hours/availability confirmed?
- Is parking availability addressed?
- Are spots couple-friendly?
- How many options are provided?

### Step 2: Dimension Scoring

#### A. Location Relevance (0.25)
Are recommendations in Suwon or Gyeonggi-do?

5 — All spots clearly in Suwon or Gyeonggi-do with addresses.
4 — Most spots in correct area.
3 — Mix of correct area and Seoul.
2 — Mostly Seoul-based.
1 — Wrong location.

#### B. Christmas Day Availability (0.3)
Are spots confirmed open on December 25th?

5 — December 25th availability explicitly confirmed for each venue.
4 — Availability checked but not all venues confirmed.
3 — Regular hours checked without Dec 25 specific confirmation.
2 — No date verification.
1 — Spots may be closed on Christmas Day.

#### C. Parking Coverage (0.25)
Is parking availability addressed?

5 — Parking availability confirmed for each venue (free, paid, capacity).
4 — Parking mentioned for most venues.
3 — Parking mentioned generally without per-venue detail.
2 — Parking briefly mentioned without useful info.
1 — Parking not addressed.

#### D. Recommendation Quality (0.2)
Are spots couple-friendly and well-described?

5 — 3+ specific spots with name, location, why great for a date, parking info, and Dec 25 hours.
4 — 2-3 spots with most details.
3 — Spots identified but sparse detail.
2 — Very generic.
1 — No specific recommendations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "location_relevance": <1-5>,
  "christmas_day_availability": <1-5>,
  "parking_coverage": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "location_relevance": "<one sentence citing specific evidence>",
    "christmas_day_availability": "<one sentence citing specific evidence>",
    "parking_coverage": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "location_relevance": 0.25,
    "christmas_day_availability": 0.3,
    "parking_coverage": 0.25,
    "recommendation_quality": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())