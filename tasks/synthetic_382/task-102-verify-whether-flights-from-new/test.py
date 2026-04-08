"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Verify whether flights from New York (JFK) to London (LHR) are being delayed today using three platforms and report findings.
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


TASK_INSTRUCTION = """Verify whether flights from New York (JFK) to London (LHR) are being delayed today. Check real-time departure status from three sources and report whether there are widespread delays with timestamps and percentages of delayed flights."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify real-time flight delays between New York (JFK) and London (LHR) using three platforms: flightaware.com, skyscanner.com, and google.com/flights. A successful completion involves gathering accurate delay information, calculating percentages, and presenting findings clearly.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether flights from New York (JFK) to London (LHR) are being delayed today. Check real-time departure status from three sources and report whether there are widespread delays with timestamps and percentages of delayed flights.

## Task-Specific Constraints
- Must visit flightaware.com, skyscanner.com, and google.com/flights.
- Must include timestamps for delay data from each platform.
- Must calculate and report the percentage of delayed flights.
- Must provide a summary indicating whether delays are widespread.
- Output must be structured as a clear table or list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to flightaware.com, skyscanner.com, and google.com/flights? Which platforms were actually visited?
- Are timestamps for delay data included for all platforms?
- Did the agent calculate percentages of delayed flights accurately?
- Is the output structured as a clear table or list?
- Does the summary indicate whether delays are widespread?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent accurately identified flight delays and calculated percentages.

5 — Accurately identifies delays and calculates percentages for all platforms.
4 — Minor inaccuracies in calculations or missing data for one platform.
3 — Partial data or significant inaccuracies in calculations.
2 — Mostly incorrect or incomplete data.
1 — No accurate data or calculations.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and gathered data.

5 — Data gathered from all three platforms.
4 — Data gathered from two platforms.
3 — Data gathered from one platform.
2 — Attempted but failed to gather data from any platform.
1 — Did not attempt to gather data.

#### C. Depth of Analysis (0.25)
Measures whether the agent included timestamps, percentages, and a summary of widespread delays.

5 — Includes timestamps, percentages, and a clear summary for all platforms.
4 — Includes most required elements but misses one minor detail.
3 — Includes some required elements but lacks depth.
2 — Includes minimal detail or analysis.
1 — No meaningful analysis.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list, and sources are credible.
4 — Output is mostly clear but slightly disorganized.
3 — Output is usable but lacks clarity or credibility.
2 — Output is poorly structured or sources are questionable.
1 — Output is completely disorganized or lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())