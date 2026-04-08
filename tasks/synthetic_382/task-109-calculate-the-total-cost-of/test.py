"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Calculate the total cost of a 5-day trip from New York City to Miami, FL, including flights, hotel stay, and meals.
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


TASK_INSTRUCTION = """Calculate the total cost of a 5-day trip from New York City to Miami, FL. Include the cost of round-trip flights, a hotel stay, and daily meals. Use Kayak to find flight prices, Booking.com for hotels, and Numbeo for average meal costs in Miami. Provide a total cost breakdown and recommend the most budget-friendly options for each category."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate the total cost of a 5-day trip from New York City to Miami, FL. This includes finding round-trip flight prices, hotel costs, and average daily meal expenses using specific platforms (Kayak, Booking.com, Numbeo). A successful completion must include a detailed cost breakdown and recommendations for budget-friendly options.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of a 5-day trip from New York City to Miami, FL. Include the cost of round-trip flights, a hotel stay, and daily meals. Use Kayak to find flight prices, Booking.com for hotels, and Numbeo for average meal costs in Miami. Provide a total cost breakdown and recommend the most budget-friendly options for each category.

## Task-Specific Constraints
- Must use Kayak, Booking.com, and Numbeo to gather data.
- Must include specific price data for flights, hotels, and meals.
- Output must be organized as a structured list or table.
- Must recommend the most budget-friendly options for each category.
- Must provide a total cost breakdown with subtotals for flights, hotels, and meals.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Kayak, Booking.com, and Numbeo? Which platforms were actually visited?
- Are flight, hotel, and meal costs present in the response?
- Is the output organized as a structured list or table?
- Are the recommendations for budget-friendly options clearly stated?
- Is the total cost breakdown accurate and consistent with the subtotals?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the total cost calculation is correct and complete.

5 — Includes accurate flight, hotel, and meal costs with correct total calculation.
4 — Includes all costs but with minor errors in calculations.
3 — Includes partial costs or calculation errors that affect usability.
2 — Missing major costs or significant calculation errors.
1 — No usable cost calculation provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used all specified platforms to gather data.

5 — Uses Kayak, Booking.com, and Numbeo with clear evidence of data from each.
4 — Uses all platforms but with unclear sourcing for some data.
3 — Uses at least two platforms with partial data coverage.
2 — Uses only one platform or incomplete data from multiple platforms.
1 — No evidence of platform usage.

#### C. Depth and Specificity of Data (0.20)
Measures whether the response includes detailed and specific data.

5 — Provides detailed flight, hotel, and meal costs with comparisons and recommendations.
4 — Provides detailed costs but lacks comparisons or recommendations.
3 — Provides basic costs with minimal detail or specificity.
2 — Provides vague or incomplete cost data.
1 — No meaningful cost data provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and uses credible sources.

5 — Response is structured as a clear table or list with credible sourcing.
4 — Response is structured but lacks clarity or minor sourcing issues.
3 — Response is usable but poorly organized or unclear.
2 — Response is disorganized or lacks credibility.
1 — Response is unusable or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity_of_data": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_data": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity_of_data": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())