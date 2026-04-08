"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Calculate the total cost of a 7-day trip to London, UK for one person, including flights, hotel, transportation, and food budget.
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


TASK_INSTRUCTION = """Calculate the total cost of a 7-day trip to London, UK for one person. Include roundtrip flight costs, 7 nights at a mid-range hotel, public transportation pass, and daily food budget. Use Expedia for flights, Booking.com for hotel costs, and the official Transport for London site for public transport pricing."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost of a 7-day trip to London, UK for one person. This includes roundtrip flight costs, 7 nights at a mid-range hotel, public transportation pass, and daily food budget. The agent must use Expedia for flights, Booking.com for hotel costs, and the official Transport for London site for public transport pricing. A successful completion must provide accurate cost data from these platforms and present the total cost in a clear and organized format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of a 7-day trip to London, UK for one person. Include roundtrip flight costs, 7 nights at a mid-range hotel, public transportation pass, and daily food budget. Use Expedia for flights, Booking.com for hotel costs, and the official Transport for London site for public transport pricing.

## Task-Specific Constraints
- Must visit Expedia, Booking.com, and the official Transport for London site.
- Must include price data for flights, hotel, transportation, and food.
- Output must be organized as a structured list or table.
- Must calculate and present the total cost clearly.
- Must source prices directly from the specified platforms.
- Must ensure all costs are in GBP and converted if necessary.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Expedia, Booking.com, and the official Transport for London site? Which ones were actually visited?
- Are flight, hotel, transportation, and food costs present in the response?
- Is the output organized as a structured list or table?
- Are all costs presented in GBP and converted correctly if necessary?
- Is the total cost calculation accurate and clearly presented?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the total cost calculation is correct and complete.

5 — Includes accurate costs for all items (flights, hotel, transportation, food) and calculates the total correctly.
4 — Includes most costs accurately but has minor errors in calculation or missing details.
3 — Includes partial costs or has significant calculation errors.
2 — Includes few costs or major calculation errors.
1 — Does not calculate the total cost or includes no valid data.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used all specified platforms and sourced data correctly.

5 — Uses Expedia, Booking.com, and Transport for London, with sourced data from all.
4 — Uses at least two platforms with sourced data but misses one.
3 — Uses only one platform or sources incomplete data.
2 — Attempts platform use but fails to retrieve usable data.
1 — Does not use any specified platforms.

#### C. Specificity and Detail (0.20)
Measures the depth and specificity of the response, including itemized costs and clear breakdowns.

5 — Provides detailed breakdowns for all items, including individual costs and conversions.
4 — Provides breakdowns for most items but lacks minor details.
3 — Provides partial breakdowns or vague descriptions.
2 — Provides minimal detail or unclear breakdowns.
1 — Provides no breakdowns or details.

#### D. Output Structure and Clarity (0.15)
Measures the organization and readability of the response.

5 — Response is well-organized, formatted as a table or structured list, and easy to read.
4 — Response is mostly organized but has minor formatting issues.
3 — Response is partially organized but difficult to follow.
2 — Response is poorly organized and unclear.
1 — Response is completely unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "specificity_and_detail": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "specificity_and_detail": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  },
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "specificity_and_detail": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())