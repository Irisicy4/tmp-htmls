"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Calculate the total cost for a 5-day trip to Bali, Indonesia, including airfare, hotel, and daily budget.
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


TASK_INSTRUCTION = """Calculate the total cost for a 5-day trip to Bali, Indonesia, including round-trip airfare from Los Angeles, a mid-range hotel stay, and a daily budget for meals and activities. Use live data from flight booking websites and hotel platforms."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate the total cost for a 5-day trip to Bali, Indonesia, including airfare, hotel, and daily budget. The domain involves travel planning, and successful completion requires accurate cost estimates from live data sources. The deliverable must include detailed pricing for airfare, hotel stay, and daily expenses.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost for a 5-day trip to Bali, Indonesia, including round-trip airfare from Los Angeles, a mid-range hotel stay, and a daily budget for meals and activities. Use live data from flight booking websites and hotel platforms.

## Task-Specific Constraints
- Must visit skyscanner.com, booking.com, and numbeo.com to gather data.
- Must provide a breakdown of airfare, hotel costs, and daily expenses.
- Output must be organized as a structured list or table.
- Must include specific price data for each category (airfare, hotel, daily budget).
- Must calculate and present the total trip cost clearly.
- Must address any discrepancies in live data with reasonable assumptions.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are airfare, hotel costs, and daily expenses present in the response?
- Is the output organized as a structured list or table?
- Are the total trip cost and individual category costs clearly presented?
- Are any discrepancies or assumptions in the data addressed reasonably?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the total trip cost calculation is correct and complete.

5 — Includes accurate airfare, hotel costs, daily expenses, and total cost.
4 — Includes most required costs but with minor inaccuracies or omissions.
3 — Includes partial costs but lacks completeness or has notable inaccuracies.
2 — Includes few required costs or has major inaccuracies.
1 — Fails to provide any meaningful cost data.

#### B. Coverage of Platforms (0.30)
Measures whether the agent used all required platforms to gather data.

5 — Uses all three platforms (skyscanner.com, booking.com, numbeo.com) with evidence.
4 — Uses two platforms with evidence, or all three with incomplete data.
3 — Uses one platform with evidence, or two with incomplete data.
2 — Uses one platform with incomplete data.
1 — Fails to use any required platforms.

#### C. Depth of Detail (0.20)
Measures the specificity and granularity of the provided data.

5 — Provides detailed breakdowns for airfare, hotel, and daily expenses.
4 — Provides detailed breakdowns for most categories but lacks minor details.
3 — Provides partial breakdowns but lacks significant details.
2 — Provides minimal details or vague estimates.
1 — Fails to provide any meaningful breakdowns.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the response.

5 — Output is well-organized, structured, and uses credible sources.
4 — Output is organized but has minor structural issues or unclear sourcing.
3 — Output is partially organized but lacks clarity or credible sourcing.
2 — Output is poorly organized or lacks credible sources.
1 — Output is disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_detail": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_detail": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_detail": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())