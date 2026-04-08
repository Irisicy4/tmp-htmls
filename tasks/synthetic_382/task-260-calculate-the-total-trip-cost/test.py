"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Calculate the total trip cost for a 5-day excursion to Thailand, including airfare, hotel stays, meals, and local transport.
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


TASK_INSTRUCTION = """Calculate the total trip cost for a 5-day excursion to Thailand (Bangkok and Phuket). Include airfare, mid-range hotel stays, daily meals, and local transport. Use data from Google Flights, Booking.com, and Numbeo for cost estimates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total trip cost for a 5-day excursion to Thailand (Bangkok and Phuket). This includes airfare, mid-range hotel stays, daily meals, and local transport. The agent must use data from Google Flights, Booking.com, and Numbeo to provide accurate cost estimates.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total trip cost for a 5-day excursion to Thailand (Bangkok and Phuket). Include airfare, mid-range hotel stays, daily meals, and local transport. Use data from Google Flights, Booking.com, and Numbeo for cost estimates.

## Task-Specific Constraints
- Must use data from Google Flights, Booking.com, and Numbeo.
- Must include cost estimates for airfare, hotel stays, meals, and local transport.
- Output must be organized as a structured list or table.
- Must provide specific numerical values for each cost category.
- Must calculate the total trip cost accurately by summing all categories.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Flights, Booking.com, and Numbeo? Which platforms were actually visited?
- Are cost estimates for airfare, hotel stays, meals, and local transport present in the response?
- Is the output organized as a structured list or table?
- Are the numerical values for each cost category accurate and sourced correctly?
- Is the total trip cost calculation correct?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the total trip cost is calculated correctly and includes all required categories.

5 — Includes accurate cost estimates for all categories and calculates the total correctly.
4 — Includes most categories with accurate estimates; minor errors in total calculation.
3 — Includes partial categories or has significant errors in total calculation.
2 — Includes few categories and/or major errors in total calculation.
1 — No meaningful attempt at calculating the total trip cost.

#### B. Coverage of Sources (0.30)
Measures whether the agent used all required platforms (Google Flights, Booking.com, Numbeo) and included relevant data.

5 — Uses all three platforms and incorporates relevant data from each.
4 — Uses two platforms and incorporates relevant data; minor omissions.
3 — Uses one platform or includes incomplete data from multiple platforms.
2 — Attempts to use platforms but fails to extract meaningful data.
1 — Does not use any required platforms.

#### C. Specificity of Data (0.25)
Measures whether the response includes detailed numerical values for each cost category.

5 — Provides detailed numerical values for all categories with clear sourcing.
4 — Provides numerical values for most categories; minor omissions or lack of detail.
3 — Provides numerical values for some categories; lacks detail or sourcing.
2 — Provides vague or incomplete numerical values.
1 — No numerical values provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and uses credible sources.

5 — Output is structured as a clear table or list; sources are credible.
4 — Output is mostly structured; minor formatting issues or unclear sourcing.
3 — Output is partially structured; significant formatting issues or unclear sourcing.
2 — Output is poorly structured and/or sources are questionable.
1 — Output is unstructured and sources are not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "specificity_of_data": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "specificity_of_data": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "specificity_of_data": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())