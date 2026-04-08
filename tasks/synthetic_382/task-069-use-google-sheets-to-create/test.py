"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Compare costs of furnishing a living room using price data from IKEA, Wayfair, and Overstock, and create a budget tracker in Google Sheets.
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


TASK_INSTRUCTION = """Use Google Sheets to create a budget tracker comparing the costs of furnishing a living room. Use price data for sofas, coffee tables, and area rugs from IKEA, Wayfair, and Overstock. Include columns for item type, description, price, and retailer."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to gather price data for sofas, coffee tables, and area rugs from IKEA, Wayfair, and Overstock and create a budget tracker in Google Sheets. The tracker must include columns for item type, description, price, and retailer. A successful completion involves visiting all three platforms, collecting accurate price data for all required items, and organizing the data into a structured table.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create a budget tracker comparing the costs of furnishing a living room. Use price data for sofas, coffee tables, and area rugs from IKEA, Wayfair, and Overstock. Include columns for item type, description, price, and retailer.

## Task-Specific Constraints
- Must visit IKEA, Wayfair, and Overstock to collect price data.
- Must include price data for sofas, coffee tables, and area rugs.
- Output must be organized as a structured table with columns for item type, description, price, and retailer.
- Must provide accurate and specific price data for each item.
- Must include at least one example for each item type from each retailer.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to IKEA, Wayfair, and Overstock? Which platforms were actually visited?
- Are price data for sofas, coffee tables, and area rugs present in the response?
- Is the output organized as a structured table with the required columns?
- Are the prices and descriptions accurate and sourced from the specified retailers?
- Does the response include at least one example for each item type from each retailer?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the budget tracker is correct, complete, and organized as specified.

5 — Includes all required items, accurate prices, and structured table with all columns.
4 — Includes most required items and accurate prices; minor issues in organization.
3 — Includes some required items and prices; partially organized.
2 — Includes few required items or inaccurate prices; poorly organized.
1 — Missing required items, prices, or table structure.

#### B. Coverage of Platforms and Items (0.30)
Measures whether the agent visited all platforms and included all required items.

5 — Visited all three platforms and included all required items from each.
4 — Visited two platforms and included most required items.
3 — Visited one platform and included some required items.
2 — Visited one platform but included few required items.
1 — Did not visit any platform or include required items.

#### C. Depth and Specificity (0.25)
Measures the level of detail in the price data and item descriptions.

5 — Provides detailed descriptions and accurate prices for all items.
4 — Provides detailed descriptions and accurate prices for most items.
3 — Provides basic descriptions and prices for some items.
2 — Provides vague descriptions or inaccurate prices.
1 — Provides no descriptions or prices.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the output.

5 — Output is well-organized, structured, and credible.
4 — Output is mostly organized and credible; minor formatting issues.
3 — Output is partially organized; some credibility issues.
2 — Output is poorly organized or lacks credibility.
1 — Output is disorganized and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms_and_items": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_items": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms_and_items": 0.30,
    "depth_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())