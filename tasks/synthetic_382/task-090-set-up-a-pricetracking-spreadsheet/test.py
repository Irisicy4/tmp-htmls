"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Set up a price-tracking spreadsheet for home kitchen products across multiple platforms.
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


TASK_INSTRUCTION = """Set up a price-tracking spreadsheet in Google Sheets for five home kitchen products (e.g., blender, toaster, coffee maker) from Amazon, Walmart, and Home Depot. Include columns for product name, current price, date last checked, and store link. Provide the publicly accessible link to the spreadsheet."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to set up a price-tracking spreadsheet in Google Sheets for five home kitchen products from Amazon, Walmart, and Home Depot. The spreadsheet must include columns for product name, current price, date last checked, and store link. A publicly accessible link to the spreadsheet must be provided.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a price-tracking spreadsheet in Google Sheets for five home kitchen products (e.g., blender, toaster, coffee maker) from Amazon, Walmart, and Home Depot. Include columns for product name, current price, date last checked, and store link. Provide the publicly accessible link to the spreadsheet.

## Task-Specific Constraints
- Must visit Amazon, Walmart, and Home Depot to gather price data.
- Must include price data for five distinct home kitchen products.
- Spreadsheet must have columns for product name, current price, date last checked, and store link.
- Spreadsheet must be publicly accessible via a shared link.
- Output must be organized as a structured table in Google Sheets.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Walmart, and Home Depot? Which platforms were actually visited?
- Are five distinct home kitchen products included in the response?
- Does the spreadsheet contain the required columns (product name, current price, date last checked, store link)?
- Is the spreadsheet publicly accessible via the provided link?
- Is the output organized as a structured table in Google Sheets?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the spreadsheet is correct and complete.

5 — Spreadsheet includes all required columns and five distinct products with accurate price data.
4 — Spreadsheet includes all required columns but has minor inaccuracies or missing details.
3 — Spreadsheet includes most required columns but is incomplete or contains errors.
2 — Spreadsheet is mostly incorrect or missing key columns.
1 — No spreadsheet provided or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered data.

5 — Data gathered from Amazon, Walmart, and Home Depot for all five products.
4 — Data gathered from at least two platforms for all five products.
3 — Data gathered from one platform or incomplete data from multiple platforms.
2 — Minimal data gathered from platforms.
1 — No platform data gathered.

#### C. Data Specificity (0.20)
Measures the depth and specificity of the provided data.

5 — Includes detailed price data, store links, and accurate dates for all products.
4 — Includes most details but minor omissions in price or dates.
3 — Includes basic details but lacks depth or specificity.
2 — Sparse or vague data provided.
1 — No specific data provided.

#### D. Output Structure (0.15)
Measures the organization and accessibility of the spreadsheet.

5 — Spreadsheet is well-organized, clearly formatted, and publicly accessible.
4 — Spreadsheet is organized but has minor formatting issues or accessibility problems.
3 — Spreadsheet is usable but poorly formatted or difficult to access.
2 — Spreadsheet is disorganized or inaccessible.
1 — No spreadsheet provided or completely unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "data_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "data_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())