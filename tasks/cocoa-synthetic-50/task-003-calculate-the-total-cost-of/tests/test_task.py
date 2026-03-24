"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Determine the cheapest platform for purchasing a new iPad Air bundle by calculating total costs across Apple, Amazon, and Best Buy.
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


TASK_INSTRUCTION = """Calculate the total cost of buying a new iPad Air (latest generation) with an Apple Pencil and Smart Keyboard Folio across Apple, Amazon, and Best Buy. Include base product prices, shipping fees, and sales tax (assume 7% sales tax for all platforms). Based on your calculations, recommend the cheapest platform for purchasing the full bundle."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost of purchasing a new iPad Air (latest generation) with an Apple Pencil and Smart Keyboard Folio from Apple, Amazon, and Best Buy. The agent must include base product prices, shipping fees, and a 7% sales tax in the calculations. The agent must then recommend the cheapest platform for purchasing the full bundle.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of buying a new iPad Air (latest generation) with an Apple Pencil and Smart Keyboard Folio across Apple, Amazon, and Best Buy. Include base product prices, shipping fees, and sales tax (assume 7% sales tax for all platforms). Based on your calculations, recommend the cheapest platform for purchasing the full bundle.

## Task-Specific Constraints
- Must visit Apple, Amazon, and Best Buy to gather price data.
- Must include base prices, shipping fees, and a 7% sales tax in the calculations.
- Must provide a clear breakdown of costs for each platform.
- Must recommend the cheapest platform based on the calculations.
- Output must be structured as a table or list for clarity.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Apple, Amazon, and Best Buy? Which platforms were visited?
- Did the agent include base prices, shipping fees, and sales tax in the calculations?
- Did the agent provide a clear breakdown of costs for all three platforms?
- Is the output structured as a table or list for clarity?
- Did the agent correctly identify the cheapest platform based on the calculations?

### Step 2: Dimension Scoring

#### A. Cost Calculation Accuracy (0.35)
Measures whether the agent accurately calculated total costs, including base prices, shipping fees, and sales tax.

5 — All calculations are correct and include all required components for all platforms.
4 — Minor errors in calculations, but overall results are mostly accurate.
3 — Partial calculations present, but missing one or more key components (e.g., shipping fees or tax).
2 — Significant errors or omissions in calculations.
1 — No meaningful calculations provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered data from them.

5 — Data from all three platforms (Apple, Amazon, Best Buy) is present and correct.
4 — Data from all three platforms is present but contains minor inaccuracies.
3 — Data from only two platforms is present or mostly accurate.
2 — Data from only one platform is present or mostly inaccurate.
1 — No data from any platform is present.

#### C. Output Clarity and Structure (0.20)
Measures whether the agent's output is well-organized and easy to understand.

5 — Output is structured as a clear table or list with all required details.
4 — Output is mostly clear but could be better organized.
3 — Output is partially clear but lacks structure or detail.
2 — Output is poorly organized or difficult to follow.
1 — Output is completely unclear or absent.

#### D. Recommendation Accuracy (0.15)
Measures whether the agent correctly identified the cheapest platform based on its calculations.

5 — Recommendation is correct and clearly justified based on the calculations.
4 — Recommendation is correct but justification is unclear or incomplete.
3 — Recommendation is partially correct but contains errors or omissions.
2 — Recommendation is incorrect or poorly justified.
1 — No recommendation provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "cost_calculation_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "output_clarity_and_structure": <1-5>,
  "recommendation_accuracy": <1-5>,
  "dimension_reasoning": {{
    "cost_calculation_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "output_clarity_and_structure": "<one sentence citing specific evidence>",
    "recommendation_accuracy": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "cost_calculation_accuracy": 0.35,
    "platform_coverage": 0.30,
    "output_clarity_and_structure": 0.20,
    "recommendation_accuracy": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())