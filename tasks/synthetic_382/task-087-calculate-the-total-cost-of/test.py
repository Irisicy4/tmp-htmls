"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Compare the total cost of purchasing a DSLR camera and accessories across three platforms and recommend the cheapest option.
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


TASK_INSTRUCTION = """Calculate the total cost of purchasing a DSLR camera from Amazon, including accessories (tripod and memory card) and shipping fees. Compare these costs with the same camera and accessory bundle on Adorama and B&H Photo Video. Recommend the platform with the lowest total cost, explaining the breakdown."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost of a DSLR camera and accessories (tripod and memory card) on three platforms: Amazon, Adorama, and B&H Photo Video. The agent must compare the costs, including shipping fees, and recommend the cheapest option with a clear breakdown of the costs.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of purchasing a DSLR camera from Amazon, including accessories (tripod and memory card) and shipping fees. Compare these costs with the same camera and accessory bundle on Adorama and B&H Photo Video. Recommend the platform with the lowest total cost, explaining the breakdown.

## Task-Specific Constraints
- Must visit Amazon, Adorama, and B&H Photo Video.
- Must include price data for the DSLR camera, tripod, memory card, and shipping fees on all platforms.
- Must calculate and compare total costs accurately.
- Output must include a clear breakdown of costs for each platform.
- Must recommend the platform with the lowest total cost, with justification.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Adorama, and B&H Photo Video?
- Did the agent include price data for the DSLR camera, tripod, memory card, and shipping fees on all platforms?
- Did the agent calculate and compare total costs accurately?
- Is the output organized with a clear breakdown of costs for each platform?
- Did the agent recommend the platform with the lowest total cost, with justification?

### Step 2: Dimension Scoring

#### A. Cost Calculation Accuracy (0.35)
Measures whether the agent accurately calculated the total costs for all platforms.

5 — Accurately calculates total costs for all platforms, including all required items and shipping fees.
4 — Accurately calculates total costs for most platforms, but minor errors or omissions exist.
3 — Calculates total costs for at least two platforms, but with significant errors or missing items.
2 — Attempts cost calculation but with major errors or missing data.
1 — Does not calculate costs or calculations are completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — Includes data from all three platforms (Amazon, Adorama, B&H Photo Video).
4 — Includes data from two platforms, with minor omissions.
3 — Includes data from at least one platform, but with significant omissions.
2 — Attempts to include platform data but fails to provide usable information.
1 — Does not include any platform data.

#### C. Recommendation Justification (0.25)
Measures whether the agent provided a clear and logical recommendation based on the cost comparison.

5 — Provides a clear recommendation with a detailed breakdown and logical justification.
4 — Provides a recommendation with some justification, but lacks detail or clarity.
3 — Provides a recommendation, but justification is weak or unclear.
2 — Attempts a recommendation, but it is illogical or unsupported.
1 — Does not provide a recommendation.

#### D. Output Organization (0.10)
Measures whether the output is well-structured and easy to understand.

5 — Output is well-structured, with a clear breakdown of costs and comparisons.
4 — Output is mostly well-structured, but minor formatting issues exist.
3 — Output is somewhat structured, but lacks clarity or organization.
2 — Output is poorly structured and difficult to follow.
1 — Output is unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "cost_calculation_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "recommendation_justification": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "cost_calculation_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "recommendation_justification": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "cost_calculation_accuracy": 0.35,
    "platform_coverage": 0.30,
    "recommendation_justification": 0.25,
    "output_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())