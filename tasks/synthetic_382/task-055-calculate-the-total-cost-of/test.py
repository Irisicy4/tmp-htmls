"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Compare the total cost and benefits of purchasing a 50-inch 4K television across Amazon, Walmart, and Best Buy.
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


TASK_INSTRUCTION = """Calculate the total cost of purchasing a 50-inch 4K television on Amazon, including shipping costs, warranty, and accessory bundles. Compare this to similar offerings on Walmart and Best Buy, and provide a recommendation based on total cost and included benefits."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost of purchasing a 50-inch 4K television on Amazon, including shipping costs, warranty, and accessory bundles. The agent must then compare this to similar offerings on Walmart and Best Buy, and provide a recommendation based on total cost and included benefits. This task is in the domain of online shopping.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of purchasing a 50-inch 4K television on Amazon, including shipping costs, warranty, and accessory bundles. Compare this to similar offerings on Walmart and Best Buy, and provide a recommendation based on total cost and included benefits.

## Task-Specific Constraints
- Must visit Amazon, Walmart, and Best Buy platforms.
- Must include price data for the television, shipping, warranty, and accessory bundles for all platforms.
- Output must be organized as a table or structured list.
- Must provide a clear recommendation based on total cost and included benefits.
- Must cite specific numerical values for costs and benefits in the comparison.
- Must ensure all data is accurate and sourced from the platforms visited.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are price data for television, shipping, warranty, and accessory bundles present for all platforms?
- Is the output organized as a table or structured list?
- Does the response include a clear recommendation based on total cost and included benefits?
- Are numerical values cited accurate and sourced from the platforms visited?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly calculated and compared total costs and benefits.

5 — Correctly calculates and compares total costs and benefits for all platforms, with accurate numerical values.
4 — Mostly correct calculations and comparisons, with minor errors or omissions.
3 — Partially correct calculations and comparisons, with significant gaps or inaccuracies.
2 — Mostly incorrect calculations and comparisons, with little usable information.
1 — No attempt or completely incorrect calculations and comparisons.

#### B. Coverage of Platforms and Items (0.30)
Measures whether the agent included all required platforms and items in the comparison.

5 — Includes all platforms (Amazon, Walmart, Best Buy) and all required items (television, shipping, warranty, accessory bundles).
4 — Includes most platforms and items, with minor omissions.
3 — Includes some platforms and items, but with significant omissions.
2 — Includes few platforms or items, with major omissions.
1 — Includes no platforms or items.

#### C. Depth and Specificity of Comparison (0.25)
Measures the level of detail and specificity in the comparison.

5 — Provides detailed comparisons with specific numerical values and benefits for all platforms.
4 — Provides mostly detailed comparisons, with minor gaps in specificity.
3 — Provides basic comparisons, with significant gaps in detail or specificity.
2 — Provides minimal comparisons, with little detail or specificity.
1 — Provides no meaningful comparisons.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and cites credible sources.

5 — Response is well-organized, formatted as a table or structured list, and cites credible sources.
4 — Response is mostly well-organized, with minor formatting or credibility issues.
3 — Response is partially organized, with significant formatting or credibility issues.
2 — Response is poorly organized, with little formatting or credibility.
1 — Response is completely disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_platforms_and_items": <1-5>,
  "depth_and_specificity_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_items": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_platforms_and_items": 0.30,
    "depth_and_specificity_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())