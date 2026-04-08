"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Calculate the total cost of buying a gaming console (PS5 or Xbox Series X) including shipping and extended warranties, and recommend the most cost-effective option.
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


TASK_INSTRUCTION = """Calculate the total cost of buying a gaming console (PS5 or Xbox Series X) including shipping and extended warranties. Use pricing data from Amazon and Newegg for the consoles and shipping costs, and get warranty prices from Best Buy. Recommend the most cost-effective option with a breakdown of the cost components."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost of buying a gaming console (PS5 or Xbox Series X), including shipping and extended warranties. The agent must gather pricing data from Amazon and Newegg for the consoles and shipping costs, and warranty prices from Best Buy. A successful completion includes a detailed breakdown of costs and a recommendation for the most cost-effective option.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of buying a gaming console (PS5 or Xbox Series X) including shipping and extended warranties. Use pricing data from Amazon and Newegg for the consoles and shipping costs, and get warranty prices from Best Buy. Recommend the most cost-effective option with a breakdown of the cost components.

## Task-Specific Constraints
- Must visit Amazon, Newegg, and Best Buy platforms.
- Must include price data for both PS5 and Xbox Series X consoles.
- Must include shipping costs and warranty prices in the calculation.
- Output must be organized as a structured list or table.
- Must provide a clear recommendation for the most cost-effective option with reasoning.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Newegg, and Best Buy platforms? Which ones were actually visited?
- Are price data for both PS5 and Xbox Series X consoles present in the response?
- Are shipping costs and warranty prices included in the calculations?
- Is the output organized as a structured list or table?
- Does the recommendation clearly identify the most cost-effective option with reasoning?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly calculates the total cost and provides a valid recommendation.

5 — Correctly calculates total costs for both consoles, includes all components, and provides a valid recommendation.
4 — Mostly correct calculations but minor omissions or errors in components.
3 — Partial calculations with significant omissions or errors.
2 — Incorrect calculations or invalid recommendation.
1 — No calculations attempted or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered necessary data.

5 — Successfully visits Amazon, Newegg, and Best Buy, and gathers all required data.
4 — Visits at least two platforms and gathers most required data.
3 — Visits one platform and gathers partial data.
2 — Visits platforms but gathers minimal or incorrect data.
1 — Does not visit any required platforms.

#### C. Detail and Specificity (0.25)
Measures the inclusion of detailed cost breakdowns and comparisons.

5 — Provides detailed breakdowns for all components (console price, shipping, warranty) and compares options.
4 — Provides most breakdowns but lacks minor details or comparisons.
3 — Provides partial breakdowns with significant omissions.
2 — Minimal breakdowns or vague comparisons.
1 — No breakdowns or comparisons provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and uses credible sources.

5 — Output is structured as a clear table or list, and sources are credible.
4 — Output is mostly structured but lacks minor clarity or credibility.
3 — Output is partially structured with noticeable issues in clarity or credibility.
2 — Output is poorly structured or sources are questionable.
1 — Output is unstructured and sources are not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())