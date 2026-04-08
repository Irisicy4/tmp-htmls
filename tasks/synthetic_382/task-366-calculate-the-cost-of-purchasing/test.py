"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Evaluate the cost-effectiveness of purchasing premium design assets for a small business campaign across three platforms.
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


TASK_INSTRUCTION = """Calculate the cost of purchasing premium design assets for a small business campaign using three popular asset sites. Include pricing for 5 icon packs, 2 font families, and 3 stock images per site. Recommend the most cost-effective option for a $200 budget."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the cost of purchasing premium design assets across three specified platforms (flaticon.com, envato.com, adobestock.com). The agent must include pricing for 5 icon packs, 2 font families, and 3 stock images per site, and recommend the most cost-effective option within a $200 budget.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the cost of purchasing premium design assets for a small business campaign using three popular asset sites. Include pricing for 5 icon packs, 2 font families, and 3 stock images per site. Recommend the most cost-effective option for a $200 budget.

## Task-Specific Constraints
- Must visit flaticon.com, envato.com, and adobestock.com.
- Must include price data for 5 icon packs, 2 font families, and 3 stock images per site.
- Output must be organized as a structured table or list.
- Must provide a recommendation based on the $200 budget constraint.
- Must include evidence of price comparisons across platforms.
- Recommendation must be supported by numerical calculations.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to flaticon.com, envato.com, and adobestock.com? Which ones were actually visited?
- Are price data for 5 icon packs, 2 font families, and 3 stock images per site present in the response?
- Is the output organized as a structured table or list?
- Does the recommendation align with the $200 budget constraint?
- Are numerical calculations accurate and supported by evidence?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's recommendation is correct and aligns with the $200 budget constraint.

5 — Recommendation is correct, aligns with the budget, and includes accurate calculations.
4 — Recommendation is correct but calculations have minor errors.
3 — Recommendation is partially correct or calculations are incomplete.
2 — Recommendation is incorrect or calculations are mostly missing.
1 — No recommendation or calculations provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and included price data for all required items.

5 — All three platforms visited and price data for all items included.
4 — Two platforms visited and price data for most items included.
3 — At least one platform visited and partial price data included.
2 — Minimal platform visits and price data missing.
1 — No platform visits or price data included.

#### C. Depth of Analysis (0.20)
Measures the level of detail in price comparisons and numerical calculations.

5 — Detailed price comparisons and calculations for all items.
4 — Detailed comparisons but minor gaps in calculations.
3 — Basic comparisons with limited calculations.
2 — Minimal comparisons and calculations.
1 — No comparisons or calculations provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and supported by credible evidence.

5 — Output is structured, well-organized, and includes credible evidence.
4 — Output is structured but minor organizational issues or gaps in evidence.
3 — Output is partially structured with limited evidence.
2 — Output is poorly organized and lacks credible evidence.
1 — Output is unstructured and unsupported.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
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
    "depth_of_analysis": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())