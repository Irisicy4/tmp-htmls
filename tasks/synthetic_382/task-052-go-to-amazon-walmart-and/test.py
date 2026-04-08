"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Search for electric kitchen blenders under $80 on Amazon, Walmart, and eBay, filter by 4-star ratings, and extract product names, prices, and seller ratings for the top 5 results on each platform.
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


TASK_INSTRUCTION = """Go to Amazon, Walmart, and eBay, and search for electric kitchen blenders under $80. Use each site's filters to narrow results to models with at least 4-star ratings and extract product names, prices, and seller ratings for the top 5 results on each platform."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves searching for electric kitchen blenders under $80 on Amazon, Walmart, and eBay. The agent must filter results to only include models with at least 4-star ratings and extract product names, prices, and seller ratings for the top 5 results on each platform. A successful completion requires accurate data extraction and proper organization of the output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Amazon, Walmart, and eBay, and search for electric kitchen blenders under $80. Use each site's filters to narrow results to models with at least 4-star ratings and extract product names, prices, and seller ratings for the top 5 results on each platform.

## Task-Specific Constraints
- Must visit Amazon, Walmart, and eBay.
- Must filter results to include only models with at least 4-star ratings.
- Must extract product names, prices, and seller ratings for the top 5 results on each platform.
- Output must be structured as a table or a structured list.
- Must include all three data points (name, price, seller rating) for each product.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Walmart, and eBay? Which platforms were actually visited?
- Did the agent filter results to include only models with at least 4-star ratings?
- Are product names, prices, and seller ratings included for each product?
- Is the output structured as a table or structured list?
- Are there exactly 5 results per platform, and are they under $80?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the extracted product data (names, prices, seller ratings) is accurate and matches the task requirements.

5 — All data points are accurate and meet the task requirements.
4 — Minor inaccuracies in 1-2 data points but overall correct.
3 — Some inaccuracies or missing data, but partially usable.
2 — Significant inaccuracies or missing data in most results.
1 — Data is completely incorrect or missing.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and extracted data from each.

5 — Data extracted from all three platforms (Amazon, Walmart, eBay).
4 — Data extracted from two platforms.
3 — Data extracted from one platform.
2 — Attempted but failed to extract usable data from any platform.
1 — Did not attempt to visit any platform.

#### C. Filtering and Relevance (0.20)
Measures whether the agent correctly filtered results to meet the specified criteria (under $80, at least 4-star ratings).

5 — All results meet the criteria.
4 — Most results meet the criteria, with minor exceptions.
3 — Some results meet the criteria, but others are irrelevant.
2 — Few results meet the criteria.
1 — No results meet the criteria.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-structured and easy to interpret.

5 — Output is perfectly structured as a table or structured list.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but difficult to interpret.
2 — Output is poorly structured or disorganized.
1 — Output is completely unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "filtering_and_relevance": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "filtering_and_relevance": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "platform_coverage": 0.30,
    "filtering_and_relevance": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())