"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Search for baby strollers under $200 on Walmart, apply filters for lightweight travel strollers, and extract top 5 results with details.
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
    aggregated["evidence_summary"] = valid[0].get("evidence_summary", "")
    aggregated["dimension_reasoning"] = valid[0].get("dimension_reasoning", {})
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


TASK_INSTRUCTION = """Go to the Walmart website and search for baby strollers under $200. Apply filters for strollers that are lightweight and designed for travel. Extract the top 5 matching results with their prices, ratings, and seller details."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to search for baby strollers under $200 on Walmart, apply filters for lightweight and travel strollers, and extract the top 5 results. The deliverable must include the price, rating, and seller details for each stroller.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the Walmart website and search for baby strollers under $200. Apply filters for strollers that are lightweight and designed for travel. Extract the top 5 matching results with their prices, ratings, and seller details.

## Task-Specific Constraints
- Must search on walmart.com.
- Must apply filters for lightweight and travel strollers.
- Must extract exactly 5 results.
- Each result must include price, rating, and seller details.
- Output must be structured as a list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to walmart.com and perform the required search?
- Did the agent apply the filters for lightweight and travel strollers?
- Did the agent extract exactly 5 results?
- Does each result include price, rating, and seller details?
- Is the output structured as a list or table?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and extracted the required strollers.

5 — Extracts 5 strollers with all required details (price, rating, seller) accurately.
4 — Extracts 4-5 strollers with minor inaccuracies in details.
3 — Extracts 3-4 strollers with partial details missing.
2 — Extracts fewer than 3 strollers or major inaccuracies.
1 — Fails to extract relevant strollers or details.

#### B. Coverage of Filters (0.30)
Measures whether the agent applied the required filters correctly.

5 — Applies both lightweight and travel filters correctly.
4 — Applies one filter correctly and partially applies the other.
3 — Applies one filter correctly but misses the other.
2 — Attempts filters but applies them incorrectly.
1 — Does not apply any filters.

#### C. Detail Specificity (0.20)
Measures the completeness and specificity of the extracted details.

5 — Includes all required details (price, rating, seller) for all results.
4 — Includes most details but misses minor elements for 1-2 results.
3 — Includes partial details for most results.
2 — Includes minimal details or significant omissions.
1 — Includes no meaningful details.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-structured and easy to interpret.

5 — Output is structured as a clear list or table with all details.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but lacks clarity.
2 — Output is poorly structured or hard to interpret.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_filters": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_filters": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_filters": 0.30,
    "detail_specificity": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())