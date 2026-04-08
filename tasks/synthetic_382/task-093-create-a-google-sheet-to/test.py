"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Research and organize a shopping list for home office supplies under $200 across three platforms.
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


TASK_INSTRUCTION = """Create a Google Sheet to organize a shopping list for home office supplies under $200. Research affordable options for a desk, ergonomic chair, and monitor stand across Ikea, Staples, and Amazon. Categorize each item in the sheet with its price, store name, and availability link."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research affordable home office supplies (desk, ergonomic chair, monitor stand) across three platforms (Ikea, Staples, Amazon) and organize the findings into a Google Sheet. The deliverable must include item names, prices, store names, and availability links, categorized in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a Google Sheet to organize a shopping list for home office supplies under $200. Research affordable options for a desk, ergonomic chair, and monitor stand across Ikea, Staples, and Amazon. Categorize each item in the sheet with its price, store name, and availability link.

## Task-Specific Constraints
- Must visit Ikea, Staples, and Amazon during the research process.
- Must include price data for all three items (desk, ergonomic chair, monitor stand).
- Output must be organized as a table in the Google Sheet.
- Each item must include a store name and availability link.
- All items must be under $200.
- Must provide evidence of research (e.g., URLs or screenshots).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Ikea, Staples, and Amazon? Which platforms were actually visited?
- Are all three items (desk, ergonomic chair, monitor stand) present in the response?
- Is the output structured as a table in the Google Sheet?
- Are price data, store names, and availability links included for each item?
- Are all items priced under $200?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the Google Sheet contains the required items with correct data.

5 — All three items are listed with accurate prices, store names, and availability links.
4 — All three items are listed, but one or two data points are incomplete or inaccurate.
3 — At least two items are listed with partial data (e.g., missing links or prices).
2 — Only one item is listed or data is largely incomplete.
1 — No items are listed or data is completely missing.

#### B. Platform Coverage (0.30)
Measures whether the agent researched all required platforms (Ikea, Staples, Amazon).

5 — All three platforms were visited and used to source items.
4 — Two platforms were visited and used to source items.
3 — At least one platform was visited and used to source items.
2 — Platforms were visited but no items were sourced.
1 — No platforms were visited.

#### C. Detail Specificity (0.25)
Measures the depth and specificity of the information provided.

5 — Includes detailed price comparisons, item descriptions, and links for all items.
4 — Includes detailed information for most items but lacks depth in one area.
3 — Includes partial details (e.g., missing comparisons or vague descriptions).
2 — Includes minimal details or vague information.
1 — No specific details are provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table, and sources are credible.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured or lacks clarity.
2 — Output is poorly structured or sources are questionable.
1 — Output is unstructured or sources are not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
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
    "detail_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())