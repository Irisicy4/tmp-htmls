"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Use the Nike shoe-finder tool to identify running shoes for men with size 11 feet, under $100, filtered by 'neutral stability' and lightweight, and report the top 3 models with prices and descriptions.
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


TASK_INSTRUCTION = """Use the Nike shoe-finder tool (Nike.com) to find running shoes for men with size 11 feet and under $100, then apply filters for 'neutral stability' and lightweight. Report the top 3 shoe models displayed with their prices and descriptions."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is to use the Nike shoe-finder tool to identify running shoes for men with size 11 feet, under $100, filtered by 'neutral stability' and lightweight. The agent must report the top 3 shoe models with their prices and descriptions. A successful completion requires accurate filtering, correct price constraints, and clear, structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Nike shoe-finder tool (Nike.com) to find running shoes for men with size 11 feet and under $100, then apply filters for 'neutral stability' and lightweight. Report the top 3 shoe models displayed with their prices and descriptions.

## Task-Specific Constraints
- Must use the Nike shoe-finder tool (Nike.com) to perform the search.
- Must apply the filters: size 11, under $100, 'neutral stability', and lightweight.
- Must report exactly 3 shoe models, including their names, prices, and descriptions.
- Must ensure all reported shoes meet the specified filters.
- Output must be structured as a clear list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use the Nike shoe-finder tool (Nike.com)?
- Did the agent apply all required filters (size 11, under $100, 'neutral stability', lightweight)?
- Are exactly 3 shoe models reported, with names, prices, and descriptions?
- Do all reported shoes meet the specified filters?
- Is the output structured as a clear list or table?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent correctly applied all required filters.

5 — All filters (size 11, under $100, 'neutral stability', lightweight) are applied correctly.
4 — One filter is slightly incorrect or missing.
3 — Two filters are incorrect or missing.
2 — Most filters are incorrect or missing.
1 — No filters are applied correctly.

#### B. Output Completeness (0.30)
Measures whether the agent reports exactly 3 shoe models with all required details.

5 — Exactly 3 models are reported, with names, prices, and descriptions.
4 — 3 models are reported, but one detail (name, price, or description) is incomplete.
3 — Fewer than 3 models are reported, or multiple details are incomplete.
2 — Only 1 model is reported, or details are mostly missing.
1 — No models are reported.

#### C. Price and Filter Consistency (0.20)
Measures whether all reported shoes meet the specified filters.

5 — All reported shoes meet the filters (size, price, stability, weight).
4 — One shoe slightly deviates from the filters.
3 — Two shoes deviate from the filters.
2 — Most shoes deviate from the filters.
1 — None of the reported shoes meet the filters.

#### D. Output Structure and Clarity (0.15)
Measures whether the response is well-organized and easy to understand.

5 — Output is structured as a clear list or table, easy to read and understand.
4 — Output is mostly clear but has minor formatting issues.
3 — Output is partially clear but difficult to follow.
2 — Output is poorly structured and hard to understand.
1 — Output is unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filtering_accuracy": <1-5>,
  "output_completeness": <1-5>,
  "price_and_filter_consistency": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "filtering_accuracy": "<one sentence citing specific evidence>",
    "output_completeness": "<one sentence citing specific evidence>",
    "price_and_filter_consistency": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "output_completeness": 0.30,
    "price_and_filter_consistency": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())