"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Verify whether the Anker PowerCore 10000 portable power bank is consistently priced across Amazon, Walmart, and Target.
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


TASK_INSTRUCTION = """Verify whether a specific portable power bank (Anker PowerCore 10000) is consistently priced across Amazon, Walmart, and Target. Extract the listed prices and check for major price discrepancies or promotions."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to verify the price consistency of the Anker PowerCore 10000 portable power bank across Amazon, Walmart, and Target. The agent must extract the listed prices from each platform and identify any major price discrepancies or promotions.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether a specific portable power bank (Anker PowerCore 10000) is consistently priced across Amazon, Walmart, and Target. Extract the listed prices and check for major price discrepancies or promotions.

## Task-Specific Constraints
- Must visit Amazon, Walmart, and Target platforms.
- Must extract and include price data for the Anker PowerCore 10000 from each platform.
- Output must be organized as a structured table or list for clarity.
- Must highlight any major price discrepancies or promotions clearly.
- Must include evidence of platform navigation in the tool-call trace.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Walmart, and Target? Which platforms were actually visited?
- Are the prices for the Anker PowerCore 10000 extracted and included in the response?
- Is the output organized as a structured table or list?
- Are any major price discrepancies or promotions highlighted clearly?
- Does the tool-call trace provide evidence of platform navigation?

### Step 2: Dimension Scoring

#### A. Price Accuracy (0.35)
Measures whether the agent correctly extracted and reported the listed prices for the Anker PowerCore 10000.

5 — Prices are accurate for all three platforms and match the tool-call trace.
4 — Prices are accurate for two platforms; minor errors in one.
3 — Prices are included but contain significant errors or omissions.
2 — Prices are mostly missing or incorrect.
1 — Prices are completely absent or wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — All three platforms visited and data included.
4 — Two platforms visited and data included; one missing.
3 — At least one platform visited and data included; others missing.
2 — Platforms visited but no data included.
1 — No platforms visited or data included.

#### C. Discrepancy Analysis (0.20)
Measures whether the agent identified and highlighted major price discrepancies or promotions.

5 — Discrepancies and promotions clearly identified and explained.
4 — Discrepancies or promotions identified but explanation incomplete.
3 — Discrepancies or promotions mentioned but unclear or incomplete.
2 — Discrepancies or promotions mostly missing or incorrect.
1 — Discrepancies or promotions completely absent.

#### D. Output Structure (0.15)
Measures whether the response is well-organized and easy to interpret.

5 — Output is structured as a clear table or list; easy to interpret.
4 — Output is structured but slightly unclear or inconsistent.
3 — Output is minimally structured; partially unclear.
2 — Output is poorly structured and difficult to interpret.
1 — Output is completely unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "price_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "discrepancy_analysis": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "price_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "discrepancy_analysis": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "price_accuracy": 0.35,
    "platform_coverage": 0.30,
    "discrepancy_analysis": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())