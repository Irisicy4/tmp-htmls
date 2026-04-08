"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Verify pricing consistency and discounts for the latest Apple AirPods Pro across Amazon, Target, and Walmart.
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


TASK_INSTRUCTION = """Verify whether the latest model of the Apple AirPods Pro has consistent pricing across Amazon, Target, and Walmart. Check if sales or discounts are being offered. Provide a report with URLs and evidence for any price discrepancies found."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying pricing consistency and identifying discounts for the latest Apple AirPods Pro across Amazon, Target, and Walmart. The agent must provide a structured report with URLs and evidence for any discrepancies found.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the latest model of the Apple AirPods Pro has consistent pricing across Amazon, Target, and Walmart. Check if sales or discounts are being offered. Provide a report with URLs and evidence for any price discrepancies found.

## Task-Specific Constraints
- Must visit Amazon, Target, and Walmart platforms.
- Must include price data for the Apple AirPods Pro from all three platforms.
- Must identify and report any sales or discounts explicitly.
- Output must be organized as a structured table or list.
- Must provide URLs as evidence for all reported prices and discounts.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Target, and Walmart platforms? Which ones were actually visited?
- Are the prices for Apple AirPods Pro from all three platforms present in the response?
- Are sales or discounts explicitly identified and reported?
- Is the output organized as a structured table or list?
- Are URLs provided as evidence for all reported prices and discounts?

### Step 2: Dimension Scoring

#### A. Pricing Accuracy (0.35)
Measures whether the reported prices are correct and consistent with the evidence provided.

5 — All reported prices are accurate and consistent with the URLs provided.
4 — Most reported prices are accurate, with minor inconsistencies.
3 — Some reported prices are accurate, but key data is missing or incorrect.
2 — Few reported prices are accurate, with significant errors.
1 — No accurate prices reported.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — Data from Amazon, Target, and Walmart is fully included.
4 — Data from two platforms is fully included; minor omissions from the third.
3 — Data from at least two platforms is included, but incomplete.
2 — Data from only one platform is included.
1 — No platform data included.

#### C. Discount Identification (0.20)
Measures whether the agent correctly identifies and reports sales or discounts.

5 — All discounts or sales are accurately identified and reported.
4 — Most discounts or sales are identified, with minor omissions.
3 — Some discounts or sales are identified, but incomplete.
2 — Few discounts or sales are identified, with significant omissions.
1 — No discounts or sales identified.

#### D. Output Structure and Evidence (0.15)
Measures whether the output is well-organized and supported by credible evidence.

5 — Output is fully structured as a table or list, with URLs for all reported data.
4 — Output is mostly structured, with URLs for most reported data.
3 — Output is partially structured, with some URLs missing.
2 — Output is poorly structured, with few URLs provided.
1 — Output is unstructured, with no URLs provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "pricing_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "discount_identification": <1-5>,
  "output_structure_and_evidence": <1-5>,
  "dimension_reasoning": {{
    "pricing_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "discount_identification": "<one sentence citing specific evidence>",
    "output_structure_and_evidence": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "pricing_accuracy": 0.35,
    "platform_coverage": 0.30,
    "discount_identification": 0.20,
    "output_structure_and_evidence": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())