"""
LLM-as-judge evaluator for EvolveBench task.

Category: Logistics & Supply Chain
Task: Calculate the total landed cost of importing 500 units of a product from Shenzhen, China, to Los Angeles, USA.
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


TASK_INSTRUCTION = """Calculate the total landed cost of importing 500 units of a product from Shenzhen, China, to Los Angeles, USA. Use Alibaba.com to find the average CIF (Cost, Insurance, and Freight) quote for a product category (e.g., electronics, HS Code 8542). Use the USITC Harmonized Tariff Schedule (https://hts.usitc.gov/) to find the applicable duty rate. Finally, calculate the total landed cost per unit by adding customs duty and assuming a local fulfillment fee of $2/unit."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate the total landed cost of importing 500 units of a product from Shenzhen, China, to Los Angeles, USA. The agent must gather CIF quotes from Alibaba.com, find the applicable duty rate from the USITC Harmonized Tariff Schedule, and calculate the total landed cost per unit, including a local fulfillment fee. A successful completion includes accurate data collection, correct calculations, and a structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total landed cost of importing 500 units of a product from Shenzhen, China, to Los Angeles, USA. Use Alibaba.com to find the average CIF (Cost, Insurance, and Freight) quote for a product category (e.g., electronics, HS Code 8542). Use the USITC Harmonized Tariff Schedule (https://hts.usitc.gov/) to find the applicable duty rate. Finally, calculate the total landed cost per unit by adding customs duty and assuming a local fulfillment fee of $2/unit.

## Task-Specific Constraints
- Must visit Alibaba.com to gather CIF quotes for the specified product category.
- Must use the USITC Harmonized Tariff Schedule to find the correct duty rate.
- Must calculate the total landed cost per unit accurately, including CIF, customs duty, and fulfillment fee.
- Output must include a breakdown of costs (CIF, customs duty, fulfillment fee).
- Must provide sources or references for CIF quotes and duty rate.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Alibaba.com and gather CIF quotes? Were the quotes relevant to the product category?
- Did the agent use the USITC Harmonized Tariff Schedule to find the correct duty rate?
- Did the agent calculate the total landed cost per unit correctly, including all components (CIF, customs duty, fulfillment fee)?
- Is the output structured and does it include a breakdown of costs?
- Are the sources or references for CIF quotes and duty rate included and credible?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the total landed cost calculation is correct and complete.

5 — Includes accurate CIF, customs duty, and fulfillment fee; calculation is correct.
4 — Includes most components but with minor inaccuracies.
3 — Includes partial components or has significant calculation errors.
2 — Includes few components or major calculation errors.
1 — Does not attempt the calculation or is entirely incorrect.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent used all required platforms and gathered relevant data.

5 — Uses both Alibaba.com and USITC Harmonized Tariff Schedule; data is relevant.
4 — Uses both platforms but data is incomplete or partially relevant.
3 — Uses one platform or gathers incomplete data from both.
2 — Attempts to use platforms but fails to gather relevant data.
1 — Does not use the required platforms.

#### C. Depth and Specificity (0.20)
Measures the level of detail in the response, including cost breakdowns and references.

5 — Provides detailed breakdown of CIF, customs duty, and fulfillment fee; includes sources.
4 — Provides a breakdown with minor omissions or lacks some references.
3 — Provides a basic breakdown but lacks detail or sources.
2 — Provides minimal detail or no breakdown.
1 — Provides no detail or breakdown.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to understand.

5 — Output is clear, structured, and easy to follow.
4 — Output is mostly clear but has minor structural issues.
3 — Output is somewhat clear but lacks organization.
2 — Output is poorly structured or difficult to follow.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())