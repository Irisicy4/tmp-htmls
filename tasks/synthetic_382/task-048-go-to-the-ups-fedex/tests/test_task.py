"""
LLM-as-judge evaluator for EvolveBench task.

Category: Logistics & Supply Chain
Task: Extract express shipping rates and estimated delivery times for a package from UPS, FedEx, and DHL.
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


TASK_INSTRUCTION = """Go to the UPS, FedEx, and DHL express shipping rate calculators. Input the same shipment details: a 2 kg package with dimensions 30x20x15 cm being shipped from New York, USA (ZIP 10001) to London, UK (ZIP WC2N). Extract the quoted express shipping rates and estimated delivery times from each site."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit the shipping rate calculators of UPS, FedEx, and DHL, input specific shipment details, and extract express shipping rates and estimated delivery times. This task is in the Logistics & Supply Chain domain. A successful completion involves providing accurate rates and delivery times from all three platforms in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the UPS, FedEx, and DHL express shipping rate calculators. Input the same shipment details: a 2 kg package with dimensions 30x20x15 cm being shipped from New York, USA (ZIP 10001) to London, UK (ZIP WC2N). Extract the quoted express shipping rates and estimated delivery times from each site.

## Task-Specific Constraints
- Must visit all three platforms: UPS, FedEx, and DHL.
- Must input the correct shipment details (weight, dimensions, origin, and destination).
- Must extract both the express shipping rates and estimated delivery times from each platform.
- Output must be organized as a structured table or list.
- Must ensure the extracted data is accurate and matches the platform results.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to UPS, FedEx, and DHL platforms? Which ones were actually visited?
- Did the agent input the correct shipment details (weight, dimensions, origin, and destination)?
- Are express shipping rates and estimated delivery times present in the response?
- Is the output organized as a structured table or list?
- Are the extracted rates and delivery times accurate and sourced correctly?

### Step 2: Dimension Scoring

#### A. Carrier Comparison Accuracy (0.35)
Measures whether the agent correctly extracted rates and delivery times from all three platforms.

5 — Rates and delivery times from all three platforms are accurate and complete.
4 — Rates and delivery times from two platforms are accurate; minor issues with the third.
3 — Rates and delivery times from at least one platform are accurate; others incomplete.
2 — Rates and delivery times mostly missing or incorrect.
1 — No rates or delivery times extracted.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them correctly.

5 — All three platforms were visited and used correctly.
4 — Two platforms were visited and used correctly; minor issues with the third.
3 — At least one platform was visited and used correctly; others incomplete.
2 — Platforms mostly missed or used incorrectly.
1 — No platforms were visited.

#### C. Data Specificity (0.20)
Measures whether the extracted data includes specific details (e.g., rates, delivery times, currency, units).

5 — All extracted data includes specific details (currency, units, etc.).
4 — Most data includes specific details; minor omissions.
3 — Some data includes specific details; others are vague or missing.
2 — Specific details mostly missing or incorrect.
1 — No specific details provided.

#### D. Output Structure (0.15)
Measures whether the response is organized in a clear and structured format.

5 — Response is organized as a structured table or list with clear labels.
4 — Response is mostly structured; minor formatting issues.
3 — Response is partially structured; lacks clarity or organization.
2 — Response is poorly structured or difficult to interpret.
1 — Response is unstructured or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "carrier_comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "data_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "carrier_comparison_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "carrier_comparison_accuracy": 0.35,
    "platform_coverage": 0.30,
    "data_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())