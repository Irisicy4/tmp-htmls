"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Fetch and compare the monthly costs of running AWS EC2, Azure VMs, and Google Cloud VMs for a specific configuration.
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


TASK_INSTRUCTION = """Fetch the latest pricing for Amazon Web Services (AWS) EC2 instances, Azure Virtual Machines, and Google Cloud VMs for a similar configuration (8 vCPUs, 32GB RAM, Linux OS). Calculate the monthly cost for running these instances 24/7 and recommend the most cost-effective option."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves fetching pricing information for cloud computing services from three platforms (AWS, Azure, Google Cloud) for a specific configuration (8 vCPUs, 32GB RAM, Linux OS). The agent must calculate monthly costs for 24/7 usage and recommend the most cost-effective option. Success requires accurate data retrieval, correct calculations, and a clear recommendation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch the latest pricing for Amazon Web Services (AWS) EC2 instances, Azure Virtual Machines, and Google Cloud VMs for a similar configuration (8 vCPUs, 32GB RAM, Linux OS). Calculate the monthly cost for running these instances 24/7 and recommend the most cost-effective option.

## Task-Specific Constraints
- Must visit aws.amazon.com, azure.microsoft.com, and cloud.google.com.
- Must include pricing data for all three platforms.
- Output must include monthly cost calculations for 24/7 usage.
- Must recommend the most cost-effective option based on the calculations.
- Output must be organized in a structured format (e.g., table or list).
- Pricing data must match the specified configuration.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to aws.amazon.com, azure.microsoft.com, and cloud.google.com? Which ones were actually visited?
- Does the response include pricing data for all three platforms?
- Are the monthly cost calculations for 24/7 usage present and correct?
- Is the recommendation for the most cost-effective option clearly stated?
- Is the output organized in a structured format (e.g., table or list)?

### Step 2: Dimension Scoring

#### A. Pricing Accuracy (0.35)
Measures whether the pricing data retrieved is correct and matches the specified configuration.

5 — Pricing data is accurate for all three platforms and matches the specified configuration.
4 — Pricing data is accurate for two platforms; minor errors for the third.
3 — Pricing data is partially accurate; significant errors or missing data for one platform.
2 — Pricing data is mostly incorrect or missing for multiple platforms.
1 — Pricing data is absent or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included their pricing data.

5 — All three platforms were visited, and pricing data is included for all.
4 — Two platforms were visited, and pricing data is included for both.
3 — At least one platform was visited, and pricing data is included for it.
2 — Platforms were visited but no pricing data was included.
1 — No platforms were visited, or no pricing data was included.

#### C. Cost Calculation Accuracy (0.25)
Measures whether the monthly cost calculations for 24/7 usage are correct.

5 — Monthly cost calculations are correct for all three platforms.
4 — Monthly cost calculations are correct for two platforms; minor errors for the third.
3 — Monthly cost calculations are partially correct; significant errors for one platform.
2 — Monthly cost calculations are mostly incorrect or missing for multiple platforms.
1 — Monthly cost calculations are absent or completely incorrect.

#### D. Output Structure and Recommendation Clarity (0.10)
Measures whether the output is well-organized and includes a clear recommendation.

5 — Output is structured (e.g., table or list) and includes a clear, correct recommendation.
4 — Output is structured but the recommendation is unclear or slightly incorrect.
3 — Output is partially structured; recommendation is present but unclear.
2 — Output is unstructured; recommendation is missing or incorrect.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "pricing_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "cost_calculation_accuracy": <1-5>,
  "output_structure_and_recommendation_clarity": <1-5>,
  "dimension_reasoning": {{
    "pricing_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "cost_calculation_accuracy": "<one sentence citing specific evidence>",
    "output_structure_and_recommendation_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "pricing_accuracy": 0.35,
    "platform_coverage": 0.30,
    "cost_calculation_accuracy": 0.25,
    "output_structure_and_recommendation_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())