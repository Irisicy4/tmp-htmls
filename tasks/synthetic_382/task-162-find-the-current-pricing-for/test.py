"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Determine the cheapest option for serverless functions pricing across AWS Lambda, Azure Functions, and Google Cloud Functions.
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


TASK_INSTRUCTION = """Find the current pricing for AWS Lambda, Azure Functions, and Google Cloud Functions for 1 million requests per month. Calculate the total monthly cost for 1 million requests and 2 GB memory usage per function instance across platforms, and determine which is the cheapest option."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to find pricing details for AWS Lambda, Azure Functions, and Google Cloud Functions, calculate the total monthly cost for 1 million requests and 2 GB memory usage per function instance, and identify the cheapest option. This task is in the domain of software engineering and cloud computing.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find the current pricing for AWS Lambda, Azure Functions, and Google Cloud Functions for 1 million requests per month. Calculate the total monthly cost for 1 million requests and 2 GB memory usage per function instance across platforms, and determine which is the cheapest option.

## Task-Specific Constraints
- Must visit the pricing pages for AWS Lambda, Azure Functions, and Google Cloud Functions.
- Must extract pricing data for 1 million requests and 2 GB memory usage per function instance.
- Must include price data for all three platforms in the comparison.
- Output must be organized as a table or structured list.
- Must clearly identify the cheapest option based on the calculated costs.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are pricing details for 1 million requests and 2 GB memory usage present for all three platforms?
- Is the output organized as a table or structured list?
- Did the agent correctly calculate the total monthly costs for each platform?
- Did the agent correctly identify the cheapest option?

### Step 2: Dimension Scoring

#### A. Pricing Accuracy (0.35)
Measures whether the agent correctly extracted and calculated the pricing details.

5 — Extracts correct pricing for all three platforms and calculates total costs accurately.
4 — Extracts pricing for all three platforms but has minor calculation errors.
3 — Extracts pricing for at least two platforms and calculates costs partially.
2 — Extracts pricing for only one platform or has major calculation errors.
1 — Fails to extract any pricing or calculate costs.

#### B. Platform Coverage (0.30)
Measures whether the agent included all required platforms in the comparison.

5 — Includes pricing data for all three platforms.
4 — Includes pricing data for two platforms.
3 — Includes pricing data for at least one platform.
2 — Mentions platforms but provides no pricing data.
1 — Fails to mention or include any platforms.

#### C. Cost Analysis Depth (0.25)
Measures the depth and specificity of the cost comparison.

5 — Provides detailed cost breakdowns and comparisons for all platforms.
4 — Provides cost breakdowns for most platforms but lacks some detail.
3 — Provides partial cost breakdowns or comparisons.
2 — Provides minimal cost analysis or vague comparisons.
1 — Provides no cost analysis or comparisons.

#### D. Output Structure (0.10)
Measures the organization and clarity of the output.

5 — Output is well-organized as a table or structured list with clear conclusions.
4 — Output is organized but lacks some clarity or structure.
3 — Output is partially organized but difficult to follow.
2 — Output is poorly organized or unclear.
1 — Output is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "pricing_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "cost_analysis_depth": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "pricing_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "cost_analysis_depth": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "pricing_accuracy": 0.35,
    "platform_coverage": 0.30,
    "cost_analysis_depth": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())