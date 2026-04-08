"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Calculate the estimated monthly cost of running an ML model inference pipeline using AWS Lambda, S3, and EC2, and recommend the most cost-effective setup.
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


TASK_INSTRUCTION = """Fetch today's AWS Lambda pricing for 1 million requests, the cost for processing 100GB of data in S3, and the GPU processing cost for a hypothetical EC2 instance (p3.2xlarge). Calculate the estimated monthly cost of running an ML model inference pipeline with these components. Recommend the most cost-effective setup."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves calculating the estimated monthly cost of running an ML model inference pipeline using AWS Lambda, S3, and EC2. The agent must fetch current pricing data from AWS and other platforms, perform calculations based on the provided constraints, and recommend the most cost-effective setup.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch today's AWS Lambda pricing for 1 million requests, the cost for processing 100GB of data in S3, and the GPU processing cost for a hypothetical EC2 instance (p3.2xlarge). Calculate the estimated monthly cost of running an ML model inference pipeline with these components. Recommend the most cost-effective setup.

## Task-Specific Constraints
- Must visit aws.amazon.com and fetch pricing data for Lambda, S3, and EC2.
- Must include price data for all three components in the calculation.
- Output must be organized as a table or structured list.
- Must calculate the total monthly cost based on the provided usage constraints.
- Must provide a clear recommendation for the most cost-effective setup.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to aws.amazon.com and fetch pricing data for Lambda, S3, and EC2?
- Are all three required pricing components (Lambda, S3, EC2) present in the response?
- Is the output organized as a table or structured list?
- Are the calculations for total monthly cost accurate and based on the provided constraints?
- Does the agent provide a clear recommendation for the most cost-effective setup?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent's calculations are correct and complete.

5 — All calculations are correct, and all required pricing components are included.
4 — Minor errors in calculations or missing one pricing component.
3 — Partial calculations or missing two pricing components.
2 — Incorrect calculations or missing most pricing components.
1 — No calculations attempted or completely incorrect.

#### B. Coverage of Required Platforms and Data (0.30)
Measures whether the agent visited the required platforms and fetched all necessary data.

5 — Agent visited all required platforms and fetched all necessary data.
4 — Agent visited most platforms but missed minor details.
3 — Agent visited some platforms but missed key data.
2 — Agent visited few platforms and fetched minimal data.
1 — Agent did not visit any required platforms or fetch data.

#### C. Depth and Specificity of Analysis (0.20)
Measures whether the agent provided detailed and specific analysis, including comparisons.

5 — Detailed analysis with specific comparisons and reasoning for recommendations.
4 — Good analysis but lacks minor details or comparisons.
3 — Basic analysis with minimal comparisons or reasoning.
2 — Poor analysis with little detail or reasoning.
1 — No analysis or reasoning provided.

#### D. Output Structure and Source Credibility (0.15)
Measures whether the output is well-organized and uses credible sources.

5 — Output is well-organized and all sources are credible.
4 — Output is mostly organized with minor credibility issues.
3 — Output is usable but poorly organized or lacks source credibility.
2 — Output is disorganized or uses questionable sources.
1 — Output is completely disorganized or lacks sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms_and_data": <1-5>,
  "depth_and_specificity_of_analysis": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms_and_data": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms_and_data": 0.30,
    "depth_and_specificity_of_analysis": 0.20,
    "output_structure_and_source_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())