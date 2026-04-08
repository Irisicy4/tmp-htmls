"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Collect cloud hosting prices, calculate monthly VM costs, and recommend the most cost-efficient platform.
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


TASK_INSTRUCTION = """Collect recent cloud hosting prices for AWS EC2, Google Cloud Compute Engine, and Microsoft Azure VMs (standard configurations). Calculate the total cost of running a medium-tier VM 24/7 for a month on each platform. Recommend the most cost-efficient platform based on the calculation."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to collect cloud hosting prices for medium-tier virtual machines (VMs) from AWS EC2, Google Cloud Compute Engine, and Microsoft Azure. The agent must calculate the monthly cost of running each VM 24/7 and recommend the most cost-efficient platform. Success depends on accurate price data, correct calculations, and a clear recommendation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Collect recent cloud hosting prices for AWS EC2, Google Cloud Compute Engine, and Microsoft Azure VMs (standard configurations). Calculate the total cost of running a medium-tier VM 24/7 for a month on each platform. Recommend the most cost-efficient platform based on the calculation.

## Task-Specific Constraints
- Must visit aws.amazon.com, cloud.google.com, and azure.microsoft.com.
- Must include price data for medium-tier VMs from all three platforms.
- Must calculate monthly costs based on 24/7 usage.
- Output must include a clear recommendation of the most cost-efficient platform.
- Must provide structured output (e.g., table or list).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to aws.amazon.com, cloud.google.com, and azure.microsoft.com?
- Are price data for medium-tier VMs from all three platforms present in the response?
- Are the monthly cost calculations accurate and based on 24/7 usage?
- Is the output organized as a table or structured list?
- Does the response include a clear recommendation of the most cost-efficient platform?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly calculated monthly VM costs and provided a valid recommendation.

5 — All calculations are correct, and the recommendation is accurate.
4 — Minor errors in calculations or recommendation.
3 — Partial calculations or unclear recommendation.
2 — Significant errors in calculations or recommendation.
1 — No calculations or recommendation provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent included price data from all required platforms.

5 — Price data from AWS, Google Cloud, and Azure are fully included.
4 — Price data from two platforms are included.
3 — Price data from one platform is included.
2 — Price data is incomplete or incorrect.
1 — No price data is included.

#### C. Depth of Analysis (0.25)
Measures the level of detail in calculations and comparisons.

5 — Detailed calculations and comparisons are provided for all platforms.
4 — Comparisons are provided but lack some detail.
3 — Basic calculations are present but lack depth.
2 — Calculations are incomplete or overly simplistic.
1 — No calculations or comparisons are provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and uses credible sources.

5 — Output is well-structured and sources are clearly cited.
4 — Output is structured but lacks some source credibility.
3 — Output is partially structured or sources are unclear.
2 — Output is poorly structured or sources are missing.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())