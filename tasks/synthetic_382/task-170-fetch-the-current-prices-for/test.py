"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Fetch and compare monthly hosting costs for Linux instances on AWS, Google Cloud, and Azure, and recommend the cheapest option.
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


TASK_INSTRUCTION = """Fetch the current prices for hosting a small-scale project on AWS EC2, Google Cloud Compute Engine, and Microsoft Azure Virtual Machines. Calculate the monthly cost for running a Linux instance with 2 CPUs and 8 GB RAM, and recommend the cheapest option."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to fetch current pricing information for hosting a Linux instance with 2 CPUs and 8 GB RAM on AWS EC2, Google Cloud Compute Engine, and Microsoft Azure Virtual Machines. The agent must calculate the monthly cost for each platform and recommend the cheapest option. This is a Software Engineering task that involves data retrieval, comparison, and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch the current prices for hosting a small-scale project on AWS EC2, Google Cloud Compute Engine, and Microsoft Azure Virtual Machines. Calculate the monthly cost for running a Linux instance with 2 CPUs and 8 GB RAM, and recommend the cheapest option.

## Task-Specific Constraints
- Must visit all three specified platforms: AWS EC2, Google Cloud Compute Engine, and Microsoft Azure Virtual Machines.
- Must include price data for Linux instances with 2 CPUs and 8 GB RAM.
- Output must be organized as a table or structured list.
- Must calculate monthly costs based on hourly rates or other pricing models provided by the platforms.
- Must recommend the cheapest option explicitly.
- Must provide evidence or URLs for the pricing data retrieved.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are price data for Linux instances with 2 CPUs and 8 GB RAM present in the response?
- Is the output organized as a table or structured list?
- Did the agent calculate monthly costs correctly based on the pricing data?
- Did the agent explicitly recommend the cheapest option?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (pricing comparison and recommendation) is correct and complete.

5 — Correct pricing data for all three platforms and accurate recommendation of the cheapest option.
4 — Correct pricing data for at least two platforms and accurate recommendation.
3 — Correct pricing data for at least one platform; recommendation may be incomplete.
2 — Incorrect pricing data or recommendation; minimal effort shown.
1 — No pricing data or recommendation provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included their pricing data.

5 — Pricing data from all three platforms included.
4 — Pricing data from two platforms included.
3 — Pricing data from one platform included.
2 — Minimal or incorrect platform coverage.
1 — No platform coverage.

#### C. Calculation Specificity (0.25)
Measures whether the agent calculated monthly costs correctly and included detailed comparisons.

5 — Monthly costs calculated correctly for all platforms with detailed comparisons.
4 — Monthly costs calculated correctly for at least two platforms.
3 — Monthly costs calculated correctly for one platform; comparisons may lack detail.
2 — Incorrect or incomplete calculations; minimal comparisons.
1 — No calculations or comparisons provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and includes credible evidence or sources.

5 — Output is structured as a table or list and includes URLs or evidence for all platforms.
4 — Output is structured and includes evidence for at least two platforms.
3 — Output is structured but lacks evidence or sources.
2 — Output is poorly structured or lacks credibility.
1 — Output is disorganized and lacks evidence.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "calculation_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "calculation_specificity": "<one sentence citing specific evidence>",
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
    "calculation_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())