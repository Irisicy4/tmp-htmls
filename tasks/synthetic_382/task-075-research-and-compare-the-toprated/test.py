"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Research and compare top-rated noise-canceling headphones under $300 across Amazon, Best Buy, and Target, summarizing key features and prices, and identifying the best value for money.
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


TASK_INSTRUCTION = """Research and compare the top-rated noise-canceling headphones under $300 across Amazon, Best Buy, and Target. Summarize the key features (battery life, audio quality, build quality) and include the price for each model. Highlight which model offers the best value for money."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare noise-canceling headphones under $300 across Amazon, Best Buy, and Target. The agent must summarize key features (battery life, audio quality, build quality), include prices, and identify the best value for money. A successful completion includes accurate data from all three platforms, structured output, and a clear value-for-money analysis.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare the top-rated noise-canceling headphones under $300 across Amazon, Best Buy, and Target. Summarize the key features (battery life, audio quality, build quality) and include the price for each model. Highlight which model offers the best value for money.

## Task-Specific Constraints
- Must visit Amazon, Best Buy, and Target to gather data.
- Must include price data for all headphones compared.
- Must summarize key features: battery life, audio quality, and build quality.
- Output must be organized as a structured table or list.
- Must identify and justify which model offers the best value for money.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Best Buy, and Target? Which platforms were actually visited?
- Are price data and key features (battery life, audio quality, build quality) included for all headphones compared?
- Is the output organized as a structured table or list?
- Does the response identify and justify the best value-for-money model?
- Are any factual claims inaccurate or unsupported?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (comparison and value-for-money analysis) is correct and complete.

5 — Includes accurate price and feature data for 4+ headphones, with a justified value-for-money analysis.
4 — Includes accurate data for 3 headphones, with a partially justified analysis.
3 — Includes accurate data for 2 headphones, with minimal analysis.
2 — Includes incomplete or inaccurate data, with no meaningful analysis.
1 — No usable comparison or analysis.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered data from each.

5 — Data from Amazon, Best Buy, and Target is fully utilized.
4 — Data from 2 platforms is fully utilized; 1 platform partially utilized.
3 — Data from 2 platforms is partially utilized.
2 — Data from only 1 platform is utilized.
1 — No platform data is utilized.

#### C. Depth of Comparison (0.20)
Measures the level of detail and specificity in the comparison.

5 — Includes detailed comparisons of battery life, audio quality, and build quality for all headphones.
4 — Includes detailed comparisons for most headphones but lacks depth in one area.
3 — Includes basic comparisons but lacks depth or specificity.
2 — Includes minimal comparisons with significant omissions.
1 — No meaningful comparisons.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and based on credible sources.

5 — Output is structured as a clear table or list, with credible sourcing.
4 — Output is structured but lacks clarity or sourcing.
3 — Output is minimally structured and partially credible.
2 — Output is poorly structured or lacks credibility.
1 — Output is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
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
    "depth_of_comparison": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())