"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Research and compare 3-5 options for noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart, focusing on features and price differences.
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


TASK_INSTRUCTION = """Research and compare 3-5 options for noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart. Focus on features like battery life, Bluetooth version, active noise-canceling quality, and customer reviews. Summarize the key features and price differences in a table."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare 3-5 options for noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart. The agent must focus on features like battery life, Bluetooth version, active noise-canceling quality, and customer reviews. A successful completion requires summarizing the key features and price differences in a table.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare 3-5 options for noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart. Focus on features like battery life, Bluetooth version, active noise-canceling quality, and customer reviews. Summarize the key features and price differences in a table.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (Amazon, Best Buy, Walmart).
- Must include price data for all items compared.
- Output must be organized as a table.
- Must address battery life, Bluetooth version, active noise-canceling quality, and customer reviews.
- Must compare at least 3 options and no more than 5.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are price data and feature comparisons present for all items?
- Is the output organized as a table?
- Are battery life, Bluetooth version, active noise-canceling quality, and customer reviews addressed?
- Are the claims accurate and sourced from credible platforms?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (table of comparisons) is correct and complete.

5 — Table includes 3-5 options with accurate price, battery life, Bluetooth version, noise-canceling quality, and customer reviews.
4 — Table includes 3-5 options but has minor inaccuracies or omissions in 1-2 features.
3 — Table includes 3-5 options but lacks detail or has significant inaccuracies in 3+ features.
2 — Table includes fewer than 3 options or is mostly inaccurate.
1 — No table or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and sourced data.

5 — Agent visited Amazon, Best Buy, and Walmart and sourced data from all.
4 — Agent visited 2 platforms and sourced data from them.
3 — Agent visited 1 platform and sourced data from it.
2 — Agent visited platforms but did not source usable data.
1 — No platforms visited or no data sourced.

#### C. Feature Depth (0.25)
Measures the depth and specificity of the feature comparisons.

5 — Includes detailed comparisons for all required features (battery life, Bluetooth version, noise-canceling quality, customer reviews).
4 — Includes detailed comparisons for 3 of the required features.
3 — Includes comparisons for 2 of the required features but lacks depth.
2 — Includes comparisons for 1 feature or is mostly vague.
1 — No feature comparisons present.

#### D. Output Structure (0.10)
Measures whether the output is well-organized and easy to interpret.

5 — Output is a clear, well-organized table with labeled columns and rows.
4 — Output is a table but has minor formatting issues.
3 — Output is a table but lacks clarity or organization.
2 — Output is not a table but has some structure.
1 — Output is unstructured or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "feature_depth": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "feature_depth": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "feature_depth": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())