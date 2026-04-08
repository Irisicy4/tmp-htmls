"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Compare three models of wireless noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart, and recommend the best option for everyday use.
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


TASK_INSTRUCTION = """Research and compare three models of wireless noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart. Summarize their features, including battery life, connectivity options, and customer ratings, and recommend the best option for everyday use based on your findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research and compare three models of wireless noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart. The agent must summarize their features, including battery life, connectivity options, and customer ratings, and recommend the best option for everyday use based on its findings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three models of wireless noise-canceling headphones under $150 across Amazon, Best Buy, and Walmart. Summarize their features, including battery life, connectivity options, and customer ratings, and recommend the best option for everyday use based on your findings.

## Task-Specific Constraints
- Must visit Amazon, Best Buy, and Walmart to gather data.
- Must include price data for all three headphone models compared.
- Must summarize features including battery life, connectivity options, and customer ratings.
- Output must be organized as a structured list or table.
- Must provide a clear recommendation for the best option based on the findings.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Best Buy, and Walmart? Which platforms were actually visited?
- Are price data for all three headphone models included in the response?
- Are features such as battery life, connectivity options, and customer ratings summarized for each model?
- Is the output organized as a structured list or table?
- Does the agent provide a clear recommendation based on its findings?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's summary and recommendation are correct and complete.

5 — Includes accurate summaries of all three headphone models, with correct prices, features, and ratings, and provides a clear recommendation.
4 — Summaries are mostly accurate but may have minor errors or omissions; recommendation is reasonable.
3 — Summaries are partially complete or contain notable errors; recommendation is unclear or weak.
2 — Summaries are mostly missing or incorrect; recommendation is absent or nonsensical.
1 — No summaries or recommendation provided.

#### B. Platform Coverage (0.30)
Measures whether the agent gathered data from all required platforms.

5 — Data from Amazon, Best Buy, and Walmart is included and correctly attributed.
4 — Data from at least two platforms is included; minor attribution issues.
3 — Data from only one platform is included or attribution is unclear.
2 — No platform data or incorrect platforms used.
1 — No attempt to gather platform data.

#### C. Feature Specificity (0.25)
Measures the depth and specificity of the feature comparisons.

5 — Includes detailed comparisons of battery life, connectivity options, and customer ratings for all models.
4 — Comparisons include most features but lack some detail or specificity.
3 — Comparisons are shallow or incomplete; some features are missing.
2 — Comparisons are mostly absent or vague.
1 — No feature comparisons provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list; sources are credible and well-referenced.
4 — Output is mostly well-organized; minor issues with structure or sourcing.
3 — Output is usable but disorganized; sources are unclear.
2 — Output is poorly organized or lacks credibility.
1 — No structure or credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "feature_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "feature_specificity": "<one sentence citing specific evidence>",
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
    "feature_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())