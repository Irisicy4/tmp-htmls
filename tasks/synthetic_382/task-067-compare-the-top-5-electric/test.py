"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Compare the top 5 electric kettles under $50 from Amazon, Walmart, and Target, and summarize findings in a table.
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


TASK_INSTRUCTION = """Compare the top 5 electric kettles for under $50 from Amazon, Walmart, and Target. Evaluate their features, including capacity, heating speed, safety mechanisms, and user reviews. Summarize the findings in a table."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare the top 5 electric kettles under $50 from Amazon, Walmart, and Target. The agent must evaluate their features, including capacity, heating speed, safety mechanisms, and user reviews, and summarize the findings in a table. A successful completion includes accurate data from all three platforms, a well-structured table, and coverage of all specified features.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare the top 5 electric kettles for under $50 from Amazon, Walmart, and Target. Evaluate their features, including capacity, heating speed, safety mechanisms, and user reviews. Summarize the findings in a table.

## Task-Specific Constraints
- Must visit Amazon, Walmart, and Target to gather data.
- Must include price, capacity, heating speed, safety mechanisms, and user review summaries for each kettle.
- Must compare exactly 5 kettles from each platform.
- Output must be organized as a table with clear labels for each feature.
- Must include a summary of findings highlighting key differences or trends.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Walmart, and Target? Which platforms were actually visited?
- Does the response include data for exactly 5 kettles from each platform?
- Are all required features (price, capacity, heating speed, safety mechanisms, user reviews) present for each kettle?
- Is the output organized as a table with clear labels?
- Are the comparisons accurate and based on credible data?

### Step 2: Dimension Scoring

#### A. Data Completeness (0.35)
Measures whether the agent provided data for all required features for all 15 kettles.

5 — Data for all 15 kettles is complete and includes all required features.
4 — Data for 12-14 kettles is complete, with minor omissions in features.
3 — Data for 9-11 kettles is present, but some features are missing.
2 — Data for fewer than 9 kettles is present, or major omissions in features.
1 — No meaningful data is provided.

#### B. Platform Coverage (0.30)
Measures whether the agent gathered data from all three specified platforms.

5 — Data is gathered from all three platforms with clear evidence.
4 — Data is gathered from two platforms, with minor omissions.
3 — Data is gathered from one platform, or incomplete data from two platforms.
2 — Minimal data from one platform or unclear sources.
1 — No data from any platform.

#### C. Detail and Specificity (0.20)
Measures the depth of the comparisons, including specific numbers and trends.

5 — Comparisons include specific numbers, trends, and meaningful insights.
4 — Comparisons include numbers but lack some trends or insights.
3 — Comparisons are vague but partially meaningful.
2 — Comparisons are mostly absent or lack specificity.
1 — No comparisons are made.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and clearly formatted as a table.

5 — Output is a well-organized table with clear labels and formatting.
4 — Output is a table but with minor formatting issues.
3 — Output is partially structured as a table but unclear in places.
2 — Output is poorly organized and difficult to interpret.
1 — Output is not structured as a table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "data_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_completeness": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())