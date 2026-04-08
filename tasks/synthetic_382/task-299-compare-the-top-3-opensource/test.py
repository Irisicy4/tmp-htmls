"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Compare and evaluate the top 3 open-source ML libraries for time-series forecasting based on specific criteria.
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


TASK_INSTRUCTION = """Compare the top 3 open-source ML libraries for time-series forecasting: Facebook Prophet, Darts, and ARIMA implementations. Focus on their feature set, ease of use, active community size, and performance benchmarks cited in recent papers or blogs. Summarize your findings in a table format."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare the top 3 open-source ML libraries for time-series forecasting: Facebook Prophet, Darts, and ARIMA implementations. The agent must evaluate these libraries based on their feature set, ease of use, active community size, and performance benchmarks cited in recent papers or blogs. A successful completion includes a structured table summarizing the findings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare the top 3 open-source ML libraries for time-series forecasting: Facebook Prophet, Darts, and ARIMA implementations. Focus on their feature set, ease of use, active community size, and performance benchmarks cited in recent papers or blogs. Summarize your findings in a table format.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: github.com, medium.com, paperswithcode.com.
- Must include a comparison of feature sets, ease of use, community size, and performance benchmarks.
- Output must be organized in a table format.
- Must cite at least one recent paper or blog for performance benchmarks.
- Must provide specific details (e.g., number of contributors, GitHub stars, or specific features).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response include a comparison of feature sets, ease of use, community size, and performance benchmarks?
- Is the output organized in a table format as required?
- Are specific details (e.g., GitHub stars, contributors, features) included in the comparison?
- Are performance benchmarks cited from at least one recent paper or blog?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparison of the libraries is accurate and complete.

5 — Includes accurate and detailed comparisons for all 4 criteria (feature set, ease of use, community size, performance benchmarks).
4 — Includes accurate comparisons for 3 criteria; minor omissions or inaccuracies.
3 — Includes partial comparisons for 2-3 criteria; lacks detail or has notable inaccuracies.
2 — Includes minimal comparisons; significant inaccuracies or omissions.
1 — No meaningful comparison provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited the required platforms and used them effectively.

5 — Visited all 3 platforms (github.com, medium.com, paperswithcode.com) and extracted relevant data.
4 — Visited 2 platforms and extracted relevant data.
3 — Visited at least 1 platform and extracted some relevant data.
2 — Visited platforms but extracted minimal or irrelevant data.
1 — Did not visit any required platforms.

#### C. Depth of Analysis (0.20)
Measures the level of detail and specificity in the response.

5 — Provides highly detailed comparisons, including quantitative data (e.g., GitHub stars, contributors, performance metrics).
4 — Provides detailed comparisons but lacks some quantitative data.
3 — Provides basic comparisons with minimal quantitative data.
2 — Provides vague or incomplete comparisons.
1 — Provides no meaningful details.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and cites credible sources.

5 — Output is well-organized in a table format and cites credible sources (e.g., recent papers or blogs).
4 — Output is mostly well-organized and cites credible sources.
3 — Output is organized but lacks credible citations or has minor formatting issues.
2 — Output is poorly organized or lacks credible citations.
1 — Output is disorganized and lacks credible citations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())