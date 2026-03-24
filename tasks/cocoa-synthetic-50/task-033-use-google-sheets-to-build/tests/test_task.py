"""
LLM-as-judge evaluator for EvolveBench task.

Category: Marketing & Analytics
Task: Build a marketing performance dashboard in Google Sheets using live data from three recent campaigns, including charts and ROI calculations.
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


TASK_INSTRUCTION = """Use Google Sheets to build a basic marketing performance dashboard. Pull live data for three recent campaigns from public examples or marketing blogs (e.g., campaign spend, clicks, and conversions). Create charts for each metric and calculate the ROI (return on investment) for each campaign."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to build a marketing performance dashboard in Google Sheets. The agent must gather live data for three recent campaigns (e.g., campaign spend, clicks, and conversions) from public examples or marketing blogs. The deliverable must include charts for each metric and ROI calculations for each campaign.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to build a basic marketing performance dashboard. Pull live data for three recent campaigns from public examples or marketing blogs (e.g., campaign spend, clicks, and conversions). Create charts for each metric and calculate the ROI (return on investment) for each campaign.

## Task-Specific Constraints
- Must use Google Sheets to create the dashboard.
- Must gather live data for three distinct campaigns from public examples or marketing blogs.
- Must include charts for campaign spend, clicks, and conversions.
- Must calculate ROI for each campaign using the formula: (Revenue - Spend) / Spend.
- Output must be structured and visually clear in the Google Sheets file.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms (Google Sheets, marketing blogs, etc.)? Which ones were actually visited?
- Did the agent gather live data for three distinct campaigns? Are the campaigns clearly identified?
- Are the required metrics (spend, clicks, conversions) present for each campaign?
- Are charts for each metric included in the Google Sheets file?
- Is ROI calculated correctly for each campaign using the provided formula?

### Step 2: Dimension Scoring

#### A. Dashboard Accuracy (0.35)
Measures whether the marketing performance dashboard is accurate, complete, and meets the task requirements.

5 — All required metrics (spend, clicks, conversions) and ROI are accurately calculated for three campaigns.
4 — Metrics and ROI are mostly accurate but contain minor errors or omissions.
3 — Metrics and ROI are partially accurate but incomplete or with significant errors.
2 — Metrics and ROI are mostly incorrect or missing.
1 — Metrics and ROI are entirely absent or incorrect.

#### B. Data Coverage (0.30)
Measures whether the agent gathered live data for three distinct campaigns from appropriate sources.

5 — Data for three distinct campaigns is gathered from credible public examples or blogs.
4 — Data for three campaigns is gathered but with minor credibility or source issues.
3 — Data for two campaigns is gathered, or sources are unclear.
2 — Data for only one campaign is gathered, or sources are unreliable.
1 — No campaign data is gathered.

#### C. Chart Quality (0.20)
Measures whether the charts in the Google Sheets file are clear, relevant, and correctly represent the data.

5 — Charts for all required metrics are present, clear, and correctly represent the data.
4 — Charts are mostly clear and correct but with minor issues.
3 — Charts are partially present or unclear but usable.
2 — Charts are mostly missing or incorrect.
1 — Charts are entirely absent or incorrect.

#### D. Output Structure (0.15)
Measures whether the final output (Google Sheets file) is well-organized and visually clear.

5 — The output is highly organized, visually clear, and easy to interpret.
4 — The output is mostly organized and clear but with minor formatting issues.
3 — The output is partially organized but usable.
2 — The output is poorly organized or difficult to interpret.
1 — The output is entirely disorganized or unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dashboard_accuracy": <1-5>,
  "data_coverage": <1-5>,
  "chart_quality": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "dashboard_accuracy": "<one sentence citing specific evidence>",
    "data_coverage": "<one sentence citing specific evidence>",
    "chart_quality": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "dashboard_accuracy": 0.35,
    "data_coverage": 0.30,
    "chart_quality": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())