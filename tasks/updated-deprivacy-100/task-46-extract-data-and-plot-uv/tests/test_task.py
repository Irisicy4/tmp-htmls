"""
LLM-as-judge evaluator for EvolveBench task-46.

Category: Data & ML Engineering
Task: Extract data and plot the UV (Unique Visitors) data for the past 5 years.
"""

import os, json, re

TASK_INSTRUCTION = "Extract data and plot the UV (Unique Visitors) data for the past 5 years."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully found, extracted, and visualized 5-year UV (Unique Visitors) metric data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Data: UV (Unique Visitors) or equivalent web traffic metric — must span approximately 5 years
- Source: data must come from a credible source (analytics platform, public report, or authoritative website)
- Output: a chart/plot must be produced (HTML/JS chart, image file, or equivalent visualization)
- Coverage: data points should cover at least 4 of the 5 most recent years

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for and find UV/traffic data? From what source?
- What time range does the data cover? Is it approximately 5 years?
- Was a chart or visualization produced? What type?
- Are the data points specific (actual numbers) or vague estimates?
- Was a file saved or an interactive chart generated?

### Step 2: Dimension Scoring

#### A. Data Source Quality
Did the agent find credible UV data?

5 — Data sourced from a credible analytics platform, official report, or authoritative site with specific yearly figures.
4 — Data sourced from a credible source but some values are estimated or interpolated.
3 — Data found from a secondary source (blog, article citing stats); numbers present but sourcing is weak.
2 — Agent described what UV data looks like without finding actual numbers.
1 — No data found or data is clearly fabricated.

#### B. Temporal Coverage
Does the data span approximately 5 years?

5 — Data covers 5 years with data points for each year (or equivalent granularity).
4 — Data covers 4 years or has one gap year.
3 — Data covers 3 years.
2 — Data covers fewer than 3 years or only shows a trend without specific years.
1 — No temporal data provided.

#### C. Visualization Quality
Was a chart produced and is it clear?

5 — Clear chart produced (HTML/JS interactive chart or image) with labeled axes, title, and data points.
4 — Chart produced but missing labels or title.
3 — Chart attempted but poorly formatted or hard to read.
2 — Data presented as a table or list instead of a chart.
1 — No visualization produced.

#### D. Report Completeness
Did the agent deliver a complete, useful output?

5 — Full report with data source, extracted numbers, and chart — saved as file or clearly presented.
4 — Report present but one element missing (e.g. no source citation or no saved file).
3 — Partial output — either data or chart present but not both.
2 — Only a description of the approach without actual data or chart.
1 — No useful output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_source_quality": <1-5>,
  "temporal_coverage": <1-5>,
  "visualization_quality": <1-5>,
  "report_completeness": <1-5>,
  "dimension_reasoning": {{
    "data_source_quality": "<one sentence citing specific evidence>",
    "temporal_coverage": "<one sentence citing specific evidence>",
    "visualization_quality": "<one sentence citing specific evidence>",
    "report_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_source_quality":   0.25,
    "temporal_coverage":     0.25,
    "visualization_quality": 0.30,
    "report_completeness":   0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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