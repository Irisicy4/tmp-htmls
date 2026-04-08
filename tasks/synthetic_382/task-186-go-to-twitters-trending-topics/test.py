"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Extract top 5 trending topics or videos with engagement metrics from Twitter, Reddit, and YouTube.
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


TASK_INSTRUCTION = """Go to Twitter's trending topics page, Reddit's 'r/AskReddit' popular posts section, and YouTube's trending tab. Extract the top 5 trending topics or videos from each platform with engagement metrics (e.g., likes, responses, or views)."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting the top 5 trending topics or videos from Twitter, Reddit, and YouTube, along with engagement metrics such as likes, responses, or views. This task is in the domain of media analysis, and successful completion requires structured output containing the required data for all three platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Twitter's trending topics page, Reddit's 'r/AskReddit' popular posts section, and YouTube's trending tab. Extract the top 5 trending topics or videos from each platform with engagement metrics (e.g., likes, responses, or views).

## Task-Specific Constraints
- Must visit Twitter, Reddit, and YouTube as specified.
- Must extract exactly 5 items from each platform.
- Must include engagement metrics (e.g., likes, responses, views) for each item.
- Output must be organized as a structured list or table.
- Must correctly identify trending sections/pages on each platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are 5 items extracted from each platform?
- Are engagement metrics (e.g., likes, responses, views) present for each item?
- Is the output organized as a structured list or table?
- Are the extracted items clearly from the trending sections/pages of each platform?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent extracted the correct items and engagement metrics.

5 — Extracts 5 items from each platform with accurate engagement metrics.
4 — Extracts 5 items from each platform but some metrics are incomplete or inaccurate.
3 — Extracts fewer than 5 items or engagement metrics are partially missing.
2 — Extracts fewer than 3 items or metrics are mostly missing/inaccurate.
1 — Fails to extract any items or metrics.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and extracted data.

5 — Visits all three platforms and extracts data from each.
4 — Visits all three platforms but extracts incomplete data from one.
3 — Visits only two platforms or extracts incomplete data from two.
2 — Visits only one platform or extracts minimal data.
1 — Fails to visit any platform or extract data.

#### C. Depth of Engagement Metrics (0.20)
Measures the specificity and detail of engagement metrics provided.

5 — Provides detailed metrics (e.g., likes, responses, views) for all items.
4 — Provides detailed metrics for most items but some are missing.
3 — Provides basic metrics but lacks detail for many items.
2 — Provides minimal or incorrect metrics.
1 — Fails to provide any metrics.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list with credible sources.
4 — Output is structured but lacks clarity or some sources are questionable.
3 — Output is partially structured or sources are unclear.
2 — Output is poorly structured or sources are mostly missing.
1 — Output is unstructured or sources are absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_engagement_metrics": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_engagement_metrics": "<one sentence citing specific evidence>",
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
    "depth_of_engagement_metrics": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())