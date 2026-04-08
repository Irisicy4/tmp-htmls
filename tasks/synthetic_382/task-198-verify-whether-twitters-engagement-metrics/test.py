"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether Twitter’s engagement metrics for hashtags #AI and #MachineLearning have decreased this month compared to last month.
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


TASK_INSTRUCTION = """Verify whether Twitter’s engagement metrics for hashtags #AI and #MachineLearning have decreased this month compared to last month. Use public analytics available on Twitter's Trends page and third-party tools like Brand24 and Keyhole to check for visible fluctuations in impressions and engagement rates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to analyze engagement metrics for hashtags #AI and #MachineLearning across Twitter and third-party analytics platforms. Successful completion involves verifying whether impressions and engagement rates have decreased this month compared to last month using data from Twitter Trends, Brand24, and Keyhole.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Twitter’s engagement metrics for hashtags #AI and #MachineLearning have decreased this month compared to last month. Use public analytics available on Twitter's Trends page and third-party tools like Brand24 and Keyhole to check for visible fluctuations in impressions and engagement rates.

## Task-Specific Constraints
- Must visit Twitter Trends, Brand24, and Keyhole platforms.
- Must extract impressions and engagement rate data for both hashtags (#AI and #MachineLearning).
- Must compare metrics between this month and last month.
- Output must include structured data (e.g., table or list) showing the comparison.
- Must provide clear conclusions about whether metrics have decreased.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Twitter Trends, Brand24, and Keyhole? Which platforms were actually visited?
- Are impressions and engagement rate data for both hashtags (#AI and #MachineLearning) present in the response?
- Is the comparison between this month and last month clearly shown?
- Is the output organized as a table or structured list?
- Are conclusions about metric changes clearly stated and supported by evidence?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and compared engagement metrics.

5 — Includes accurate impressions and engagement rate data for both hashtags, with correct conclusions.
4 — Includes most required data but with minor inaccuracies or incomplete conclusions.
3 — Includes partial data or unclear conclusions but shows effort to compare metrics.
2 — Includes minimal or incorrect data, with no meaningful comparison.
1 — No data or completely incorrect conclusions.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them appropriately.

5 — Successfully visited Twitter Trends, Brand24, and Keyhole, extracting relevant data from all.
4 — Visited at least two platforms and extracted most relevant data.
3 — Visited one platform and extracted partial data.
2 — Attempted platform visits but failed to extract meaningful data.
1 — Did not visit any required platforms.

#### C. Depth of Analysis (0.25)
Measures the level of detail in the comparison and conclusions.

5 — Provides detailed comparisons with specific numerical data and trends for both hashtags.
4 — Provides comparisons with some numerical data but lacks depth in trends or analysis.
3 — Provides basic comparisons with limited numerical data or trends.
2 — Provides minimal or vague comparisons with little numerical data.
1 — No meaningful analysis or comparison.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and uses credible sources.

5 — Output is structured as a table or list, with clear sourcing and organization.
4 — Output is mostly structured but lacks clarity or sourcing.
3 — Output is partially structured but disorganized or unclear.
2 — Output is poorly structured and lacks credibility.
1 — Output is completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
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
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())