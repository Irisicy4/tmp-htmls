"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Extract monthly traffic, engagement metrics, and demographics for the top 3 social media platforms globally and organize the data into a CSV file.
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


TASK_INSTRUCTION = """Go to Social Blade, SimilarWeb, and Statista. Extract the monthly traffic data, engagement metrics, and demographics for the top 3 social media platforms globally (e.g., Facebook, Instagram, TikTok). Organize the extracted data into a CSV file with clear labeling."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit Social Blade, SimilarWeb, and Statista to extract monthly traffic data, engagement metrics, and demographics for the top 3 social media platforms globally (e.g., Facebook, Instagram, TikTok). The extracted data must be organized into a CSV file with clear labeling.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Social Blade, SimilarWeb, and Statista. Extract the monthly traffic data, engagement metrics, and demographics for the top 3 social media platforms globally (e.g., Facebook, Instagram, TikTok). Organize the extracted data into a CSV file with clear labeling.

## Task-Specific Constraints
- Must visit Social Blade, SimilarWeb, and Statista.
- Must extract monthly traffic data, engagement metrics, and demographics for the top 3 platforms.
- Must include data for Facebook, Instagram, and TikTok (or justify substitutions).
- Output must be in CSV format with clear column headers.
- Data must be specific and sourced from the required platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Social Blade, SimilarWeb, and Statista? Which ones were actually visited?
- Are monthly traffic data, engagement metrics, and demographics for the top 3 platforms included?
- Does the response include data for Facebook, Instagram, and TikTok (or justify substitutions)?
- Is the output organized as a CSV file with clear column headers?
- Are the data points specific, accurate, and sourced from the required platforms?

### Step 2: Dimension Scoring

#### A. Data Completeness (0.35)
Measures whether all required data points (monthly traffic, engagement metrics, demographics) are included for the top 3 platforms.

5 — Includes all required data points for all 3 platforms.
4 — Includes most data points but misses minor details.
3 — Includes some data points but misses major details.
2 — Includes very few data points or is incomplete.
1 — Does not include any required data points.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and extracted data from them.

5 — Visited all 3 platforms and extracted data from each.
4 — Visited 2 platforms and extracted data from them.
3 — Visited 1 platform and extracted data from it.
2 — Visited platforms but extracted little or no data.
1 — Did not visit any required platforms.

#### C. Data Specificity (0.20)
Measures whether the extracted data is specific, accurate, and detailed.

5 — Data is highly specific, accurate, and detailed.
4 — Data is mostly specific and accurate but lacks some detail.
3 — Data is somewhat specific but lacks accuracy or detail.
2 — Data is vague or mostly inaccurate.
1 — Data is entirely missing or incorrect.

#### D. Output Structure (0.15)
Measures whether the output is correctly formatted as a CSV file with clear labeling.

5 — Output is a well-structured CSV file with clear column headers.
4 — Output is mostly well-structured but has minor formatting issues.
3 — Output is partially structured but lacks clarity or proper formatting.
2 — Output is poorly structured or difficult to interpret.
1 — Output is unstructured or entirely missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "data_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "data_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_completeness": 0.35,
    "platform_coverage": 0.30,
    "data_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())