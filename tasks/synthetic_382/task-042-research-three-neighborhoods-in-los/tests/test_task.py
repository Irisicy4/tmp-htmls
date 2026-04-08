"""
LLM-as-judge evaluator for EvolveBench task.

Category: Real Estate
Task: Compare three neighborhoods in Los Angeles, CA based on home prices, walkability, and school ratings using Zillow, Niche, and Redfin.
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


TASK_INSTRUCTION = """Research three neighborhoods in Los Angeles, CA — Silver Lake, Culver City, and Downtown LA — on Zillow (https://www.zillow.com/), Niche (https://www.niche.com/), and Redfin (https://www.redfin.com/). Compare these neighborhoods based on average home price for 3-bedroom properties, walkability score, and school ratings for K-12. Create a comparison table with columns: Neighborhood, Average 3-Bedroom Home Price, Walkability Score, Best Rated K-12 School, and School Rating."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research three neighborhoods in Los Angeles, CA — Silver Lake, Culver City, and Downtown LA — using Zillow, Niche, and Redfin. The agent must collect data on average home price for 3-bedroom properties, walkability score, and school ratings for K-12. A successful completion includes a structured comparison table with all required data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research three neighborhoods in Los Angeles, CA — Silver Lake, Culver City, and Downtown LA — on Zillow (https://www.zillow.com/), Niche (https://www.niche.com/), and Redfin (https://www.redfin.com/). Compare these neighborhoods based on average home price for 3-bedroom properties, walkability score, and school ratings for K-12. Create a comparison table with columns: Neighborhood, Average 3-Bedroom Home Price, Walkability Score, Best Rated K-12 School, and School Rating.

## Task-Specific Constraints
- Must visit Zillow, Niche, and Redfin to gather data.
- Must include average home price for 3-bedroom properties for all neighborhoods.
- Must include walkability scores for all neighborhoods.
- Must include the best-rated K-12 school and its rating for each neighborhood.
- Output must be organized as a structured table with all required columns.
- Data must be accurate and sourced from the specified platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Zillow, Niche, and Redfin? Which platforms were actually visited?
- Are average home prices for 3-bedroom properties present for all neighborhoods?
- Are walkability scores included for all neighborhoods?
- Are the best-rated K-12 schools and their ratings included for all neighborhoods?
- Is the output organized as a structured table with the required columns?

### Step 2: Dimension Scoring

#### A. Data Accuracy and Completeness (0.35)
Measures whether the agent provided correct and complete data for all required fields.

5 — All data is correct and complete for all neighborhoods and fields.
4 — Minor inaccuracies or missing data in one field or neighborhood.
3 — Partial data provided; some neighborhoods or fields are incomplete.
2 — Significant inaccuracies or missing data in multiple fields or neighborhoods.
1 — No usable data provided.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms (Zillow, Niche, Redfin) to gather data.

5 — Data sourced from all three platforms for all neighborhoods.
4 — Data sourced from two platforms; minor omissions from the third.
3 — Data sourced from one platform; significant omissions from others.
2 — Minimal platform usage; most data missing.
1 — No evidence of platform usage.

#### C. Depth and Specificity (0.20)
Measures whether the agent provided detailed comparisons, including specific numbers and named schools.

5 — Includes specific numbers and named schools for all neighborhoods.
4 — Minor omissions or lack of specificity in one field.
3 — Partial specificity; some fields lack detailed comparisons.
2 — Minimal specificity; most fields are vague or missing.
1 — No specificity or comparisons provided.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-organized and presented as a structured table.

5 — Output is fully organized as a table with all required columns.
4 — Output is mostly organized; minor formatting issues.
3 — Output is partially organized; significant formatting issues.
2 — Output is poorly organized; hard to interpret.
1 — Output is unstructured or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy_and_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy_and_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy_and_completeness": 0.35,
    "platform_coverage": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())