"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Identify top 3 trending sans-serif fonts from specified platforms, determine their licenses and costs, and recommend the most cost-effective option.
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


TASK_INSTRUCTION = """Find the top 3 trending sans-serif fonts and their licenses from Font Squirrel and Google Fonts. Then, calculate the total cost of purchasing these fonts (if applicable) for commercial use, and recommend the most cost-effective option."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to identify the top 3 trending sans-serif fonts from Font Squirrel and Google Fonts, determine their licenses, calculate their total cost for commercial use, and recommend the most cost-effective option. This is a Design task requiring accurate data collection, cost analysis, and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find the top 3 trending sans-serif fonts and their licenses from Font Squirrel and Google Fonts. Then, calculate the total cost of purchasing these fonts (if applicable) for commercial use, and recommend the most cost-effective option.

## Task-Specific Constraints
- Must visit both Font Squirrel and Google Fonts to gather data.
- Must identify exactly 3 sans-serif fonts from each platform.
- Must include license details for all fonts.
- Must calculate and include total costs for commercial use (if applicable).
- Must recommend the most cost-effective option based on the data.
- Output must be structured as a table or clearly organized list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both Font Squirrel and Google Fonts? Were other platforms used unnecessarily?
- Did the agent identify exactly 3 sans-serif fonts from each platform? Were they labeled as trending?
- Are license details included for all fonts? Are they accurate?
- Are costs for commercial use calculated and included for all fonts?
- Is the output structured as a table or clearly organized list?

### Step 2: Dimension Scoring

#### A. Font Identification Accuracy (0.35)
Measures whether the agent correctly identified the top 3 trending sans-serif fonts from each platform.

5 — Correctly identifies 3 trending sans-serif fonts from both platforms.
4 — Identifies 3 fonts from both platforms but not all are trending.
3 — Identifies fewer than 3 fonts or includes incorrect ones.
2 — Identifies fewer than 2 fonts or includes unrelated fonts.
1 — Fails to identify any relevant fonts.

#### B. License and Cost Accuracy (0.30)
Measures whether the agent accurately includes license details and cost calculations for all fonts.

5 — Includes accurate license and cost details for all fonts.
4 — Includes mostly accurate license and cost details, with minor errors.
3 — Includes partial or incomplete license and cost details.
2 — Includes significant errors or omissions in license or cost details.
1 — Fails to include any license or cost details.

#### C. Recommendation Quality (0.20)
Measures whether the agent provides a clear and logical recommendation based on cost-effectiveness.

5 — Provides a clear, logical recommendation based on accurate data.
4 — Provides a reasonable recommendation with minor logical flaws.
3 — Provides a recommendation but lacks clarity or supporting data.
2 — Provides an unclear or unsupported recommendation.
1 — Fails to provide any recommendation.

#### D. Output Structure and Organization (0.15)
Measures whether the agent's output is well-structured and easy to follow.

5 — Output is well-organized, with a clear table or structured list.
4 — Output is mostly organized but could be clearer.
3 — Output is somewhat disorganized but still usable.
2 — Output is poorly organized or difficult to follow.
1 — Output is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "font_identification_accuracy": <1-5>,
  "license_and_cost_accuracy": <1-5>,
  "recommendation_quality": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "font_identification_accuracy": "<one sentence citing specific evidence>",
    "license_and_cost_accuracy": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "font_identification_accuracy": 0.35,
    "license_and_cost_accuracy": 0.30,
    "recommendation_quality": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())