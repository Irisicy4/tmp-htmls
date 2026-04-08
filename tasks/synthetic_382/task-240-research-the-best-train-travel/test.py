"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Compare train travel options between Milan and Zurich based on price, duration, and amenities.
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


TASK_INSTRUCTION = """Research the best train travel options for a scenic journey between Milan, Italy, and Zurich, Switzerland. Compare the trains on ticket price, travel duration, and onboard amenities across multiple operators."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare train travel options between Milan, Italy, and Zurich, Switzerland. The agent must evaluate ticket prices, travel durations, and onboard amenities across multiple operators. A successful completion includes a structured comparison of these factors, sourced from at least three specified platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research the best train travel options for a scenic journey between Milan, Italy, and Zurich, Switzerland. Compare the trains on ticket price, travel duration, and onboard amenities across multiple operators.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: bahn.com, trenitalia.com, sbb.ch.
- Must include ticket price, travel duration, and onboard amenities for all comparisons.
- Output must be organized as a structured table or list.
- Must identify the most scenic option and justify the choice.
- Must provide data for at least 3 different train operators.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response include ticket price, travel duration, and onboard amenities for all comparisons?
- Is the output organized as a structured table or list?
- Does the response identify the most scenic option and justify the choice?
- Are the comparisons accurate and sourced from credible platforms?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the agent's comparisons of ticket price, travel duration, and onboard amenities are accurate and complete.

5 — All comparisons are accurate, complete, and sourced from credible platforms.
4 — Most comparisons are accurate and complete, with minor omissions or errors.
3 — Comparisons are partially complete but lack some key details or have notable errors.
2 — Comparisons are mostly inaccurate or incomplete.
1 — No meaningful comparisons provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and utilized the required platforms.

5 — Agent used all 3 specified platforms and cited them in the response.
4 — Agent used 2 of the specified platforms and cited them in the response.
3 — Agent used 1 of the specified platforms or cited incomplete data.
2 — Agent attempted platform usage but failed to retrieve meaningful data.
1 — No platform usage evident.

#### C. Detail and Specificity (0.20)
Measures the depth of information provided, including specific numbers and justifications.

5 — Includes detailed ticket prices, durations, amenities, and a justified scenic recommendation.
4 — Includes most details but lacks depth in one area (e.g., amenities or justification).
3 — Includes some details but lacks depth in multiple areas.
2 — Includes minimal details or vague comparisons.
1 — No meaningful details provided.

#### D. Output Structure and Clarity (0.15)
Measures whether the response is well-organized and easy to understand.

5 — Output is structured as a clear table or list with all required elements.
4 — Output is mostly clear but has minor formatting issues.
3 — Output is partially clear but lacks organization or key elements.
2 — Output is poorly structured or difficult to follow.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
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
    "comparison_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())