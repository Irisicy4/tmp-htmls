"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Research and compare train travel options from Paris to Venice based on cost, duration, departure time flexibility, and onboard amenities.
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


TASK_INSTRUCTION = """Research and compare three major train travel options from Paris to Venice. Look at train providers such as SNCF, Trenitalia, and Rail Europe. Compare them based on cost, duration, departure time flexibility, and onboard amenities."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare train travel options from Paris to Venice using three specified platforms: SNCF, Trenitalia, and Rail Europe. A successful completion requires the agent to provide a structured comparison based on cost, duration, departure time flexibility, and onboard amenities.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three major train travel options from Paris to Venice. Look at train providers such as SNCF, Trenitalia, and Rail Europe. Compare them based on cost, duration, departure time flexibility, and onboard amenities.

## Task-Specific Constraints
- Must visit all three specified platforms: SNCF, Trenitalia, and Rail Europe.
- Must include cost, duration, departure time flexibility, and onboard amenities for each option.
- Output must be organized as a structured table or list.
- Must provide specific numerical comparisons (e.g., exact prices, durations).
- Must cite sources or provide evidence for claims made.
- Must identify any missing data or limitations in the comparison.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are cost, duration, departure time flexibility, and onboard amenities present for all options?
- Is the output organized as a structured table or list?
- Are specific numerical comparisons (e.g., prices, durations) included and accurate?
- Are sources cited or evidence provided for claims made?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparison includes correct and complete data for cost, duration, departure time flexibility, and onboard amenities.

5 — Includes all required data points for all three platforms, with no errors.
4 — Includes most required data points, with minor inaccuracies.
3 — Includes partial data points, with some missing or incorrect.
2 — Includes minimal data points, with significant errors or omissions.
1 — No meaningful comparison provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all specified platforms and used them to gather information.

5 — Successfully visited and used all three platforms.
4 — Visited two platforms and used them effectively.
3 — Visited at least one platform and used it partially.
2 — Attempted but failed to gather meaningful data from platforms.
1 — Did not visit any specified platforms.

#### C. Detail and Specificity (0.20)
Measures the depth of the comparison, including numerical data and specific details.

5 — Provides detailed numerical comparisons and specific amenities for all options.
4 — Provides numerical comparisons and some specific details for most options.
3 — Provides partial numerical comparisons with limited details.
2 — Provides minimal numerical comparisons and lacks specific details.
1 — No numerical comparisons or specific details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list, with credible sources cited.
4 — Output is mostly structured, with minor issues in organization or sourcing.
3 — Output is partially structured, with some missing citations or unclear formatting.
2 — Output is poorly structured, with significant issues in organization or sourcing.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
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
    "detail_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())