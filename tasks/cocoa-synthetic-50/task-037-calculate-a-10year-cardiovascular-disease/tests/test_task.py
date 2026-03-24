"""
LLM-as-judge evaluator for EvolveBench task.

Category: Medical & Clinical & Bio
Task: Evaluate whether the agent correctly calculates and compares 10-year cardiovascular disease risk scores using three specified online tools and recommends the most appropriate one based on U.S.-specific guidelines.
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


TASK_INSTRUCTION = """Calculate a 10-year cardiovascular disease risk score for a hypothetical 55-year-old male living in the U.S. with the following health metrics: total cholesterol of 220 mg/dL, HDL cholesterol of 45 mg/dL, a systolic blood pressure of 140 mmHg (not treated with medication), and a smoking history of 10 years. Use the ASCVD Risk Calculator tool (http://tools.acc.org/ascvd-risk-estimator-plus). Then cross-check the calculated risk percentage with two other online risk calculators: the QRISK3 from https://qrisk.org/ and the Framingham Risk Score tool at https://www.mdcalc.com/. Document and compare the outputs from all three tools, and recommend one of them based on closest alignment with U.S.-specific guidelines."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate a 10-year cardiovascular disease risk score for a hypothetical patient using three specific online tools, compare the results, and recommend the most appropriate tool based on U.S.-specific guidelines. The domain is medical and clinical, and a successful completion requires accurate calculations, proper use of the specified platforms, and a clear recommendation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate a 10-year cardiovascular disease risk score for a hypothetical 55-year-old male living in the U.S. with the following health metrics: total cholesterol of 220 mg/dL, HDL cholesterol of 45 mg/dL, a systolic blood pressure of 140 mmHg (not treated with medication), and a smoking history of 10 years. Use the ASCVD Risk Calculator tool (http://tools.acc.org/ascvd-risk-estimator-plus). Then cross-check the calculated risk percentage with two other online risk calculators: the QRISK3 from https://qrisk.org/ and the Framingham Risk Score tool at https://www.mdcalc.com/. Document and compare the outputs from all three tools, and recommend one of them based on closest alignment with U.S.-specific guidelines.

## Task-Specific Constraints
- Must visit all three specified platforms: tools.acc.org, qrisk.org, and mdcalc.com.
- Must provide the calculated risk scores from each platform.
- Must compare the results from all three tools in a structured format.
- Must recommend one tool based on alignment with U.S.-specific guidelines.
- Must ensure that the recommendation is justified with evidence from the task.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms? Which ones were actually visited?
- Are the calculated risk scores from all three tools included in the response?
- Is the comparison of results structured and easy to follow?
- Does the recommendation align with U.S.-specific guidelines, and is it justified with evidence?
- Are there any factual inaccuracies in the calculations or comparisons?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly calculated and documented the risk scores from all three tools.

5 — All three tools used, and all risk scores are correctly calculated and documented.
4 — All three tools used, but one risk score has minor inaccuracies.
3 — At least two tools used, with mostly correct calculations.
2 — Only one tool used, or significant inaccuracies in calculations.
1 — No tools used, or all calculations are incorrect.

#### B. Coverage of Requirements (0.30)
Measures whether the agent fulfilled all task-specific constraints.

5 — All task-specific constraints are fully satisfied.
4 — All major constraints are satisfied, but one minor detail is missing.
3 — Most constraints are satisfied, but one major requirement is incomplete.
2 — Few constraints are satisfied, with multiple major omissions.
1 — None of the constraints are satisfied.

#### C. Depth and Specificity (0.20)
Measures the level of detail in the comparisons and the justification for the recommendation.

5 — Comparisons are detailed, and the recommendation is thoroughly justified with evidence.
4 — Comparisons are clear but lack some detail; recommendation is justified.
3 — Comparisons are present but minimal; recommendation is vague.
2 — Comparisons are unclear or missing; recommendation is unsupported.
1 — No comparisons or recommendation provided.

#### D. Output Structure and Clarity (0.15)
Measures the organization and readability of the response.

5 — Response is well-organized, with a clear structure and no ambiguities.
4 — Response is mostly well-organized, with minor issues in clarity.
3 — Response is somewhat organized but has notable clarity issues.
2 — Response is poorly organized and difficult to follow.
1 — Response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())