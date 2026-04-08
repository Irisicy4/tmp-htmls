"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Compare three travel insurance plans for a 2-week trip to Europe, focusing on coverage for medical expenses, trip cancellations, and personal belongings.
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


TASK_INSTRUCTION = """Compare three travel insurance plans suitable for a 2-week trip to Europe. Focus on coverage for medical expenses, trip cancellations, and personal belongings. Use data from InsureMyTrip, Squaremouth, and TravelInsurance.com."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare three travel insurance plans suitable for a 2-week trip to Europe. The comparison must focus on coverage for medical expenses, trip cancellations, and personal belongings. The agent must use data from InsureMyTrip, Squaremouth, and TravelInsurance.com.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare three travel insurance plans suitable for a 2-week trip to Europe. Focus on coverage for medical expenses, trip cancellations, and personal belongings. Use data from InsureMyTrip, Squaremouth, and TravelInsurance.com.

## Task-Specific Constraints
- Must visit all three specified platforms: InsureMyTrip, Squaremouth, and TravelInsurance.com.
- Must include coverage details for medical expenses, trip cancellations, and personal belongings for each plan.
- Must provide price data for each plan compared.
- Output must be organized as a table or structured list.
- Must include a summary of which plan is most suitable based on the comparison.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms? Which ones were actually visited?
- Are coverage details for medical expenses, trip cancellations, and personal belongings present for each plan?
- Is price data included for all three plans?
- Is the output organized as a table or structured list?
- Does the response include a summary of which plan is most suitable based on the comparison?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures whether the comparison of plans is accurate and complete.

5 — All three plans are compared accurately, with complete details for medical expenses, trip cancellations, and personal belongings.
4 — All three plans are compared, but some details are incomplete or slightly inaccurate.
3 — At least two plans are compared with partial details.
2 — Only one plan is compared or details are mostly incorrect.
1 — No plans are compared or details are completely absent.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms.

5 — All three platforms (InsureMyTrip, Squaremouth, TravelInsurance.com) were used.
4 — Two platforms were used, with relevant data extracted.
3 — At least one platform was used with relevant data extracted.
2 — Platforms were visited but no relevant data was extracted.
1 — No platforms were visited.

#### C. Depth of Comparison (0.20)
Measures the specificity and depth of the comparison.

5 — Includes detailed coverage amounts, exclusions, and price breakdowns for all plans.
4 — Includes coverage amounts and prices but lacks some exclusions or details.
3 — Includes general coverage and price data but lacks depth.
2 — Includes minimal coverage or price data.
1 — No specific details are provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the output.

5 — Output is well-organized (e.g., table or structured list) and sources are clearly cited.
4 — Output is organized but lacks some citations or clarity.
3 — Output is partially organized but difficult to follow.
2 — Output is disorganized or lacks credibility.
1 — Output is completely unstructured or not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
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
    "depth_of_comparison": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())