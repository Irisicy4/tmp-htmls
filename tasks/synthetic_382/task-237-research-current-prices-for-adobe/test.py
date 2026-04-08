"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Research and compare annual costs for Adobe Premiere Pro, Final Cut Pro, and DaVinci Resolve licenses, recommending the most cost-effective option.
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


TASK_INSTRUCTION = """Research current prices for Adobe Premiere Pro, Final Cut Pro, and DaVinci Resolve licenses. Calculate the total annual cost for a creator using each tool, including any subscription fees. Recommend the most cost-effective option based on the findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research current prices for Adobe Premiere Pro, Final Cut Pro, and DaVinci Resolve licenses, calculate the total annual cost for each tool, and recommend the most cost-effective option. This task is in the domain of media software pricing research, and successful completion requires accurate price data, correct calculations, and a clear recommendation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research current prices for Adobe Premiere Pro, Final Cut Pro, and DaVinci Resolve licenses. Calculate the total annual cost for a creator using each tool, including any subscription fees. Recommend the most cost-effective option based on the findings.

## Task-Specific Constraints
- Must visit adobe.com, apple.com, and blackmagicdesign.com to gather pricing information.
- Must include price data for Adobe Premiere Pro, Final Cut Pro, and DaVinci Resolve.
- Must calculate annual costs correctly, including subscription fees if applicable.
- Output must be organized as a table or structured list.
- Must provide a clear recommendation for the most cost-effective option.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are price data for Adobe Premiere Pro, Final Cut Pro, and DaVinci Resolve present in the response?
- Are the annual costs calculated correctly for each tool?
- Is the output organized as a table or structured list?
- Does the agent provide a clear recommendation for the most cost-effective option?

### Step 2: Dimension Scoring

#### A. Pricing Accuracy (0.35)
Measures whether the agent correctly identified and reported the prices for all three tools.

5 — Prices for all three tools are correct and sourced from the specified platforms.
4 — Prices for all three tools are mostly correct but with minor inaccuracies.
3 — Prices for at least two tools are correct; one may be missing or inaccurate.
2 — Prices for only one tool are correct or mostly incorrect.
1 — Prices for all tools are missing or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered data from them.

5 — Agent visited adobe.com, apple.com, and blackmagicdesign.com and sourced data from all.
4 — Agent visited at least two platforms and sourced data from them.
3 — Agent visited at least one platform and sourced partial data.
2 — Agent visited platforms but did not source relevant data.
1 — Agent did not visit any of the required platforms.

#### C. Calculation Accuracy (0.25)
Measures whether the agent correctly calculated the annual costs based on the pricing data.

5 — Annual costs are calculated correctly for all tools, including subscription fees.
4 — Annual costs are mostly correct but with minor errors.
3 — Annual costs are partially correct; one tool may be missing or incorrect.
2 — Annual costs are mostly incorrect or missing for multiple tools.
1 — Annual costs are completely incorrect or missing.

#### D. Output Structure and Recommendation (0.10)
Measures whether the output is well-organized and includes a clear recommendation.

5 — Output is organized as a table or structured list and includes a clear recommendation.
4 — Output is mostly organized and includes a recommendation with minor issues.
3 — Output is partially organized and includes a vague recommendation.
2 — Output is disorganized and lacks a clear recommendation.
1 — Output is completely disorganized and missing a recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "pricing_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "calculation_accuracy": <1-5>,
  "output_structure_and_recommendation": <1-5>,
  "dimension_reasoning": {{
    "pricing_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "calculation_accuracy": "<one sentence citing specific evidence>",
    "output_structure_and_recommendation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "pricing_accuracy": 0.35,
    "platform_coverage": 0.30,
    "calculation_accuracy": 0.25,
    "output_structure_and_recommendation": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())