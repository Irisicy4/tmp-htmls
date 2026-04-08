"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Estimate the total cost of a 7-day trip to Bali, including flights, mid-range accommodation, daily meals, and site entrance fees.
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


TASK_INSTRUCTION = """Estimate the total cost of a 7-day trip to Bali, including flights, mid-range accommodation, daily meals, and site entrance fees. Gather flight prices from Skyscanner, accommodation rates from Agoda, and meal/site cost estimates from Nomadic Matt’s travel blog."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to estimate the total cost of a 7-day trip to Bali, including flights, mid-range accommodation, daily meals, and site entrance fees. It involves gathering data from Skyscanner, Agoda, and Nomadic Matt’s travel blog. A successful completion requires accurate cost estimates from these sources, organized in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Estimate the total cost of a 7-day trip to Bali, including flights, mid-range accommodation, daily meals, and site entrance fees. Gather flight prices from Skyscanner, accommodation rates from Agoda, and meal/site cost estimates from Nomadic Matt’s travel blog.

## Task-Specific Constraints
- Must visit Skyscanner, Agoda, and Nomadic Matt’s travel blog.
- Must include price data for flights, accommodation, meals, and site entrance fees.
- Output must be organized as a structured table or list.
- Must provide specific numerical estimates for each cost category.
- Must ensure data sources are credible and relevant to Bali travel.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Skyscanner, Agoda, and Nomadic Matt’s travel blog? Which ones were actually visited?
- Are flight, accommodation, meal, and site entrance fee costs present in the response?
- Is the output organized as a structured table or list?
- Are numerical estimates for each cost category accurate and sourced?
- Are the data sources credible and relevant to Bali travel?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the total cost estimate is correct and complete.

5 — Includes accurate and complete cost estimates for all categories (flights, accommodation, meals, site fees).
4 — Includes cost estimates for all categories but with minor inaccuracies.
3 — Includes cost estimates for most categories but with noticeable inaccuracies.
2 — Includes cost estimates for few categories or major inaccuracies.
1 — No accurate cost estimates provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent used all required platforms and sources.

5 — Uses Skyscanner, Agoda, and Nomadic Matt’s blog with relevant data extracted.
4 — Uses all required platforms but misses minor data points.
3 — Uses at least two required platforms with partial data extraction.
2 — Uses only one platform or extracts irrelevant data.
1 — Does not use any required platform.

#### C. Specificity of Estimates (0.25)
Measures the level of detail in the cost breakdown.

5 — Provides detailed numerical estimates for each category with breakdowns.
4 — Provides numerical estimates for each category but lacks detailed breakdowns.
3 — Provides partial numerical estimates with limited detail.
2 — Provides vague or incomplete numerical estimates.
1 — Provides no numerical estimates.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list; sources are credible and relevant.
4 — Output is mostly structured but has minor formatting issues; sources are credible.
3 — Output is partially structured; sources are somewhat credible.
2 — Output is poorly structured; sources are not credible.
1 — Output is unstructured and sources are irrelevant.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "specificity_of_estimates": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "specificity_of_estimates": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "specificity_of_estimates": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())