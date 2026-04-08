"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Evaluate whether the agent successfully fetched benchmark results for three image classification models, calculated efficiency, and recommended the most efficient model based on accuracy per training parameter count.
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


TASK_INSTRUCTION = """Fetch the latest benchmark results from Papers With Code for three leading image classification models (ResNet-50, EfficientNet-B3, and Vision Transformer) on the ImageNet dataset. Calculate the model with the highest accuracy per training parameter count and recommend the most efficient model with supporting evidence."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves fetching benchmark results for three specified image classification models (ResNet-50, EfficientNet-B3, and Vision Transformer) from Papers With Code and calculating their efficiency based on accuracy per training parameter count. The agent must recommend the most efficient model with supporting evidence.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch the latest benchmark results from Papers With Code for three leading image classification models (ResNet-50, EfficientNet-B3, and Vision Transformer) on the ImageNet dataset. Calculate the model with the highest accuracy per training parameter count and recommend the most efficient model with supporting evidence.

## Task-Specific Constraints
- Must fetch benchmark results for all three specified models.
- Must calculate accuracy per training parameter count for each model.
- Must recommend the most efficient model based on the calculated metric.
- Output must include supporting evidence for the recommendation.
- The response must be structured and clearly organized.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Papers With Code and fetch benchmark results for all three specified models?
- Are the accuracy and training parameter counts for each model present in the response?
- Did the agent calculate the efficiency metric (accuracy per training parameter count) correctly?
- Is the recommendation for the most efficient model supported with evidence?
- Is the output structured and clearly organized?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly calculated the efficiency metric and recommended the most efficient model.

5 — Efficiency metric calculated correctly for all three models; recommendation is accurate and supported.
4 — Efficiency metric calculated for all models but recommendation has minor errors or lacks full support.
3 — Efficiency metric calculated for at least two models; recommendation is partially correct.
2 — Efficiency metric calculated for one model or recommendation is mostly incorrect.
1 — Efficiency metric not calculated or recommendation is completely wrong.

#### B. Coverage of Required Models (0.30)
Measures whether benchmark results for all three specified models were fetched and included.

5 — Results for all three models are fetched and included.
4 — Results for two models are fetched and included.
3 — Results for at least one model are fetched and included.
2 — Results are mostly missing or incorrect.
1 — No results fetched or included.

#### C. Depth of Analysis (0.25)
Measures the detail and specificity of the analysis, including accuracy values, parameter counts, and efficiency comparisons.

5 — Includes detailed accuracy values, parameter counts, and efficiency comparisons for all models.
4 — Includes most details but lacks minor elements or comparisons.
3 — Includes basic details but lacks depth or specificity.
2 — Includes very few details or comparisons.
1 — No details or comparisons included.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and evidence is credible.

5 — Output is clearly structured, well-organized, and evidence is credible.
4 — Output is mostly well-organized but has minor structural issues.
3 — Output is usable but lacks clarity or organization.
2 — Output is poorly organized or evidence credibility is questionable.
1 — Output is disorganized or evidence is not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_models": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_models": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_models": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())