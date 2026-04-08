"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Fetch training cost estimates for ResNet-50, GPT-3, and ViT, calculate GPU hours, and recommend the most cost-effective option.
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


TASK_INSTRUCTION = """Fetch the latest training cost estimates for three popular machine learning models (ResNet-50, GPT-3, ViT) from public sources. Calculate the total GPU hours needed for training each based on published benchmark data and recommend the most cost-effective option for a mid-sized dataset."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to fetch training cost estimates for ResNet-50, GPT-3, and ViT from public sources, calculate GPU hours needed for training each model, and recommend the most cost-effective option for a mid-sized dataset. This task is in the domain of data and ML engineering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch the latest training cost estimates for three popular machine learning models (ResNet-50, GPT-3, ViT) from public sources. Calculate the total GPU hours needed for training each based on published benchmark data and recommend the most cost-effective option for a mid-sized dataset.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (paperswithcode.com, huggingface.co, nvidia.com).
- Must include GPU hour cost data for all three models (ResNet-50, GPT-3, ViT).
- Output must be organized as a structured table or list.
- Must recommend the most cost-effective option based on calculations.
- Must cite sources for all numerical claims.
- Must calculate GPU hours correctly based on benchmark data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are GPU hour cost estimates present for ResNet-50, GPT-3, and ViT?
- Is the output organized as a structured table or list?
- Are numerical claims (e.g., GPU hours) accurate and sourced?
- Did the agent recommend the most cost-effective option based on calculations?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the GPU hour calculations and cost-effective recommendation are correct and complete.

5 — All calculations are accurate, and the recommendation is correct.
4 — Minor errors in calculations or recommendation.
3 — Partial calculations or recommendation; usable but incomplete.
2 — Major errors in calculations or recommendation.
1 — No calculations or recommendation provided.

#### B. Coverage of Platforms and Models (0.30)
Measures whether the agent visited the required platforms and included data for all three models.

5 — All three platforms visited and data for all three models included.
4 — Two platforms visited or data for two models included.
3 — At least one platform visited and data for one model included.
2 — Minimal platform usage or data inclusion.
1 — No platform usage or data inclusion.

#### C. Depth of Analysis (0.20)
Measures the level of detail in calculations and comparisons.

5 — Detailed calculations and comparisons for all models.
4 — Good detail but minor omissions.
3 — Basic calculations and comparisons; lacks depth.
2 — Minimal calculations or comparisons.
1 — No calculations or comparisons.

#### D. Source Credibility and Output Structure (0.15)
Measures whether sources are credible and output is well-organized.

5 — All sources are credible, and output is well-organized.
4 — Minor issues with source credibility or organization.
3 — Acceptable sources and organization; minor flaws.
2 — Poor sources or disorganized output.
1 — No credible sources or disorganized output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms_and_models": <1-5>,
  "depth_of_analysis": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_models": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms_and_models": 0.30,
    "depth_of_analysis": 0.20,
    "source_credibility_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())