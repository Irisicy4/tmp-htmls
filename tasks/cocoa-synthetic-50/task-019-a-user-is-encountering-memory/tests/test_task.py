"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Diagnose and resolve memory errors when fine-tuning BERT-large on an NVIDIA 3090 GPU using PyTorch.
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


TASK_INSTRUCTION = """A user is encountering memory errors when fine-tuning BERT-large with a batch size of 32 on an NVIDIA 3090 GPU using PyTorch. Research PyTorch forums, GitHub issues, and the NVIDIA documentation to identify the root cause. Provide a diagnosis of why the error occurs, specify affected configurations, and recommend possible solutions such as reducing batch size or using gradient checkpointing."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing memory errors encountered during fine-tuning BERT-large on an NVIDIA 3090 GPU using PyTorch. The agent must research forums, GitHub issues, and NVIDIA documentation to identify the root cause, specify affected configurations, and recommend solutions such as reducing batch size or using gradient checkpointing. A successful completion includes a clear diagnosis, accurate identification of affected configurations, and actionable recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user is encountering memory errors when fine-tuning BERT-large with a batch size of 32 on an NVIDIA 3090 GPU using PyTorch. Research PyTorch forums, GitHub issues, and the NVIDIA documentation to identify the root cause. Provide a diagnosis of why the error occurs, specify affected configurations, and recommend possible solutions such as reducing batch size or using gradient checkpointing.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: discuss.pytorch.org, github.com, developer.nvidia.com.
- Must provide a clear diagnosis of the memory error cause.
- Must specify configurations affected (e.g., batch size, model size, GPU memory limits).
- Must recommend actionable solutions (e.g., reducing batch size, gradient checkpointing).
- Output must be structured and easy to follow.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response include a clear diagnosis of the memory error cause?
- Are affected configurations (e.g., batch size, GPU memory) identified accurately?
- Are actionable recommendations provided (e.g., reducing batch size, gradient checkpointing)?
- Is the output structured and easy to follow?

### Step 2: Dimension Scoring

#### A. Diagnosis Accuracy (0.35)
Measures whether the agent correctly identifies the root cause of the memory error.

5 — Provides a precise and accurate diagnosis of the memory error cause.
4 — Diagnosis is mostly accurate but lacks minor details.
3 — Diagnosis is partially correct but incomplete or unclear.
2 — Diagnosis is mostly incorrect or vague.
1 — No diagnosis provided or completely wrong.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited the required platforms and used them effectively.

5 — Visited all 3 specified platforms and used them effectively.
4 — Visited 2 platforms and used them effectively.
3 — Visited at least 1 platform and used it effectively.
2 — Visited platforms but did not use them effectively.
1 — Did not visit any of the required platforms.

#### C. Recommendations Quality (0.25)
Measures the quality and feasibility of the recommendations provided.

5 — Provides multiple actionable and feasible recommendations.
4 — Provides at least one actionable and feasible recommendation.
3 — Provides recommendations, but they are incomplete or lack feasibility.
2 — Recommendations are mostly incorrect or impractical.
1 — No recommendations provided or completely wrong.

#### D. Output Structure and Clarity (0.10)
Measures how well the response is structured and easy to follow.

5 — Output is well-structured, clear, and easy to follow.
4 — Output is mostly clear but could be better structured.
3 — Output is somewhat clear but lacks structure.
2 — Output is poorly structured or unclear.
1 — Output is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "diagnosis_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "recommendations_quality": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "diagnosis_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "recommendations_quality": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "diagnosis_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "recommendations_quality": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())