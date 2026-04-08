"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Diagnose and resolve PyTorch-to-ONNX compatibility issues during model migration.
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


TASK_INSTRUCTION = """A machine learning engineer reports that migrating a PyTorch model trained with `torch==1.13` to `torch==2.0` causes runtime errors with ONNX export. Diagnose the compatibility issue by researching GitHub issues, PyTorch release notes, and ONNX documentation. Provide a diagnosis note with the root cause and recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task involves diagnosing compatibility issues between PyTorch and ONNX during model migration. The agent must research relevant documentation, GitHub issues, and release notes to identify the root cause and propose a fix. Successful completion requires a detailed diagnosis note with actionable recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A machine learning engineer reports that migrating a PyTorch model trained with `torch==1.13` to `torch==2.0` causes runtime errors with ONNX export. Diagnose the compatibility issue by researching GitHub issues, PyTorch release notes, and ONNX documentation. Provide a diagnosis note with the root cause and recommended fix.

## Task-Specific Constraints
- Must visit at least 2 of the specified platforms (pytorch.org, github.com, onnx.ai).
- Must identify the root cause of the compatibility issue based on credible sources.
- Must provide a clear and actionable recommended fix.
- Output must be structured as a diagnosis note with clear sections (e.g., "Root Cause", "Recommendation").
- Must reference specific documentation, GitHub issues, or release notes in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least 2 of the required platforms? Which ones were visited?
- Does the response identify the root cause of the compatibility issue?
- Is the recommendation actionable and clearly explained?
- Are specific references to documentation, GitHub issues, or release notes included?
- Is the output organized as a structured diagnosis note?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the root cause and provided an actionable fix.

5 — Identifies the exact root cause and provides a clear, actionable fix with references.
4 — Identifies the root cause and provides a mostly actionable fix, minor gaps in clarity.
3 — Identifies the root cause but fix is incomplete or unclear.
2 — Attempts to identify the root cause but is mostly incorrect or vague.
1 — Fails to identify the root cause or provide a fix.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent visited the required platforms and referenced credible sources.

5 — References all 3 platforms (pytorch.org, github.com, onnx.ai) with specific details.
4 — References 2 platforms with specific details.
3 — References at least 2 platforms but lacks specificity.
2 — References only 1 platform or is vague about sources.
1 — Fails to reference any credible sources.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in the diagnosis and recommendation.

5 — Provides detailed analysis with specific examples, numbers, or comparisons.
4 — Provides detailed analysis but lacks minor specifics.
3 — Provides basic analysis but lacks depth or examples.
2 — Provides shallow analysis with significant gaps.
1 — Provides no meaningful analysis.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and references credible sources.

5 — Organized as a structured diagnosis note with credible references.
4 — Mostly organized but minor formatting issues or unclear references.
3 — Basic organization but lacks structure or credibility.
2 — Poorly organized and lacks credible references.
1 — Completely disorganized and lacks references.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
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
    "coverage_of_required_sources": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())