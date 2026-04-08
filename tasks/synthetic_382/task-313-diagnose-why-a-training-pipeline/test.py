"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Diagnose why a training pipeline for fine-tuning BERT on custom text data fails when using PyTorch Lightning, and recommend a fix.
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


TASK_INSTRUCTION = """Diagnose why a training pipeline for fine-tuning BERT on custom text data fails when using PyTorch Lightning. Investigate error logs on Stack Overflow, GitHub issues, and the official PyTorch Lightning documentation, and summarize the root cause and recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires diagnosing the failure of a training pipeline for fine-tuning BERT using PyTorch Lightning. The agent must investigate error logs on Stack Overflow, GitHub issues, and the official PyTorch Lightning documentation. A successful completion includes identifying the root cause and providing a recommended fix.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a training pipeline for fine-tuning BERT on custom text data fails when using PyTorch Lightning. Investigate error logs on Stack Overflow, GitHub issues, and the official PyTorch Lightning documentation, and summarize the root cause and recommended fix.

## Task-Specific Constraints
- Must visit Stack Overflow, GitHub, and the official PyTorch Lightning documentation.
- Must identify the root cause of the pipeline failure.
- Must provide a recommended fix that is actionable and specific.
- Output must include references to the investigated sources.
- Must summarize findings in a clear and structured format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Stack Overflow, GitHub, and the official PyTorch Lightning documentation?
- Does the response identify the root cause of the pipeline failure?
- Is the recommended fix actionable and specific?
- Are references to the investigated sources included in the response?
- Is the output structured clearly and logically?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent correctly identifies the root cause of the pipeline failure.

5 — Clearly identifies the root cause with supporting evidence from multiple sources.
4 — Identifies the root cause but lacks supporting evidence from all required sources.
3 — Partially identifies the root cause with minimal evidence.
2 — Incorrect or vague identification of the root cause.
1 — Does not identify the root cause.

#### B. Recommended Fix Quality (0.30)
Measures the quality and specificity of the recommended fix.

5 — Provides a specific, actionable fix with detailed steps.
4 — Provides a fix that is actionable but lacks detail.
3 — Provides a fix that is partially actionable or vague.
2 — Provides a fix that is mostly incorrect or not actionable.
1 — Does not provide a fix.

#### C. Source Coverage (0.20)
Measures whether the agent uses all required platforms and references them in the response.

5 — Investigates and references all required platforms (Stack Overflow, GitHub, PyTorch Lightning documentation).
4 — Investigates most platforms but misses one or more references.
3 — Investigates some platforms but misses key references.
2 — Investigates only one platform or provides no references.
1 — Does not investigate any platforms.

#### D. Output Structure and Clarity (0.15)
Measures the organization and clarity of the response.

5 — Response is well-structured, clear, and logically organized.
4 — Response is mostly clear but has minor structural issues.
3 — Response is partially clear but lacks logical organization.
2 — Response is unclear or poorly structured.
1 — Response is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "recommended_fix_quality": <1-5>,
  "source_coverage": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "recommended_fix_quality": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "recommended_fix_quality": 0.30,
    "source_coverage": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())