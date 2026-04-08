"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Investigate and resolve library incompatibility between 'transformers' and 'torch>=2.0.0'.
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


TASK_INSTRUCTION = """A user reports that installing the 'transformers' library alongside 'torch>=2.0.0' causes runtime errors during tokenization. Search PyPI, GitHub issues, and StackOverflow for the exact cause of this incompatibility, including the affected version ranges and recommended resolution path."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task involves investigating a library incompatibility issue between 'transformers' and 'torch>=2.0.0'. The agent must identify the root cause, affected version ranges, and provide a resolution path. Success requires accurate identification of the issue and actionable recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that installing the 'transformers' library alongside 'torch>=2.0.0' causes runtime errors during tokenization. Search PyPI, GitHub issues, and StackOverflow for the exact cause of this incompatibility, including the affected version ranges and recommended resolution path.

## Task-Specific Constraints
- Must visit PyPI, GitHub, and StackOverflow to gather evidence.
- Must identify specific version ranges of 'transformers' and 'torch' causing the issue.
- Must provide a clear resolution path (e.g., downgrade, upgrade, or patch).
- Must include references or links to credible sources for claims.
- Output must be structured as a summary with actionable steps.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to PyPI, GitHub, and StackOverflow? Which platforms were actually visited?
- Are the affected version ranges of 'transformers' and 'torch' identified?
- Is the resolution path clear and actionable?
- Are references or links to credible sources included in the response?
- Is the output organized as a clear summary with actionable steps?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the incompatibility and resolution.

5 — Identifies the exact cause, affected version ranges, and provides a clear resolution path.
4 — Identifies the cause and resolution but misses minor details (e.g., partial version ranges).
3 — Identifies the cause but resolution is unclear or incomplete.
2 — Incorrect or vague identification of the cause; resolution is missing.
1 — No meaningful attempt to identify the cause or resolution.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and gathered evidence.

5 — Gathers evidence from PyPI, GitHub, and StackOverflow with relevant links.
4 — Gathers evidence from at least two platforms with relevant links.
3 — Gathers evidence from one platform or provides incomplete links.
2 — Minimal evidence gathered; links missing or irrelevant.
1 — No evidence gathered from required platforms.

#### C. Depth and Specificity (0.20)
Measures the level of detail in the response, including version ranges and actionable steps.

5 — Provides detailed version ranges, specific steps, and supporting evidence.
4 — Provides version ranges and steps but lacks minor details.
3 — Provides general steps but lacks specificity or version ranges.
2 — Steps are vague and lack supporting evidence.
1 — No actionable steps or supporting evidence.

#### D. Source Quality and Output Structure (0.15)
Measures the credibility of sources and organization of the output.

5 — Sources are credible and output is well-structured as a summary with actionable steps.
4 — Sources are credible but output structure is slightly unclear.
3 — Sources are partially credible or output structure is disorganized.
2 — Sources lack credibility or output is poorly structured.
1 — No credible sources or structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_quality_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_quality_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity": 0.20,
    "source_quality_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())